# -*- coding: utf-8 -*-
"""
Enhanced Batch Processor - Xử lý hàng loạt với progress tracking
VIP Pro MAX feature - Parallel processing như render farm chuyên nghiệp
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import (
    Dict, List, Optional, Callable, Any, Tuple, Generator
)
import json


class TaskStatus(Enum):
    """Trạng thái task"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class BatchStatus(Enum):
    """Trạng thái batch"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskResult:
    """Kết quả của một task"""
    task_id: str
    status: TaskStatus
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: int = 0
    retries: int = 0


@dataclass
class TaskProgress:
    """Progress của một task"""
    task_id: str
    status: TaskStatus
    progress_percent: float = 0.0
    current_step: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class BatchTask:
    """
    Một task trong batch
    """
    id: str
    index: int  # Thứ tự trong batch
    
    # Input
    text: str
    voice_file: Optional[str] = None
    
    # Output
    output_path: Optional[str] = None
    
    # Processing info
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    error: Optional[str] = None
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: int = 0
    
    # Retry
    retry_count: int = 0
    max_retries: int = 3
    
    # Custom settings per task
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Thời gian đã chạy"""
        if self.start_time is None:
            return None
        
        end = self.end_time or datetime.now()
        return end - self.start_time


@dataclass
class BatchProgress:
    """
    Progress tổng của batch
    """
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    current_task_index: int = 0
    
    # Timing
    start_time: Optional[datetime] = None
    estimated_end_time: Optional[datetime] = None
    
    # Speed metrics
    tasks_per_minute: float = 0.0
    average_task_duration_ms: int = 0
    
    # Current processing
    current_task_id: Optional[str] = None
    current_task_text: str = ""
    
    @property
    def progress_percent(self) -> float:
        """Phần trăm hoàn thành"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Thời gian đã chạy"""
        if self.start_time is None:
            return None
        return datetime.now() - self.start_time
    
    @property
    def remaining_time(self) -> Optional[timedelta]:
        """Thời gian còn lại ước tính"""
        if self.estimated_end_time is None:
            return None
        return self.estimated_end_time - datetime.now()


# Type aliases
TaskCallback = Callable[[BatchTask], TaskResult]
ProgressCallback = Callable[[BatchProgress], None]
TaskProgressCallback = Callable[[TaskProgress], None]


class BatchProcessor:
    """
    Enhanced Batch Processor
    Xử lý hàng loạt với:
    - Parallel processing
    - Progress tracking chi tiết
    - Pause/Resume/Cancel
    - Auto retry failed tasks
    - Speed estimation
    """
    
    def __init__(
        self,
        max_workers: int = 2,
        auto_retry: bool = True,
        max_retries: int = 3,
        continue_on_error: bool = True
    ):
        """
        Initialize batch processor
        
        Args:
            max_workers: Số thread xử lý song song
            auto_retry: Tự động retry task lỗi
            max_retries: Số lần retry tối đa
            continue_on_error: Tiếp tục khi có lỗi
        """
        self.max_workers = max_workers
        self.auto_retry = auto_retry
        self.max_retries = max_retries
        self.continue_on_error = continue_on_error
        
        # State
        self._status: BatchStatus = BatchStatus.IDLE
        self._tasks: List[BatchTask] = []
        self._progress = BatchProgress()
        
        # Threading
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        self._cancel_flag = threading.Event()
        
        # Callbacks
        self._on_progress: Optional[ProgressCallback] = None
        self._on_task_progress: Optional[TaskProgressCallback] = None
        self._on_task_complete: Optional[Callable[[BatchTask, TaskResult], None]] = None
        self._on_batch_complete: Optional[Callable[[BatchProgress], None]] = None
        self._on_error: Optional[Callable[[BatchTask, str], None]] = None
        
        # Task processor
        self._task_processor: Optional[TaskCallback] = None
        
        # Metrics
        self._task_durations: List[int] = []
        self._completed_count = 0
    
    @property
    def status(self) -> BatchStatus:
        """Trạng thái hiện tại"""
        return self._status
    
    @property
    def progress(self) -> BatchProgress:
        """Progress hiện tại"""
        return self._progress
    
    @property
    def tasks(self) -> List[BatchTask]:
        """Danh sách tasks"""
        return self._tasks.copy()
    
    def set_callbacks(
        self,
        on_progress: Optional[ProgressCallback] = None,
        on_task_progress: Optional[TaskProgressCallback] = None,
        on_task_complete: Optional[Callable[[BatchTask, TaskResult], None]] = None,
        on_batch_complete: Optional[Callable[[BatchProgress], None]] = None,
        on_error: Optional[Callable[[BatchTask, str], None]] = None
    ) -> None:
        """Set callbacks"""
        self._on_progress = on_progress
        self._on_task_progress = on_task_progress
        self._on_task_complete = on_task_complete
        self._on_batch_complete = on_batch_complete
        self._on_error = on_error
    
    def set_task_processor(self, processor: TaskCallback) -> None:
        """
        Set hàm xử lý task
        
        Args:
            processor: Hàm nhận BatchTask và trả về TaskResult
        """
        self._task_processor = processor
    
    def add_tasks(self, tasks: List[BatchTask]) -> None:
        """Thêm tasks vào queue"""
        with self._lock:
            start_index = len(self._tasks)
            for i, task in enumerate(tasks):
                task.index = start_index + i
                self._tasks.append(task)
            
            self._progress.total_tasks = len(self._tasks)
    
    def add_texts(
        self,
        texts: List[str],
        output_dir: str,
        voice_file: Optional[str] = None,
        naming_pattern: str = "{index}",
        start_index: int = 1,
        index_padding: int = 4
    ) -> None:
        """
        Thêm tasks từ list texts
        
        Args:
            texts: Danh sách text cần TTS
            output_dir: Thư mục output
            voice_file: File voice để clone
            naming_pattern: Pattern đặt tên file
            start_index: Số bắt đầu
            index_padding: Số digit padding
        """
        os.makedirs(output_dir, exist_ok=True)
        
        tasks = []
        for i, text in enumerate(texts):
            task_index = start_index + i
            
            # Generate filename
            filename = naming_pattern.format(
                index=str(task_index).zfill(index_padding),
                text=self._sanitize_filename(text[:30])
            )
            
            if not filename.endswith('.wav'):
                filename += '.wav'
            
            output_path = os.path.join(output_dir, filename)
            
            task = BatchTask(
                id=f"task_{task_index}",
                index=task_index,
                text=text,
                voice_file=voice_file,
                output_path=output_path
            )
            tasks.append(task)
        
        self.add_tasks(tasks)
    
    def _sanitize_filename(self, text: str) -> str:
        """Sanitize text để làm filename"""
        invalid = '<>:"/\\|?*'
        result = text
        for char in invalid:
            result = result.replace(char, '_')
        return result.strip()[:50]
    
    def start(self) -> None:
        """Bắt đầu xử lý batch"""
        if self._status == BatchStatus.RUNNING:
            return
        
        if not self._task_processor:
            raise ValueError("Task processor not set. Call set_task_processor() first.")
        
        if not self._tasks:
            raise ValueError("No tasks to process. Call add_tasks() first.")
        
        self._status = BatchStatus.RUNNING
        self._cancel_flag.clear()
        self._pause_event.set()
        
        # Reset progress
        self._progress = BatchProgress(
            total_tasks=len(self._tasks),
            start_time=datetime.now()
        )
        
        # Start processing thread
        thread = threading.Thread(target=self._run_batch, daemon=True)
        thread.start()
    
    def pause(self) -> None:
        """Tạm dừng batch"""
        if self._status == BatchStatus.RUNNING:
            self._pause_event.clear()
            self._status = BatchStatus.PAUSED
            self._notify_progress()
    
    def resume(self) -> None:
        """Tiếp tục batch"""
        if self._status == BatchStatus.PAUSED:
            self._pause_event.set()
            self._status = BatchStatus.RUNNING
            self._notify_progress()
    
    def cancel(self) -> None:
        """Hủy batch"""
        self._cancel_flag.set()
        self._pause_event.set()  # Release pause if paused
        
        # Cancel pending futures
        for future in self._futures.values():
            future.cancel()
        
        # Mark remaining tasks as cancelled
        with self._lock:
            for task in self._tasks:
                if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
                    task.status = TaskStatus.CANCELLED
                    self._progress.cancelled_tasks += 1
        
        self._status = BatchStatus.CANCELLED
        self._notify_progress()
    
    def retry_failed(self) -> None:
        """Retry tất cả tasks thất bại"""
        with self._lock:
            for task in self._tasks:
                if task.status == TaskStatus.FAILED:
                    task.status = TaskStatus.PENDING
                    task.retry_count = 0
                    task.error = None
                    self._progress.failed_tasks -= 1
        
        if self._status != BatchStatus.RUNNING:
            self.start()
    
    def get_failed_tasks(self) -> List[BatchTask]:
        """Lấy danh sách tasks thất bại"""
        return [t for t in self._tasks if t.status == TaskStatus.FAILED]
    
    def get_completed_tasks(self) -> List[BatchTask]:
        """Lấy danh sách tasks hoàn thành"""
        return [t for t in self._tasks if t.status == TaskStatus.COMPLETED]
    
    def clear(self) -> None:
        """Clear tất cả tasks"""
        self.cancel()
        self._tasks.clear()
        self._progress = BatchProgress()
        self._task_durations.clear()
        self._completed_count = 0
        self._status = BatchStatus.IDLE
    
    def _run_batch(self) -> None:
        """Main processing loop"""
        try:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self._futures.clear()
            
            # Submit all tasks
            for task in self._tasks:
                if self._cancel_flag.is_set():
                    break
                
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.QUEUED
                    future = self._executor.submit(self._process_task, task)
                    self._futures[task.id] = future
            
            # Wait for completion
            for future in as_completed(self._futures.values()):
                if self._cancel_flag.is_set():
                    break
                
                # Handle pause
                self._pause_event.wait()
            
            # Final status
            if self._cancel_flag.is_set():
                self._status = BatchStatus.CANCELLED
            elif self._progress.failed_tasks > 0 and not self.continue_on_error:
                self._status = BatchStatus.FAILED
            else:
                self._status = BatchStatus.COMPLETED
            
            # Callback
            if self._on_batch_complete:
                self._on_batch_complete(self._progress)
            
        except Exception as e:
            self._status = BatchStatus.FAILED
            print(f"Batch processing error: {e}")
        finally:
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None
    
    def _process_task(self, task: BatchTask) -> TaskResult:
        """Xử lý một task"""
        # Wait if paused
        self._pause_event.wait()
        
        if self._cancel_flag.is_set():
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.CANCELLED
            )
        
        task.status = TaskStatus.PROCESSING
        task.start_time = datetime.now()
        
        # Update progress
        with self._lock:
            self._progress.current_task_id = task.id
            self._progress.current_task_index = task.index
            self._progress.current_task_text = task.text[:50]
        
        self._notify_task_progress(TaskProgress(
            task_id=task.id,
            status=TaskStatus.PROCESSING,
            progress_percent=0.0,
            current_step="Starting...",
            start_time=task.start_time
        ))
        
        try:
            # Call the actual processor
            result = self._task_processor(task)
            
            task.end_time = datetime.now()
            task.duration_ms = int(
                (task.end_time - task.start_time).total_seconds() * 1000
            )
            
            if result.status == TaskStatus.COMPLETED:
                task.status = TaskStatus.COMPLETED
                task.output_path = result.output_path
                
                with self._lock:
                    self._progress.completed_tasks += 1
                    self._task_durations.append(task.duration_ms)
                    self._update_metrics()
                
                if self._on_task_complete:
                    self._on_task_complete(task, result)
            else:
                self._handle_task_failure(task, result.error_message or "Unknown error")
                
        except Exception as e:
            task.end_time = datetime.now()
            self._handle_task_failure(task, str(e))
            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
        
        self._notify_progress()
        return result
    
    def _handle_task_failure(self, task: BatchTask, error: str) -> None:
        """Xử lý task thất bại"""
        task.error = error
        
        # Check retry
        if self.auto_retry and task.retry_count < self.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            
            # Re-submit
            if self._executor:
                future = self._executor.submit(self._process_task, task)
                self._futures[task.id] = future
        else:
            task.status = TaskStatus.FAILED
            
            with self._lock:
                self._progress.failed_tasks += 1
            
            if self._on_error:
                self._on_error(task, error)
            
            if not self.continue_on_error:
                self.cancel()
    
    def _update_metrics(self) -> None:
        """Cập nhật metrics"""
        if not self._task_durations:
            return
        
        # Average duration
        avg_duration = sum(self._task_durations) / len(self._task_durations)
        self._progress.average_task_duration_ms = int(avg_duration)
        
        # Tasks per minute
        if self._progress.start_time:
            elapsed = (datetime.now() - self._progress.start_time).total_seconds()
            if elapsed > 0:
                self._progress.tasks_per_minute = (
                    self._progress.completed_tasks / elapsed * 60
                )
        
        # Estimate remaining time
        remaining = self._progress.total_tasks - self._progress.completed_tasks
        if remaining > 0 and self._progress.tasks_per_minute > 0:
            remaining_minutes = remaining / self._progress.tasks_per_minute
            self._progress.estimated_end_time = (
                datetime.now() + timedelta(minutes=remaining_minutes)
            )
    
    def _notify_progress(self) -> None:
        """Notify progress callback"""
        if self._on_progress:
            try:
                self._on_progress(self._progress)
            except Exception:
                pass
    
    def _notify_task_progress(self, progress: TaskProgress) -> None:
        """Notify task progress callback"""
        if self._on_task_progress:
            try:
                self._on_task_progress(progress)
            except Exception:
                pass
    
    def export_report(self, output_path: str) -> None:
        """
        Export báo cáo batch processing
        
        Args:
            output_path: Đường dẫn file report
        """
        report = {
            "batch_status": self._status.value,
            "summary": {
                "total_tasks": self._progress.total_tasks,
                "completed": self._progress.completed_tasks,
                "failed": self._progress.failed_tasks,
                "cancelled": self._progress.cancelled_tasks,
                "success_rate": (
                    self._progress.completed_tasks / self._progress.total_tasks * 100
                    if self._progress.total_tasks > 0 else 0
                )
            },
            "timing": {
                "start_time": (
                    self._progress.start_time.isoformat() 
                    if self._progress.start_time else None
                ),
                "total_duration_seconds": (
                    self._progress.elapsed_time.total_seconds()
                    if self._progress.elapsed_time else 0
                ),
                "average_task_ms": self._progress.average_task_duration_ms,
                "tasks_per_minute": round(self._progress.tasks_per_minute, 2)
            },
            "tasks": [
                {
                    "id": task.id,
                    "index": task.index,
                    "text": task.text[:100],
                    "status": task.status.value,
                    "output_path": task.output_path,
                    "duration_ms": task.duration_ms,
                    "error": task.error,
                    "retries": task.retry_count
                }
                for task in self._tasks
            ],
            "failed_tasks": [
                {
                    "id": task.id,
                    "index": task.index,
                    "text": task.text,
                    "error": task.error
                }
                for task in self.get_failed_tasks()
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


class SRTBatchProcessor(BatchProcessor):
    """
    Specialized processor for SRT files
    Kế thừa BatchProcessor với các tính năng đặc biệt cho SRT
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_srt_file: Optional[str] = None
    
    def load_srt(
        self,
        srt_path: str,
        output_dir: str,
        voice_file: Optional[str] = None,
        index_padding: int = 4
    ) -> int:
        """
        Load file SRT và tạo tasks
        
        Args:
            srt_path: Đường dẫn file SRT
            output_dir: Thư mục output
            voice_file: Voice file để clone
            index_padding: Số digit padding cho index
            
        Returns:
            Số lượng tasks đã tạo
        """
        self._current_srt_file = srt_path
        
        # Parse SRT
        subtitles = self._parse_srt(srt_path)
        
        if not subtitles:
            return 0
        
        # Create tasks
        texts = [sub['text'] for sub in subtitles]
        self.add_texts(
            texts=texts,
            output_dir=output_dir,
            voice_file=voice_file,
            naming_pattern="{index}",
            start_index=1,
            index_padding=index_padding
        )
        
        return len(texts)
    
    def _parse_srt(self, srt_path: str) -> List[Dict[str, Any]]:
        """Parse file SRT"""
        subtitles = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(srt_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
            except Exception:
                return []
        
        # Split by blank lines
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0].strip())
                    timing = lines[1].strip()
                    text = ' '.join(lines[2:]).strip()
                    
                    # Parse timing
                    start, end = timing.split(' --> ')
                    
                    subtitles.append({
                        'index': index,
                        'start': start.strip(),
                        'end': end.strip(),
                        'text': text
                    })
                except (ValueError, IndexError):
                    continue
        
        return subtitles


# Singleton instance
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor(reset: bool = False, **kwargs) -> BatchProcessor:
    """Get singleton batch processor"""
    global _batch_processor
    
    if _batch_processor is None or reset:
        _batch_processor = BatchProcessor(**kwargs)
    
    return _batch_processor
