"""
Background Worker for VietTTS
Handles TTS processing in background thread
"""

import os
import threading
import queue
from typing import Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class TaskType(Enum):
    """Types of background tasks"""
    LOAD_MODEL = "load_model"
    SYNTHESIZE = "synthesize"
    SYNTHESIZE_BATCH = "synthesize_batch"
    DOWNLOAD_VOICES = "download_voices"


@dataclass
class Task:
    """Background task definition"""
    task_type: TaskType
    params: dict
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None


class TTSWorker:
    """
    Background worker for TTS operations
    Runs TTS tasks in a separate thread to keep UI responsive
    """
    
    def __init__(self, tts_engine=None):
        """
        Initialize worker
        
        Args:
            tts_engine: TTSEngine instance
        """
        self.tts_engine = tts_engine
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        self._worker_thread = None
        self._running = False
        self._cancel_flag = False
        self._current_task = None
        
        # Callbacks
        self.on_progress: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    def start(self):
        """Start the worker thread"""
        if self._running:
            return
        
        self._running = True
        self._cancel_flag = False
        
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self._worker_thread.start()
        logger.info("TTS Worker started")
    
    def stop(self):
        """Stop the worker thread"""
        self._running = False
        self._cancel_flag = True
        
        # Clear pending tasks
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        
        # Add sentinel to unblock queue
        self.task_queue.put(None)
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        
        logger.info("TTS Worker stopped")
    
    def cancel_current(self):
        """Cancel the current task"""
        self._cancel_flag = True
        logger.info("Current task cancelled")
    
    def is_busy(self) -> bool:
        """Check if worker is processing a task"""
        return self._current_task is not None
    
    def submit(self, task: Task):
        """
        Submit a task to the worker
        
        Args:
            task: Task to execute
        """
        self.task_queue.put(task)
    
    def submit_synthesize(
        self,
        text: str,
        voice_path: str,
        output_path: str,
        speed: float = 1.0,
        ref_text: str = "",
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None
    ):
        """
        Submit a synthesis task
        
        Args:
            text: Text to synthesize
            voice_path: Path to voice sample
            output_path: Path to save output
            speed: Speech speed
            ref_text: Transcript of reference audio (for voice cloning)
            callback: Success callback(output_path)
            error_callback: Error callback(error)
        """
        task = Task(
            task_type=TaskType.SYNTHESIZE,
            params={
                "text": text,
                "voice_path": voice_path,
                "output_path": output_path,
                "speed": speed,
                "ref_text": ref_text
            },
            callback=callback,
            error_callback=error_callback
        )
        self.submit(task)
    
    def submit_batch_synthesize(
        self,
        texts: list,
        voice_path: str,
        output_dir: str,
        speed: float = 1.0,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None
    ):
        """
        Submit a batch synthesis task (for SRT processing)
        
        Args:
            texts: List of (index, text) tuples
            voice_path: Path to voice sample
            output_dir: Directory to save outputs
            speed: Speech speed
            callback: Success callback(output_files)
            error_callback: Error callback(error)
        """
        task = Task(
            task_type=TaskType.SYNTHESIZE_BATCH,
            params={
                "texts": texts,
                "voice_path": voice_path,
                "output_dir": output_dir,
                "speed": speed
            },
            callback=callback,
            error_callback=error_callback
        )
        self.submit(task)
    
    def submit_load_model(
        self,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None
    ):
        """Submit model loading task"""
        task = Task(
            task_type=TaskType.LOAD_MODEL,
            params={},
            callback=callback,
            error_callback=error_callback
        )
        self.submit(task)
    
    def _worker_loop(self):
        """Main worker loop"""
        while self._running:
            try:
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Sentinel
                    continue
                
                self._current_task = task
                self._cancel_flag = False
                
                try:
                    result = self._execute_task(task)
                    
                    if task.callback and not self._cancel_flag:
                        task.callback(result)
                        
                except Exception as e:
                    logger.error(f"Task error: {e}")
                    if task.error_callback:
                        task.error_callback(e)
                    if self.on_error:
                        self.on_error(e)
                
                finally:
                    self._current_task = None
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a task"""
        if task.task_type == TaskType.LOAD_MODEL:
            return self._task_load_model(task.params)
        
        elif task.task_type == TaskType.SYNTHESIZE:
            return self._task_synthesize(task.params)
        
        elif task.task_type == TaskType.SYNTHESIZE_BATCH:
            return self._task_synthesize_batch(task.params)
        
        elif task.task_type == TaskType.DOWNLOAD_VOICES:
            return self._task_download_voices(task.params)
        
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _task_load_model(self, params: dict) -> bool:
        """Execute model loading task"""
        if not self.tts_engine:
            raise RuntimeError("TTS engine not set")
        
        return self.tts_engine.load_model()
    
    def _task_synthesize(self, params: dict) -> str:
        """Execute synthesis task"""
        if not self.tts_engine:
            raise RuntimeError("TTS engine not set")
        
        # voice_path is actually the voice name/key for lookup in TTSEngine.voices
        voice_name = params["voice_path"]
        ref_text = params.get("ref_text", "")  # User-provided reference text
        
        # Progress callback wrapper
        def progress_wrapper(status, progress):
            if self.on_progress:
                self.on_progress(progress, 1.0, status)  # Map to (idx, total, status) interface
        
        wav = self.tts_engine.synthesize(
            text=params["text"],
            voice_name=voice_name,  # Use voice_name parameter (matches TTSEngine.synthesize signature)
            speed=params["speed"],
            output_path=params["output_path"],
            ref_text=ref_text,  # Pass user-provided reference text
            progress_callback=progress_wrapper
        )
        
        return params["output_path"]
    
    def _task_synthesize_batch(self, params: dict) -> list:
        """Execute batch synthesis task"""
        if not self.tts_engine:
            raise RuntimeError("TTS engine not set")
        
        texts = params["texts"]
        voice_name = params["voice_path"]  # Actually voice name for lookup
        output_dir = params["output_dir"]
        speed = params["speed"]
        
        # Progress callback wrapper
        def progress_wrapper(idx, total, status):
            if self.on_progress:
                self.on_progress(idx, total, status)
        
        # Cancel flag wrapper
        def check_cancel():
            return self._cancel_flag
        
        return self.tts_engine.synthesize_batch(
            texts=texts,  # Pass full (index, text) tuples
            voice_name=voice_name,  # Use voice_name parameter
            output_dir=output_dir,
            speed=speed,
            progress_callback=progress_wrapper,
            cancel_flag=check_cancel
        )
    
    def _task_download_voices(self, params: dict) -> bool:
        """Execute voice download task"""
        from utils.voice_downloader import VoiceDownloader
        
        downloader = VoiceDownloader()
        
        def progress_wrapper(idx, total, filename, status):
            if self.on_progress:
                self.on_progress(idx, total, f"{filename}: {status}")
        
        def check_cancel():
            return self._cancel_flag
        
        success, failed, _ = downloader.download_all(
            progress_callback=progress_wrapper,
            cancel_flag=check_cancel
        )
        
        return success > 0


class ProgressCallback:
    """
    Thread-safe progress callback for UI updates
    Uses after() to schedule updates on main thread
    """
    
    def __init__(self, root, callback: Callable):
        """
        Initialize progress callback
        
        Args:
            root: Tkinter root window
            callback: Callback function to call on main thread
        """
        self.root = root
        self.callback = callback
    
    def __call__(self, *args, **kwargs):
        """Schedule callback on main thread"""
        self.root.after(0, lambda: self.callback(*args, **kwargs))


class ResultCallback:
    """
    Thread-safe result callback for UI updates
    """
    
    def __init__(self, root, callback: Callable):
        self.root = root
        self.callback = callback
    
    def __call__(self, result):
        """Schedule callback on main thread"""
        self.root.after(0, lambda: self.callback(result))


class ErrorCallback:
    """
    Thread-safe error callback for UI updates
    """
    
    def __init__(self, root, callback: Callable):
        self.root = root
        self.callback = callback
    
    def __call__(self, error):
        """Schedule callback on main thread"""
        self.root.after(0, lambda: self.callback(error))
