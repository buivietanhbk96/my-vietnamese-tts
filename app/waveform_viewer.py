# -*- coding: utf-8 -*-
"""
Waveform Viewer - Hiển thị và edit audio waveform
VIP Pro MAX feature - Visualize audio như DAW chuyên nghiệp
"""

import tkinter as tk
import numpy as np
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass
from enum import Enum
import threading
import time


class SelectionMode(Enum):
    """Chế độ selection"""
    NONE = "none"
    RANGE = "range"  # Chọn vùng
    POINT = "point"  # Chọn điểm


@dataclass
class Selection:
    """Thông tin vùng chọn"""
    start_sample: int = 0
    end_sample: int = 0
    
    @property
    def start(self) -> int:
        return min(self.start_sample, self.end_sample)
    
    @property
    def end(self) -> int:
        return max(self.start_sample, self.end_sample)
    
    @property
    def length(self) -> int:
        return self.end - self.start
    
    def to_time(self, sample_rate: int) -> Tuple[float, float]:
        """Chuyển sang thời gian (giây)"""
        return self.start / sample_rate, self.end / sample_rate


@dataclass
class WaveformStyle:
    """Style cho waveform"""
    background: str = "#1a1a2e"
    waveform_color: str = "#4cc9f0"
    waveform_fill: str = "#4cc9f040"
    selection_color: str = "#f72585"
    selection_fill: str = "#f7258540"
    playhead_color: str = "#ffd60a"
    grid_color: str = "#2a2a4a"
    text_color: str = "#ffffff"
    zero_line_color: str = "#3a3a5a"
    rms_color: str = "#7209b7"


class WaveformViewer(tk.Canvas):
    """
    Professional Waveform Viewer
    Hiển thị waveform với đầy đủ tính năng như DAW
    """
    
    def __init__(
        self,
        parent,
        width: int = 800,
        height: int = 150,
        style: Optional[WaveformStyle] = None,
        on_selection_change: Optional[Callable[[Selection], None]] = None,
        on_playhead_change: Optional[Callable[[int], None]] = None,
        **kwargs
    ):
        self.style = style or WaveformStyle()
        
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=self.style.background,
            highlightthickness=0,
            **kwargs
        )
        
        # Audio data
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 22050
        self._duration: float = 0.0
        
        # View settings
        self._zoom_level: float = 1.0  # 1.0 = fit all
        self._scroll_position: float = 0.0  # 0.0 to 1.0
        self._samples_per_pixel: int = 1
        
        # Selection
        self._selection: Selection = Selection()
        self._selection_mode: SelectionMode = SelectionMode.NONE
        self._is_selecting: bool = False
        
        # Playhead
        self._playhead_sample: int = 0
        self._is_playing: bool = False
        self._playhead_thread: Optional[threading.Thread] = None
        self._stop_playhead: bool = False
        
        # Cache
        self._waveform_cache: Optional[np.ndarray] = None
        self._cache_params: Tuple = ()
        
        # Callbacks
        self._on_selection_change = on_selection_change
        self._on_playhead_change = on_playhead_change
        
        # Bind events
        self.bind("<Configure>", self._on_resize)
        self.bind("<ButtonPress-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Double-Button-1>", self._on_double_click)
        
        # Bind keyboard
        self.bind("<Left>", self._on_key_left)
        self.bind("<Right>", self._on_key_right)
        self.bind("<Home>", self._on_key_home)
        self.bind("<End>", self._on_key_end)
        self.bind("<Escape>", self._on_key_escape)
        
        # Initial draw
        self._draw_empty()
    
    def load_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 22050
    ) -> None:
        """Load audio data để hiển thị"""
        # Validate input
        if len(audio_data) == 0 or sample_rate <= 0:
            self.clear()
            return
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        self._audio_data = audio_data.astype(np.float32)
        self._sample_rate = sample_rate
        self._duration = len(audio_data) / sample_rate
        
        # Reset view
        self._zoom_level = 1.0
        self._scroll_position = 0.0
        self._selection = Selection()
        self._playhead_sample = 0
        
        # Clear cache
        self._waveform_cache = None
        self._cache_params = ()
        
        # Redraw
        self._draw()
    
    def clear(self) -> None:
        """Xóa audio data"""
        self._audio_data = None
        self._duration = 0.0
        self._waveform_cache = None
        self._draw_empty()
    
    def set_playhead(self, sample: int) -> None:
        """Đặt vị trí playhead"""
        if self._audio_data is None:
            return
        
        self._playhead_sample = max(0, min(sample, len(self._audio_data) - 1))
        self._draw()
        
        if self._on_playhead_change:
            self._on_playhead_change(self._playhead_sample)
    
    def set_playhead_time(self, time_seconds: float) -> None:
        """Đặt vị trí playhead theo thời gian"""
        sample = int(time_seconds * self._sample_rate)
        self.set_playhead(sample)
    
    def get_selection(self) -> Optional[Selection]:
        """Lấy vùng chọn hiện tại"""
        if self._selection.length > 0:
            return self._selection
        return None
    
    def get_selection_audio(self) -> Optional[np.ndarray]:
        """Lấy audio data của vùng chọn"""
        if self._audio_data is None or self._selection.length == 0:
            return None
        
        return self._audio_data[self._selection.start:self._selection.end].copy()
    
    def set_selection(self, start_sample: int, end_sample: int) -> None:
        """Đặt vùng chọn"""
        if self._audio_data is None:
            return
        
        self._selection.start_sample = max(0, start_sample)
        self._selection.end_sample = min(len(self._audio_data), end_sample)
        self._draw()
        
        if self._on_selection_change:
            self._on_selection_change(self._selection)
    
    def clear_selection(self) -> None:
        """Xóa vùng chọn"""
        self._selection = Selection()
        self._draw()
    
    def zoom_in(self) -> None:
        """Zoom in waveform"""
        self._zoom_level = min(self._zoom_level * 1.5, 100.0)
        self._waveform_cache = None
        self._draw()
    
    def zoom_out(self) -> None:
        """Zoom out waveform"""
        self._zoom_level = max(self._zoom_level / 1.5, 1.0)
        self._waveform_cache = None
        self._draw()
    
    def zoom_fit(self) -> None:
        """Zoom để fit toàn bộ audio"""
        self._zoom_level = 1.0
        self._scroll_position = 0.0
        self._waveform_cache = None
        self._draw()
    
    def zoom_selection(self) -> None:
        """Zoom vào vùng chọn"""
        if self._audio_data is None or len(self._audio_data) == 0 or self._selection.length == 0:
            return
        
        total_samples = len(self._audio_data)
        selection_ratio = self._selection.length / total_samples
        
        self._zoom_level = 1.0 / selection_ratio
        self._scroll_position = self._selection.start / total_samples
        self._waveform_cache = None
        self._draw()
    
    def scroll_to(self, position: float) -> None:
        """Scroll tới vị trí (0.0 - 1.0)"""
        max_scroll = max(0.0, 1.0 - 1.0 / self._zoom_level)
        self._scroll_position = max(0.0, min(position, max_scroll))
        self._waveform_cache = None
        self._draw()
    
    def start_playhead_animation(
        self,
        start_sample: Optional[int] = None,
        speed_multiplier: float = 1.0
    ) -> None:
        """Bắt đầu animation playhead"""
        if self._audio_data is None:
            return
        
        self.stop_playhead_animation()
        
        if start_sample is not None:
            self._playhead_sample = start_sample
        
        self._is_playing = True
        self._stop_playhead = False
        
        def animate():
            update_interval = 0.03  # 30ms
            samples_per_update = int(
                self._sample_rate * update_interval * speed_multiplier
            )
            
            while not self._stop_playhead:
                self._playhead_sample += samples_per_update
                
                if self._playhead_sample >= len(self._audio_data):
                    self._playhead_sample = 0
                    self._is_playing = False
                    break
                
                # Update on main thread
                try:
                    self.after_idle(self._draw_playhead_only)
                except tk.TclError:
                    break
                
                time.sleep(update_interval)
            
            self._is_playing = False
        
        self._playhead_thread = threading.Thread(target=animate, daemon=True)
        self._playhead_thread.start()
    
    def stop_playhead_animation(self) -> None:
        """Dừng animation playhead"""
        self._stop_playhead = True
        self._is_playing = False
        
        if self._playhead_thread and self._playhead_thread.is_alive():
            self._playhead_thread.join(timeout=0.1)
    
    def get_time_at_position(self, x: int) -> float:
        """Lấy thời gian tại vị trí x"""
        sample = self._pixel_to_sample(x)
        return sample / self._sample_rate
    
    def _pixel_to_sample(self, x: int) -> int:
        """Chuyển pixel x sang sample index"""
        if self._audio_data is None or len(self._audio_data) == 0:
            return 0
        
        width = self.winfo_width()
        total_samples = len(self._audio_data)
        visible_samples = int(total_samples / self._zoom_level)
        start_sample = int(self._scroll_position * total_samples)
        
        sample = start_sample + int((x / width) * visible_samples)
        return max(0, min(sample, total_samples - 1))
    
    def _sample_to_pixel(self, sample: int) -> int:
        """Chuyển sample index sang pixel x"""
        if self._audio_data is None or len(self._audio_data) == 0:
            return 0
        
        width = self.winfo_width()
        total_samples = len(self._audio_data)
        visible_samples = int(total_samples / self._zoom_level)
        start_sample = int(self._scroll_position * total_samples)
        
        relative_sample = sample - start_sample
        if visible_samples == 0:
            return 0
        
        return int((relative_sample / visible_samples) * width)
    
    def _compute_waveform(self) -> np.ndarray:
        """Tính toán waveform để vẽ"""
        if self._audio_data is None or len(self._audio_data) == 0:
            return np.array([])
        
        width = self.winfo_width()
        height = self.winfo_height()
        
        # Check cache
        cache_key = (width, height, self._zoom_level, self._scroll_position)
        if self._cache_params == cache_key and self._waveform_cache is not None:
            return self._waveform_cache
        
        total_samples = len(self._audio_data)
        visible_samples = int(total_samples / self._zoom_level)
        start_sample = int(self._scroll_position * total_samples)
        end_sample = min(start_sample + visible_samples, total_samples)
        
        visible_audio = self._audio_data[start_sample:end_sample]
        
        if len(visible_audio) == 0:
            return np.array([])
        
        # Compute samples per pixel
        samples_per_pixel = max(1, len(visible_audio) // width)
        self._samples_per_pixel = samples_per_pixel
        
        # Downsample for display
        num_chunks = width
        chunk_size = len(visible_audio) // num_chunks if num_chunks > 0 else 1
        
        if chunk_size == 0:
            chunk_size = 1
        
        # Compute min/max/rms for each pixel
        waveform_data = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(visible_audio))
            chunk = visible_audio[start:end]
            
            if len(chunk) > 0:
                min_val = float(np.min(chunk))
                max_val = float(np.max(chunk))
                rms_val = float(np.sqrt(np.mean(chunk ** 2)))
            else:
                min_val = max_val = rms_val = 0.0
            
            waveform_data.append((min_val, max_val, rms_val))
        
        self._waveform_cache = waveform_data
        self._cache_params = cache_key
        
        return waveform_data
    
    def _draw_empty(self) -> None:
        """Vẽ trạng thái trống"""
        self.delete("all")
        
        width = self.winfo_width()
        height = self.winfo_height()
        
        # Draw zero line
        self.create_line(
            0, height // 2, width, height // 2,
            fill=self.style.zero_line_color,
            dash=(2, 4)
        )
        
        # Draw text
        self.create_text(
            width // 2, height // 2,
            text="No audio loaded",
            fill=self.style.text_color,
            font=("Segoe UI", 10)
        )
    
    def _draw(self) -> None:
        """Vẽ toàn bộ waveform"""
        self.delete("all")
        
        if self._audio_data is None:
            self._draw_empty()
            return
        
        width = self.winfo_width()
        height = self.winfo_height()
        center_y = height // 2
        
        # Draw grid
        self._draw_grid()
        
        # Draw zero line
        self.create_line(
            0, center_y, width, center_y,
            fill=self.style.zero_line_color,
            width=1
        )
        
        # Draw selection background
        if self._selection.length > 0:
            self._draw_selection()
        
        # Compute and draw waveform
        waveform_data = self._compute_waveform()
        
        if len(waveform_data) > 0:
            # Draw RMS (inner fill)
            rms_points = []
            for i, (_, _, rms) in enumerate(waveform_data):
                y_top = center_y - int(rms * (height // 2) * 0.9)
                y_bottom = center_y + int(rms * (height // 2) * 0.9)
                rms_points.append((i, y_top, y_bottom))
            
            # Draw RMS as filled polygon
            if len(rms_points) >= 2:
                polygon_points = []
                for x, y_top, _ in rms_points:
                    polygon_points.extend([x, y_top])
                for x, _, y_bottom in reversed(rms_points):
                    polygon_points.extend([x, y_bottom])
                
                if len(polygon_points) >= 6:
                    self.create_polygon(
                        polygon_points,
                        fill=self.style.rms_color,
                        outline=""
                    )
            
            # Draw waveform peaks
            for i, (min_val, max_val, _) in enumerate(waveform_data):
                y_min = center_y - int(max_val * (height // 2) * 0.95)
                y_max = center_y - int(min_val * (height // 2) * 0.95)
                
                self.create_line(
                    i, y_min, i, y_max,
                    fill=self.style.waveform_color,
                    width=1
                )
        
        # Draw playhead
        self._draw_playhead_only()
        
        # Draw time markers
        self._draw_time_markers()
    
    def _draw_grid(self) -> None:
        """Vẽ grid"""
        width = self.winfo_width()
        height = self.winfo_height()
        
        # Vertical grid lines (time divisions)
        num_divisions = 10
        for i in range(1, num_divisions):
            x = (width * i) // num_divisions
            self.create_line(
                x, 0, x, height,
                fill=self.style.grid_color,
                dash=(1, 3)
            )
        
        # Horizontal grid lines (amplitude)
        for amp in [0.25, 0.5, 0.75]:
            y_top = int(height // 2 - amp * height // 2)
            y_bottom = int(height // 2 + amp * height // 2)
            
            self.create_line(
                0, y_top, width, y_top,
                fill=self.style.grid_color,
                dash=(1, 3)
            )
            self.create_line(
                0, y_bottom, width, y_bottom,
                fill=self.style.grid_color,
                dash=(1, 3)
            )
    
    def _draw_selection(self) -> None:
        """Vẽ vùng chọn"""
        height = self.winfo_height()
        
        x_start = self._sample_to_pixel(self._selection.start)
        x_end = self._sample_to_pixel(self._selection.end)
        
        # Draw selection rectangle
        self.create_rectangle(
            x_start, 0, x_end, height,
            fill=self.style.selection_fill,
            outline=self.style.selection_color,
            width=1
        )
        
        # Draw selection edges
        self.create_line(
            x_start, 0, x_start, height,
            fill=self.style.selection_color,
            width=2
        )
        self.create_line(
            x_end, 0, x_end, height,
            fill=self.style.selection_color,
            width=2
        )
    
    def _draw_playhead_only(self) -> None:
        """Vẽ chỉ playhead (cho animation)"""
        self.delete("playhead")
        
        height = self.winfo_height()
        x = self._sample_to_pixel(self._playhead_sample)
        
        # Draw playhead line
        self.create_line(
            x, 0, x, height,
            fill=self.style.playhead_color,
            width=2,
            tags="playhead"
        )
        
        # Draw playhead triangle
        self.create_polygon(
            x - 6, 0,
            x + 6, 0,
            x, 8,
            fill=self.style.playhead_color,
            tags="playhead"
        )
    
    def _draw_time_markers(self) -> None:
        """Vẽ markers thời gian"""
        if self._audio_data is None:
            return
        
        width = self.winfo_width()
        height = self.winfo_height()
        
        total_samples = len(self._audio_data)
        visible_samples = int(total_samples / self._zoom_level)
        start_sample = int(self._scroll_position * total_samples)
        
        start_time = start_sample / self._sample_rate
        visible_duration = visible_samples / self._sample_rate
        
        # Determine time interval for markers
        intervals = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60]
        interval = intervals[0]
        for i in intervals:
            if visible_duration / i <= 10:
                interval = i
                break
        
        # Draw markers
        current_time = (int(start_time / interval) + 1) * interval
        while current_time < start_time + visible_duration:
            x = int(((current_time - start_time) / visible_duration) * width)
            
            # Format time
            if interval < 1:
                time_str = f"{current_time:.2f}s"
            else:
                minutes = int(current_time // 60)
                seconds = current_time % 60
                if minutes > 0:
                    time_str = f"{minutes}:{seconds:05.2f}"
                else:
                    time_str = f"{seconds:.1f}s"
            
            self.create_text(
                x, height - 10,
                text=time_str,
                fill=self.style.text_color,
                font=("Segoe UI", 8),
                anchor="s"
            )
            
            current_time += interval
    
    def _on_resize(self, event) -> None:
        """Xử lý resize"""
        self._waveform_cache = None
        self._draw()
    
    def _on_mouse_down(self, event) -> None:
        """Xử lý mouse down"""
        self.focus_set()
        
        if self._audio_data is None:
            return
        
        sample = self._pixel_to_sample(event.x)
        
        self._is_selecting = True
        self._selection.start_sample = sample
        self._selection.end_sample = sample
        
        self.set_playhead(sample)
    
    def _on_mouse_drag(self, event) -> None:
        """Xử lý mouse drag"""
        if not self._is_selecting or self._audio_data is None:
            return
        
        sample = self._pixel_to_sample(event.x)
        self._selection.end_sample = sample
        
        self._draw()
    
    def _on_mouse_up(self, event) -> None:
        """Xử lý mouse up"""
        self._is_selecting = False
        
        if self._on_selection_change and self._selection.length > 0:
            self._on_selection_change(self._selection)
    
    def _on_mouse_wheel(self, event) -> None:
        """Xử lý mouse wheel zoom"""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def _on_double_click(self, event) -> None:
        """Xử lý double click - select all"""
        if self._audio_data is not None:
            self.set_selection(0, len(self._audio_data))
    
    def _on_key_left(self, event) -> None:
        """Di chuyển playhead sang trái"""
        samples = self._samples_per_pixel * 10
        self.set_playhead(self._playhead_sample - samples)
    
    def _on_key_right(self, event) -> None:
        """Di chuyển playhead sang phải"""
        samples = self._samples_per_pixel * 10
        self.set_playhead(self._playhead_sample + samples)
    
    def _on_key_home(self, event) -> None:
        """Di chuyển playhead về đầu"""
        self.set_playhead(0)
    
    def _on_key_end(self, event) -> None:
        """Di chuyển playhead về cuối"""
        if self._audio_data is not None:
            self.set_playhead(len(self._audio_data) - 1)
    
    def _on_key_escape(self, event) -> None:
        """Xóa selection"""
        self.clear_selection()


class WaveformToolbar(tk.Frame):
    """
    Toolbar cho WaveformViewer
    """
    
    def __init__(
        self,
        parent,
        waveform_viewer: WaveformViewer,
        bg: str = "#16213e",
        **kwargs
    ):
        super().__init__(parent, bg=bg, **kwargs)
        
        self.viewer = waveform_viewer
        
        # Zoom controls
        btn_zoom_in = tk.Button(
            self,
            text="🔍+",
            command=self.viewer.zoom_in,
            bg="#0f3460",
            fg="white",
            relief="flat",
            padx=8,
            pady=4
        )
        btn_zoom_in.pack(side="left", padx=2, pady=4)
        
        btn_zoom_out = tk.Button(
            self,
            text="🔍-",
            command=self.viewer.zoom_out,
            bg="#0f3460",
            fg="white",
            relief="flat",
            padx=8,
            pady=4
        )
        btn_zoom_out.pack(side="left", padx=2, pady=4)
        
        btn_zoom_fit = tk.Button(
            self,
            text="📐 Fit",
            command=self.viewer.zoom_fit,
            bg="#0f3460",
            fg="white",
            relief="flat",
            padx=8,
            pady=4
        )
        btn_zoom_fit.pack(side="left", padx=2, pady=4)
        
        btn_zoom_sel = tk.Button(
            self,
            text="🔲 Zoom Selection",
            command=self.viewer.zoom_selection,
            bg="#0f3460",
            fg="white",
            relief="flat",
            padx=8,
            pady=4
        )
        btn_zoom_sel.pack(side="left", padx=2, pady=4)
        
        # Separator
        tk.Frame(self, bg="#3a3a5a", width=2).pack(
            side="left", fill="y", padx=8, pady=4
        )
        
        # Selection info
        self.selection_label = tk.Label(
            self,
            text="Selection: None",
            bg=bg,
            fg="white",
            font=("Segoe UI", 9)
        )
        self.selection_label.pack(side="left", padx=8)
        
        # Time display
        self.time_label = tk.Label(
            self,
            text="00:00.000 / 00:00.000",
            bg=bg,
            fg="#4cc9f0",
            font=("Consolas", 10)
        )
        self.time_label.pack(side="right", padx=8)
        
        # Bind viewer events
        self.viewer._on_selection_change = self._on_selection_change
        self.viewer._on_playhead_change = self._on_playhead_change
    
    def _on_selection_change(self, selection: Selection) -> None:
        """Xử lý thay đổi selection"""
        start_time, end_time = selection.to_time(self.viewer._sample_rate)
        duration = end_time - start_time
        
        self.selection_label.config(
            text=f"Selection: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)"
        )
    
    def _on_playhead_change(self, sample: int) -> None:
        """Xử lý thay đổi playhead"""
        current_time = sample / self.viewer._sample_rate
        total_time = self.viewer._duration
        
        current_str = self._format_time(current_time)
        total_str = self._format_time(total_time)
        
        self.time_label.config(text=f"{current_str} / {total_str}")
    
    def _format_time(self, seconds: float) -> str:
        """Format thời gian"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
