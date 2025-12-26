"""
Theme configuration for VietTTS Desktop Application
Dark mode theme with modern aesthetics
"""

import customtkinter as ctk
from app.config import ThemeColors, Fonts


def setup_theme():
    """
    Configure CustomTkinter theme
    """
    # Set appearance mode
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")


def get_button_style(style: str = "primary") -> dict:
    """
    Get button styling configuration
    
    Args:
        style: Button style (primary, secondary, danger, success)
        
    Returns:
        dict: Button configuration
    """
    styles = {
        "primary": {
            "fg_color": ThemeColors.BTN_PRIMARY_BG,
            "hover_color": ThemeColors.BTN_PRIMARY_HOVER,
            "text_color": ThemeColors.TEXT_PRIMARY,
            "corner_radius": 8,
        },
        "secondary": {
            "fg_color": ThemeColors.BTN_SECONDARY_BG,
            "hover_color": ThemeColors.BTN_SECONDARY_HOVER,
            "text_color": ThemeColors.TEXT_PRIMARY,
            "corner_radius": 8,
        },
        "danger": {
            "fg_color": ThemeColors.BTN_DANGER_BG,
            "hover_color": ThemeColors.BTN_DANGER_HOVER,
            "text_color": ThemeColors.TEXT_PRIMARY,
            "corner_radius": 8,
        },
        "success": {
            "fg_color": ThemeColors.SUCCESS,
            "hover_color": "#16A34A",
            "text_color": ThemeColors.TEXT_PRIMARY,
            "corner_radius": 8,
        },
        "ghost": {
            "fg_color": "transparent",
            "hover_color": ThemeColors.BG_TERTIARY,
            "text_color": ThemeColors.TEXT_SECONDARY,
            "corner_radius": 8,
        }
    }
    
    return styles.get(style, styles["primary"])


def get_entry_style() -> dict:
    """Get text entry styling"""
    return {
        "fg_color": ThemeColors.BG_TERTIARY,
        "border_color": ThemeColors.BORDER,
        "text_color": ThemeColors.TEXT_PRIMARY,
        "placeholder_text_color": ThemeColors.TEXT_MUTED,
        "corner_radius": 8,
        "border_width": 1,
    }


def get_textbox_style() -> dict:
    """Get textbox styling"""
    return {
        "fg_color": ThemeColors.BG_TERTIARY,
        "border_color": ThemeColors.BORDER,
        "text_color": ThemeColors.TEXT_PRIMARY,
        "corner_radius": 8,
        "border_width": 1,
    }


def get_frame_style(style: str = "default") -> dict:
    """Get frame styling"""
    styles = {
        "default": {
            "fg_color": ThemeColors.BG_SECONDARY,
            "corner_radius": 12,
        },
        "card": {
            "fg_color": ThemeColors.BG_SECONDARY,
            "corner_radius": 12,
            "border_width": 1,
            "border_color": ThemeColors.BORDER,
        },
        "transparent": {
            "fg_color": "transparent",
            "corner_radius": 0,
        }
    }
    
    return styles.get(style, styles["default"])


def get_label_style(style: str = "default") -> dict:
    """Get label styling"""
    styles = {
        "default": {
            "text_color": ThemeColors.TEXT_PRIMARY,
            "font": Fonts.BODY,
        },
        "heading": {
            "text_color": ThemeColors.TEXT_PRIMARY,
            "font": Fonts.HEADING,
        },
        "subheading": {
            "text_color": ThemeColors.TEXT_PRIMARY,
            "font": Fonts.SUBHEADING,
        },
        "muted": {
            "text_color": ThemeColors.TEXT_MUTED,
            "font": Fonts.BODY_SMALL,
        },
        "success": {
            "text_color": ThemeColors.SUCCESS,
            "font": Fonts.BODY,
        },
        "error": {
            "text_color": ThemeColors.ERROR,
            "font": Fonts.BODY,
        },
        "info": {
            "text_color": ThemeColors.INFO,
            "font": Fonts.BODY,
        }
    }
    
    return styles.get(style, styles["default"])


def get_slider_style() -> dict:
    """Get slider styling"""
    return {
        "fg_color": ThemeColors.BG_TERTIARY,
        "progress_color": ThemeColors.PRIMARY,
        "button_color": ThemeColors.PRIMARY,
        "button_hover_color": ThemeColors.PRIMARY_HOVER,
    }


def get_dropdown_style() -> dict:
    """Get dropdown/combobox styling"""
    return {
        "fg_color": ThemeColors.BG_TERTIARY,
        "button_color": ThemeColors.BG_TERTIARY,
        "button_hover_color": ThemeColors.BTN_SECONDARY_HOVER,
        "dropdown_fg_color": ThemeColors.BG_SECONDARY,
        "dropdown_hover_color": ThemeColors.BG_TERTIARY,
        "text_color": ThemeColors.TEXT_PRIMARY,
        "corner_radius": 8,
    }


def get_progressbar_style() -> dict:
    """Get progress bar styling"""
    return {
        "fg_color": ThemeColors.BG_TERTIARY,
        "progress_color": ThemeColors.PRIMARY,
        "corner_radius": 4,
    }


def get_switch_style() -> dict:
    """Get switch/toggle styling"""
    return {
        "fg_color": ThemeColors.BG_TERTIARY,
        "progress_color": ThemeColors.PRIMARY,
        "button_color": ThemeColors.TEXT_PRIMARY,
        "button_hover_color": ThemeColors.TEXT_SECONDARY,
    }


def get_tabview_style() -> dict:
    """Get tab view styling"""
    return {
        "fg_color": ThemeColors.BG_PRIMARY,
        "segmented_button_fg_color": ThemeColors.BG_SECONDARY,
        "segmented_button_selected_color": ThemeColors.PRIMARY,
        "segmented_button_selected_hover_color": ThemeColors.PRIMARY_HOVER,
        "segmented_button_unselected_color": ThemeColors.BG_TERTIARY,
        "segmented_button_unselected_hover_color": ThemeColors.BTN_SECONDARY_HOVER,
        "corner_radius": 12,
    }


class ThemedFrame(ctk.CTkFrame):
    """Pre-styled frame with dark theme"""
    
    def __init__(self, master, style: str = "default", **kwargs):
        style_config = get_frame_style(style)
        style_config.update(kwargs)
        super().__init__(master, **style_config)


class ThemedButton(ctk.CTkButton):
    """Pre-styled button with theme variants"""
    
    def __init__(self, master, style: str = "primary", **kwargs):
        style_config = get_button_style(style)
        # Extract font if not provided
        if "font" not in kwargs:
            kwargs["font"] = Fonts.BUTTON
        style_config.update(kwargs)
        super().__init__(master, **style_config)


class ThemedLabel(ctk.CTkLabel):
    """Pre-styled label with theme variants"""
    
    def __init__(self, master, style: str = "default", **kwargs):
        style_config = get_label_style(style)
        style_config.update(kwargs)
        super().__init__(master, **style_config)


class ThemedEntry(ctk.CTkEntry):
    """Pre-styled entry with dark theme"""
    
    def __init__(self, master, **kwargs):
        style_config = get_entry_style()
        style_config.update(kwargs)
        super().__init__(master, **style_config)


class ThemedTextbox(ctk.CTkTextbox):
    """Pre-styled textbox with dark theme"""
    
    def __init__(self, master, **kwargs):
        style_config = get_textbox_style()
        style_config.update(kwargs)
        super().__init__(master, **style_config)


class ThemedSlider(ctk.CTkSlider):
    """Pre-styled slider with dark theme"""
    
    def __init__(self, master, **kwargs):
        style_config = get_slider_style()
        style_config.update(kwargs)
        super().__init__(master, **style_config)


class ThemedOptionMenu(ctk.CTkOptionMenu):
    """Pre-styled option menu with dark theme"""
    
    def __init__(self, master, **kwargs):
        style_config = get_dropdown_style()
        style_config.update(kwargs)
        super().__init__(master, **style_config)


class ThemedProgressBar(ctk.CTkProgressBar):
    """Pre-styled progress bar with dark theme"""
    
    def __init__(self, master, **kwargs):
        style_config = get_progressbar_style()
        style_config.update(kwargs)
        super().__init__(master, **style_config)


class ThemedTabview(ctk.CTkTabview):
    """Pre-styled tab view with dark theme"""
    
    def __init__(self, master, **kwargs):
        style_config = get_tabview_style()
        style_config.update(kwargs)
        super().__init__(master, **style_config)
