"""
User Interface components for interactive CLI.

Provides rich, user-friendly interface components for the interactive CLI experience.
"""

import sys
import time
from typing import List, Optional, Any, Dict
from contextlib import contextmanager


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\\033[0m"
    BOLD = "\\033[1m"
    DIM = "\\033[2m"
    
    # Basic colors
    BLACK = "\\033[30m"
    RED = "\\033[31m"
    GREEN = "\\033[32m"
    YELLOW = "\\033[33m"
    BLUE = "\\033[34m"
    MAGENTA = "\\033[35m"
    CYAN = "\\033[36m"
    WHITE = "\\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\\033[91m"
    BRIGHT_GREEN = "\\033[92m"
    BRIGHT_YELLOW = "\\033[93m"
    BRIGHT_BLUE = "\\033[94m"
    BRIGHT_MAGENTA = "\\033[95m"
    BRIGHT_CYAN = "\\033[96m"
    BRIGHT_WHITE = "\\033[97m"
    
    # Background colors
    BG_RED = "\\033[41m"
    BG_GREEN = "\\033[42m"
    BG_YELLOW = "\\033[43m"
    BG_BLUE = "\\033[44m"


class UserInterface:
    """Rich user interface for interactive CLI interactions."""
    
    def __init__(self, use_colors: bool = True):
        """Initialize the user interface.
        
        Args:
            use_colors: Whether to use ANSI colors in output
        """
        self.use_colors = use_colors and sys.stdout.isatty()
        self.indent_level = 0
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{color}{text}{Colors.RESET}"
    
    def show_message(self, message: str, msg_type: str = "info"):
        """Display a formatted message.
        
        Args:
            message: The message to display
            msg_type: Type of message (info, success, warning, error)
        """
        indent = "  " * self.indent_level
        
        if msg_type == "success":
            colored_msg = self._colorize(message, Colors.BRIGHT_GREEN)
        elif msg_type == "warning":
            colored_msg = self._colorize(message, Colors.BRIGHT_YELLOW)
        elif msg_type == "error":
            colored_msg = self._colorize(message, Colors.BRIGHT_RED)
        elif msg_type == "info":
            colored_msg = self._colorize(message, Colors.BRIGHT_CYAN)
        else:
            colored_msg = message
        
        print(f"{indent}{colored_msg}")
    
    def show_header(self, title: str, level: int = 1):
        """Display a formatted header.
        
        Args:
            title: Header title
            level: Header level (1-3)
        """
        if level == 1:
            print("\\n" + "=" * 60)
            print(self._colorize(f"  {title}  ", Colors.BOLD + Colors.BRIGHT_WHITE))
            print("=" * 60)
        elif level == 2:
            print("\\n" + "-" * 40)
            print(self._colorize(f" {title} ", Colors.BOLD + Colors.BRIGHT_BLUE))
            print("-" * 40)
        else:
            print("\\n" + self._colorize(f"• {title}", Colors.BOLD + Colors.CYAN))
    
    async def get_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get text input from user.
        
        Args:
            prompt: Input prompt
            default: Default value if user presses Enter
            
        Returns:
            User input string
        """
        if default:
            full_prompt = f"{prompt} [{default}]: "
        else:
            full_prompt = f"{prompt}: "
        
        colored_prompt = self._colorize(full_prompt, Colors.BRIGHT_WHITE)
        user_input = input(colored_prompt).strip()
        
        return user_input if user_input else (default or "")
    
    async def get_numeric_input(self, prompt: str, min_value: Optional[int] = None,
                               max_value: Optional[int] = None, default: Optional[int] = None) -> int:
        """Get numeric input from user with validation.
        
        Args:
            prompt: Input prompt
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            default: Default value
            
        Returns:
            Valid numeric input
        """
        while True:
            try:
                constraints = []
                if min_value is not None and max_value is not None:
                    constraints.append(f"{min_value}-{max_value}")
                elif min_value is not None:
                    constraints.append(f"≥{min_value}")
                elif max_value is not None:
                    constraints.append(f"≤{max_value}")
                
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                full_prompt = f"{prompt}{constraint_str}"
                
                value_str = await self.get_input(full_prompt, str(default) if default is not None else None)
                
                if not value_str and default is not None:
                    return default
                
                value = int(value_str)
                
                if min_value is not None and value < min_value:
                    self.show_message(f"Value must be at least {min_value}", "error")
                    continue
                
                if max_value is not None and value > max_value:
                    self.show_message(f"Value must be at most {max_value}", "error")
                    continue
                
                return value
                
            except ValueError:
                self.show_message("Please enter a valid number", "error")
    
    async def confirm(self, question: str, default: bool = False) -> bool:
        """Get yes/no confirmation from user.
        
        Args:
            question: Question to ask
            default: Default answer if user presses Enter
            
        Returns:
            Boolean confirmation
        """
        default_str = "Y/n" if default else "y/N"
        response = await self.get_input(f"{question} ({default_str})")
        
        if not response:
            return default
        
        return response.lower().startswith('y')
    
    async def select_option(self, prompt: str, options: List[str], default: Optional[str] = None) -> str:
        """Let user select from a list of options.
        
        Args:
            prompt: Selection prompt
            options: List of available options
            default: Default option
            
        Returns:
            Selected option
        """
        self.show_message(prompt, "info")
        
        for i, option in enumerate(options, 1):
            marker = " (default)" if option == default else ""
            print(f"  {i}. {option}{marker}")
        
        while True:
            try:
                choice = await self.get_input("Select option", str(options.index(default) + 1) if default else None)
                
                if not choice and default:
                    return default
                
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return options[index]
                else:
                    self.show_message(f"Please select a number between 1 and {len(options)}", "error")
                    
            except ValueError:
                self.show_message("Please enter a valid number", "error")
    
    async def multi_select(self, prompt: str, options: List[str], max_selections: Optional[int] = None) -> List[str]:
        """Let user select multiple options from a list.
        
        Args:
            prompt: Selection prompt
            options: List of available options
            max_selections: Maximum number of selections allowed
            
        Returns:
            List of selected options
        """
        self.show_message(prompt, "info")
        
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        max_str = f" (max {max_selections})" if max_selections else ""
        self.show_message(f"Enter numbers separated by commas{max_str}:", "info")
        
        while True:
            try:
                response = await self.get_input("Selections")
                
                if not response:
                    return []
                
                # Parse comma-separated numbers
                selections = []
                for num_str in response.split(','):
                    num_str = num_str.strip()
                    if num_str:
                        index = int(num_str) - 1
                        if 0 <= index < len(options):
                            if options[index] not in selections:  # Avoid duplicates
                                selections.append(options[index])
                        else:
                            self.show_message(f"Invalid option: {num_str}", "error")
                            break
                else:
                    # Check max selections limit
                    if max_selections and len(selections) > max_selections:
                        self.show_message(f"Too many selections. Maximum is {max_selections}", "error")
                        continue
                    
                    if selections:
                        return selections
                    else:
                        self.show_message("No valid selections made", "warning")
                        if not await self.confirm("Try again?", True):
                            return []
                
            except ValueError:
                self.show_message("Please enter valid numbers separated by commas", "error")
    
    def show_progress(self, message: str, current: int, total: int):
        """Display progress information.
        
        Args:
            message: Progress message
            current: Current progress value
            total: Total progress value
        """
        if total > 0:
            percentage = int((current / total) * 100)
            bar_width = 30
            filled_width = int((current / total) * bar_width)
            bar = "█" * filled_width + "░" * (bar_width - filled_width)
            
            progress_line = f"{message} [{bar}] {percentage}% ({current}/{total})"
            print(f"\\r{progress_line}", end="", flush=True)
            
            if current >= total:
                print()  # New line when complete
        else:
            print(f"{message} ({current})")
    
    def show_table(self, headers: List[str], rows: List[List[str]], title: Optional[str] = None):
        """Display a formatted table.
        
        Args:
            headers: Column headers
            rows: Table rows
            title: Optional table title
        """
        if title:
            self.show_header(title, level=3)
        
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Print table
        def print_row(row_data: List[str], is_header: bool = False):
            row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row_data))
            if is_header:
                print(self._colorize(row_str, Colors.BOLD))
                print("-" * len(row_str))
            else:
                print(row_str)
        
        print_row(headers, is_header=True)
        for row in rows:
            print_row(row)
        print()
    
    def indent(self):
        """Increase indentation level."""
        self.indent_level += 1
    
    def dedent(self):
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)
    
    @contextmanager
    def indented(self):
        """Context manager for temporary indentation."""
        self.indent()
        try:
            yield
        finally:
            self.dedent()


class ProgressTracker:
    """Context manager for showing progress during operations."""
    
    def __init__(self, message: str, show_spinner: bool = True):
        """Initialize progress tracker.
        
        Args:
            message: Progress message to display
            show_spinner: Whether to show animated spinner
        """
        self.message = message
        self.show_spinner = show_spinner
        self.ui = UserInterface()
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_index = 0
    
    def __enter__(self):
        """Start progress tracking."""
        if self.show_spinner:
            self._start_spinner()
        else:
            self.ui.show_message(self.message, "info")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop progress tracking."""
        if self.show_spinner:
            self._stop_spinner()
        
        if exc_type is None:
            self.ui.show_message("✓ Completed", "success")
        else:
            self.ui.show_message(f"✗ Error: {exc_val}", "error")
    
    def _start_spinner(self):
        """Start spinner animation."""
        import threading
        self._stop_spinner_flag = False
        self._spinner_thread = threading.Thread(target=self._spinner_loop)
        self._spinner_thread.daemon = True
        self._spinner_thread.start()
    
    def _stop_spinner(self):
        """Stop spinner animation."""
        if hasattr(self, '_stop_spinner_flag'):
            self._stop_spinner_flag = True
            if hasattr(self, '_spinner_thread'):
                self._spinner_thread.join(timeout=0.1)
            print("\\r" + " " * (len(self.message) + 10) + "\\r", end="", flush=True)
    
    def _spinner_loop(self):
        """Spinner animation loop."""
        while not getattr(self, '_stop_spinner_flag', True):
            char = self.spinner_chars[self.spinner_index]
            print(f"\\r{char} {self.message}", end="", flush=True)
            self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
            time.sleep(0.1)
    
    def update_message(self, new_message: str):
        """Update progress message.
        
        Args:
            new_message: New progress message
        """
        self.message = new_message