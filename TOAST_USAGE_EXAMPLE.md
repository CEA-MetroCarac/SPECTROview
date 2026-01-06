# Toast Notification Usage Guide

The toast notification functionality has been moved to `spectroview.viewmodel.utils` for reuse across all workspaces.

## Installation

First, install the required package:
```bash
pip install pyqttoast
```

## Basic Usage

### Import the function

```python
from spectroview.viewmodel.utils import show_toast_notification
```

### Simple notification

```python
show_toast_notification(
    parent=self,
    message="Operation completed successfully!"
)
```

### Custom duration

```python
show_toast_notification(
    parent=self,
    message="File saved successfully",
    duration=2000  # 2 seconds
)
```

### With optional title (if needed)

```python
show_toast_notification(
    parent=self,
    message="File saved successfully",
    title="File Manager",  # Optional - omit for minimal size
    duration=2000
)
```

### Different notification types

```python
from spectroview.viewmodel.utils import show_toast_notification
from pyqttoast import ToastPreset

# Success notification (default - green)
show_toast_notification(
    parent=self,
    message="Data loaded successfully",
    preset=ToastPreset.SUCCESS
)

# Error notification (red)
show_toast_notification(
    parent=self,
    message="Failed to load file",
    title="Error",
    preset=ToastPreset.ERROR
)

# Warning notification (orange/yellow)
show_toast_notification(
    parent=self,
    message="This action cannot be undone",
    title="Warning",
    preset=ToastPreset.WARNING
)

# Info notification (blue)
show_toast_notification(
    parent=self,
    message="New update available",
    title="Info",
    preset=ToastPreset.INFO
)
```

## Integration in ViewModel → View Pattern

### In your ViewModel

```python
class VMWorkspaceExample:
    notify = Signal(str)  # Signal to emit notifications
    
    def some_operation(self):
        # ... do work ...
        self.notify.emit("Operation completed!")
```

### In your View

```python
from spectroview.viewmodel.utils import show_toast_notification

class VWorkspaceExample(QWidget):
    def __init__(self):
        # ... setup ...
        self.vm.notify.connect(self._show_notification)
    
    def _show_notification(self, message: str):
        """Show toast notification."""
        show_toast_notification(
            parent=self,
            message=message,
            duration=3000
        )
```

## Fallback Behavior

If `pyqttoast` is not installed, the function will gracefully fallback to printing messages to the console instead of showing GUI notifications.

## Parameters

- **parent** (QWidget): Parent widget for the toast notification
- **message** (str): The notification message to display
- **title** (str, optional): Title of the notification (default: None - no title for minimal size)
- **duration** (int, optional): Duration in milliseconds (default: 3000)
- **preset** (ToastPreset, optional): Style preset - SUCCESS, ERROR, WARNING, INFO, etc. (default: SUCCESS)

## Return Value

Returns the `Toast` instance if successful, or `None` if pyqttoast is not available.
