"""Exception hierarchy for spectroview.api.

Every error raised by spectroview.api is one of these types, so callers can
catch `SpectroviewError` broadly or a specific subtype narrowly. Internal
model/fit_engine exceptions (KeyError, ValueError, ...) are always caught
and re-raised as one of these (with `from e` to preserve the original
traceback), never left to leak un-annotated.
"""


class SpectroviewError(Exception):
    """Base class for all spectroview.api errors."""


class LoadError(SpectroviewError):
    """A file could not be loaded (unsupported extension, malformed content)."""


class FitModelError(SpectroviewError):
    """A fit_model dict is malformed or references an unusable configuration."""


class FitError(SpectroviewError):
    """The fitting engine failed to produce a usable result."""


class WorkspaceError(SpectroviewError):
    """A workspace-state error: unknown map/spectrum, empty workspace, save/load failure."""


class TemplateError(SpectroviewError):
    """A fit-model or plot-template CRUD operation failed."""
