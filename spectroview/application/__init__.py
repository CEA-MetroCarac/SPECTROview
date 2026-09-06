"""API boundary for the currently running SPECTROview desktop session.

The public :mod:`spectroview.api` package creates independent, headless
workspaces.  This package is deliberately different: it presents the state of
the *running* GUI application to trusted local integrations such as MCP.
"""

from spectroview.application.service import SpectroviewApplicationAPI

__all__ = ["SpectroviewApplicationAPI"]
