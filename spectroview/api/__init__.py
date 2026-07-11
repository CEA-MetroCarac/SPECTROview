"""
SPECTROview Programmatic API
----------------------------
A comprehensive API to script and automate spectroscopic analysis workflows
without relying on the Qt Graphical User Interface.
"""

from spectroview.api import io
from spectroview.api import fitting
from spectroview.api import processing
from spectroview.api import analysis
from spectroview.api import calculators
from spectroview.api import graphs

__all__ = [
    "io",
    "fitting",
    "processing",
    "analysis",
    "calculators",
    "graphs"
]
