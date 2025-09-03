"""
Pipelines module for Rules Maker.

This module provides high-level pipelines that orchestrate the entire
documentation processing workflow from scraping to rule generation.
"""

from .nextjs_pipeline import NextJSPipeline

__all__ = [
    "NextJSPipeline",
]
