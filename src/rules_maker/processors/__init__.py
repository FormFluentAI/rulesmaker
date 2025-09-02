"""
Content processing components for Rules Maker.
"""

from .base import ContentProcessor
from .documentation_processor import DocumentationProcessor  
from .api_processor import APIDocumentationProcessor
from .code_processor import CodeDocumentationProcessor

__all__ = [
    "ContentProcessor",
    "DocumentationProcessor",
    "APIDocumentationProcessor", 
    "CodeDocumentationProcessor",
    "MLDocumentationProcessor",
]

# Lazy import to avoid importing optional heavy ML deps unless requested
def __getattr__(name: str):
    if name == "MLDocumentationProcessor":
        from .ml_documentation_processor import MLDocumentationProcessor as _MLDP
        return _MLDP
    raise AttributeError(name)
