"""
Web scraping components for Rules Maker.

This package avoids importing heavy/optional dependencies at import time.
Each scraper class is imported lazily where used. If a scraper's optional
dependencies are missing, the import will fail only when that scraper is used.
"""

__all__ = [
    "BaseScraper",
    "DocumentationScraper",
    "AsyncDocumentationScraper",
    "AdaptiveDocumentationScraper",
]

def __getattr__(name):  # pragma: no cover - small convenience shim
    if name == "BaseScraper":
        from .base import BaseScraper
        return BaseScraper
    if name == "DocumentationScraper":
        from .documentation_scraper import DocumentationScraper
        return DocumentationScraper
    if name == "AsyncDocumentationScraper":
        from .async_documentation_scraper import AsyncDocumentationScraper
        return AsyncDocumentationScraper
    if name == "AdaptiveDocumentationScraper":
        from .adaptive_documentation_scraper import AdaptiveDocumentationScraper
        return AdaptiveDocumentationScraper
    raise AttributeError(name)
