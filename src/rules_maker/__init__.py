"""
🤖📋 Rules Maker: Transform web documentation into AI coding assistant rules

A powerful Python library that scrapes documentation from the web and transforms it
into structured rules and workflows for AI coding assistants like Cursor and Windsurf.
"""

from .version import __version__

__all__ = ["__version__"]

# Import optional submodules lazily/safely to avoid heavy deps at import time.
# Each block is wrapped so that missing optional dependencies do not break basic usage.

try:
    from .scrapers import (
        DocumentationScraper,
        AsyncDocumentationScraper,
        AdaptiveDocumentationScraper,
    )
    __all__ += [
        "DocumentationScraper",
        "AsyncDocumentationScraper",
        "AdaptiveDocumentationScraper",
    ]
except Exception:
    pass

try:
    from .extractors import (
        ContentExtractor,
        MLContentExtractor,
        LLMContentExtractor,
        StructuredContentExtractor,
    )
    __all__ += [
        "ContentExtractor",
        "MLContentExtractor",
        "LLMContentExtractor",
        "StructuredContentExtractor",
    ]
except Exception:
    pass

try:
    from .transformers import (
        RuleTransformer,
        CursorRuleTransformer,
        WindsurfRuleTransformer,
        WorkflowTransformer,
    )
    __all__ += [
        "RuleTransformer",
        "CursorRuleTransformer",
        "WindsurfRuleTransformer",
        "WorkflowTransformer",
    ]
except Exception:
    pass

try:
    from .templates import (
        TemplateEngine,
        RuleTemplate,
        WorkflowTemplate,
    )
    __all__ += [
        "TemplateEngine",
        "RuleTemplate",
        "WorkflowTemplate",
    ]
except Exception:
    pass

try:
    from .processors import (
        ContentProcessor,
        DocumentationProcessor,
        APIDocumentationProcessor,
        CodeDocumentationProcessor,
    )
    __all__ += [
        "ContentProcessor",
        "DocumentationProcessor",
        "APIDocumentationProcessor",
        "CodeDocumentationProcessor",
    ]
except Exception:
    pass

try:
    from .models import (
        ScrapingResult,
        DocumentationStructure,
        RuleSet,
        Workflow,
        ScrapingConfig,
        TransformationConfig,
    )
    __all__ += [
        "ScrapingResult",
        "DocumentationStructure",
        "RuleSet",
        "Workflow",
        "ScrapingConfig",
        "TransformationConfig",
    ]
except Exception:
    pass

try:
    from .strategies import (
        ScrapingStrategy,
        ContentExtractionStrategy,
        RuleGenerationStrategy,
        LearningStrategy,
    )
    __all__ += [
        "ScrapingStrategy",
        "ContentExtractionStrategy",
        "RuleGenerationStrategy",
        "LearningStrategy",
    ]
except Exception:
    pass

try:
    from .filters import (
        ContentFilter,
        RelevanceFilter,
        QualityFilter,
        DuplicateFilter,
    )
    __all__ += [
        "ContentFilter",
        "RelevanceFilter",
        "QualityFilter",
        "DuplicateFilter",
    ]
except Exception:
    pass

try:
    from .utils import (
        validate_url,
        clean_content,
        detect_documentation_type,
        setup_logging,
    )
    __all__ += [
        "validate_url",
        "clean_content",
        "detect_documentation_type",
        "setup_logging",
    ]
except Exception:
    pass

# Learning engine (Phase 2)
try:
    from .learning import (
        LearningEngine,
    )
    __all__ += [
        "LearningEngine",
    ]
except Exception:
    pass
