"""
🤖📋 Rules Maker: Transform web documentation into AI coding assistant rules

A powerful Python library that scrapes documentation from the web and transforms it 
into structured rules and workflows for AI coding assistants like Cursor and Windsurf.
"""

from .version import __version__

# Core components
from .scrapers import (
    DocumentationScraper,
    AsyncDocumentationScraper,
    AdaptiveDocumentationScraper,
)

from .extractors import (
    ContentExtractor,
    MLContentExtractor, 
    LLMContentExtractor,
    StructuredContentExtractor,
)

from .transformers import (
    RuleTransformer,
    CursorRuleTransformer,
    WindsurfRuleTransformer,
    WorkflowTransformer,
)

from .templates import (
    TemplateEngine,
    RuleTemplate,
    WorkflowTemplate,
)

from .processors import (
    ContentProcessor,
    DocumentationProcessor,
    APIDocumentationProcessor,
    CodeDocumentationProcessor,
)

from .models import (
    ScrapingResult,
    DocumentationStructure,
    RuleSet,
    Workflow,
    ScrapingConfig,
    TransformationConfig,
)

from .strategies import (
    ScrapingStrategy,
    ContentExtractionStrategy,
    RuleGenerationStrategy,
    LearningStrategy,
)

from .filters import (
    ContentFilter,
    RelevanceFilter,
    QualityFilter,
    DuplicateFilter,
)

from .utils import (
    validate_url,
    clean_content,
    detect_documentation_type,
    setup_logging,
)

__all__ = [
    "__version__",
    # Scrapers
    "DocumentationScraper",
    "AsyncDocumentationScraper", 
    "AdaptiveDocumentationScraper",
    # Extractors
    "ContentExtractor",
    "MLContentExtractor",
    "LLMContentExtractor", 
    "StructuredContentExtractor",
    # Transformers
    "RuleTransformer",
    "CursorRuleTransformer",
    "WindsurfRuleTransformer",
    "WorkflowTransformer",
    # Templates
    "TemplateEngine",
    "RuleTemplate",
    "WorkflowTemplate",
    # Processors
    "ContentProcessor",
    "DocumentationProcessor",
    "APIDocumentationProcessor",
    "CodeDocumentationProcessor",
    # Models
    "ScrapingResult",
    "DocumentationStructure",
    "RuleSet",
    "Workflow",
    "ScrapingConfig",
    "TransformationConfig",
    # Strategies
    "ScrapingStrategy",
    "ContentExtractionStrategy",
    "RuleGenerationStrategy",
    "LearningStrategy",
    # Filters
    "ContentFilter",
    "RelevanceFilter",
    "QualityFilter",
    "DuplicateFilter",
    # Utils
    "validate_url",
    "clean_content",
    "detect_documentation_type",
    "setup_logging",
]
