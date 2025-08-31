"""
Core data models for Rules Maker.

Defines the main data structures used throughout the application.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl


class DocumentationType(str, Enum):
    """Types of documentation that can be scraped."""
    API = "api"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    GUIDE = "guide"
    CHANGELOG = "changelog"
    README = "readme"
    UNKNOWN = "unknown"


class RuleFormat(str, Enum):
    """Supported rule output formats."""
    CURSOR = "cursor"
    WINDSURF = "windsurf"
    CUSTOM = "custom"
    JSON = "json"
    YAML = "yaml"


class ScrapingStatus(str, Enum):
    """Status of scraping operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ContentSection(BaseModel):
    """Represents a section of documentation content."""
    title: str
    content: str
    level: int = 1
    url: Optional[HttpUrl] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    subsections: List['ContentSection'] = Field(default_factory=list)


class ScrapingConfig(BaseModel):
    """Configuration for scraping operations."""
    max_depth: int = 3
    max_pages: int = 100
    timeout: int = 30
    delay: float = 1.0
    user_agent: str = "RulesMaker/0.1.0"
    headers: Dict[str, str] = Field(default_factory=dict)
    cookies: Dict[str, str] = Field(default_factory=dict)
    proxy: Optional[str] = None
    javascript_enabled: bool = True
    follow_links: bool = True
    respect_robots_txt: bool = True
    rate_limit: float = 1.0


class TransformationConfig(BaseModel):
    """Configuration for content transformation."""
    rule_format: RuleFormat = RuleFormat.CURSOR
    template_name: str = "default"
    max_rules: int = 50
    include_examples: bool = True
    include_metadata: bool = True
    language_hint: Optional[str] = None
    framework_hint: Optional[str] = None
    custom_instructions: str = ""


class ScrapingResult(BaseModel):
    """Result of a scraping operation."""
    url: HttpUrl
    title: str
    content: str
    sections: List[ContentSection] = Field(default_factory=list)
    documentation_type: DocumentationType = DocumentationType.UNKNOWN
    status: ScrapingStatus = ScrapingStatus.PENDING
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    raw_html: Optional[str] = None


class DocumentationStructure(BaseModel):
    """Represents the overall structure of documentation."""
    name: str
    base_url: HttpUrl
    documentation_type: DocumentationType
    sections: List[ContentSection] = Field(default_factory=list)
    navigation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_updated: Optional[datetime] = None


class Rule(BaseModel):
    """Represents a single coding rule."""
    id: str
    title: str
    description: str
    category: str
    priority: int = 1
    tags: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RuleSet(BaseModel):
    """Collection of rules for a specific context."""
    name: str
    description: str
    rules: List[Rule] = Field(default_factory=list)
    format: RuleFormat = RuleFormat.CURSOR
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class WorkflowStep(BaseModel):
    """A single step in a workflow."""
    id: str
    name: str
    description: str
    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    conditions: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class Workflow(BaseModel):
    """Represents a workflow for AI assistants."""
    name: str
    description: str
    steps: List[WorkflowStep] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ExtractionPattern(BaseModel):
    """Pattern for extracting specific content."""
    name: str
    pattern_type: Literal["css", "xpath", "regex", "ml"]
    pattern: str
    description: str
    examples: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class LearningExample(BaseModel):
    """Example for training ML extractors."""
    input_html: str
    expected_output: Dict[str, Any]
    url: HttpUrl
    documentation_type: DocumentationType
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingSet(BaseModel):
    """Collection of learning examples."""
    name: str
    description: str
    examples: List[LearningExample] = Field(default_factory=list)
    documentation_type: DocumentationType
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Forward reference resolution
ContentSection.model_rebuild()
