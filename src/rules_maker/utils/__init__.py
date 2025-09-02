"""
Utility modules for Rules Maker.

This module provides utility functions and credential management.
"""

from .credentials import CredentialManager, get_credential_manager, setup_bedrock_credentials

# Import utility functions from main_utils to maintain backward compatibility
from .main_utils import (
    validate_url,
    normalize_url,
    clean_content,
    extract_metadata_from_html,
    setup_logging,
    extract_navigation_links,
    is_documentation_url,
    extract_main_content,
    extract_domain,
    generate_content_hash,
    extract_code_blocks,
    sanitize_filename,
    get_content_type,
    split_text_into_chunks,
    calculate_text_similarity
)
from .main_utils import detect_documentation_type as _basic_detect_doc_type
from ..models import DocumentationType

def detect_documentation_type(url: str, title: str = "", content: str = "") -> DocumentationType:
    """Compatibility wrapper: map 3-arg signature to basic detector and enum.

    This preserves the legacy call sites that expect (url, title, content)
    while leveraging the simpler detector from main_utils.
    """
    result = _basic_detect_doc_type(content or "", url or "")
    mapping = {
        'api': DocumentationType.API,
        'tutorial': DocumentationType.TUTORIAL,
        'installation': DocumentationType.REFERENCE,
        'reference': DocumentationType.REFERENCE,
        'framework': DocumentationType.FRAMEWORK,
        'readme': DocumentationType.README,
        'changelog': DocumentationType.CHANGELOG,
        'library': DocumentationType.LIBRARY if hasattr(DocumentationType, 'LIBRARY') else DocumentationType.FRAMEWORK,
        'unknown': DocumentationType.UNKNOWN,
    }
    return mapping.get(str(result).lower(), DocumentationType.UNKNOWN)

__all__ = [
    'CredentialManager', 
    'get_credential_manager', 
    'setup_bedrock_credentials',
    'validate_url',
    'normalize_url',
    'clean_content', 
    'extract_metadata_from_html',
    'detect_documentation_type',
    'setup_logging',
    'extract_navigation_links',
    'is_documentation_url',
    'extract_main_content',
    'extract_domain',
    'generate_content_hash',
    'extract_code_blocks',
    'sanitize_filename',
    'get_content_type',
    'split_text_into_chunks',
    'calculate_text_similarity'
]
