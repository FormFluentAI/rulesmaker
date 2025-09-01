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
    detect_documentation_type,
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