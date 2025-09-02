"""
Natural Language Processing module for Rules Maker.

This module provides natural language query processing capabilities,
allowing users to ask questions about documentation and rules in plain English.
"""

from .query_processor import NaturalLanguageQueryProcessor, QueryResponse
from .intent_analyzer import IntentAnalyzer, QueryIntent

__all__ = [
    "NaturalLanguageQueryProcessor",
    "QueryResponse",
    "IntentAnalyzer", 
    "QueryIntent",
]