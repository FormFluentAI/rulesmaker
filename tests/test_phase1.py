"""
Tests for Phase 1 implementations.

Tests for async scraper, ML extractor, and LLM integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from bs4 import BeautifulSoup

from rules_maker.scrapers.async_documentation_scraper import AsyncDocumentationScraper
from rules_maker.scrapers.adaptive_documentation_scraper import AdaptiveDocumentationScraper
from rules_maker.extractors.ml_extractor import MLContentExtractor
from rules_maker.extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider
from rules_maker.models import (
    ScrapingConfig, ScrapingResult, ScrapingStatus,
    ContentSection, TrainingSet, LearningExample, DocumentationType
)


class TestAsyncDocumentationScraper:
    """Tests for async documentation scraper."""
    
    @pytest.fixture
    def scraper_config(self):
        """Create test scraper configuration."""
        return ScrapingConfig(
            max_pages=5,
            max_depth=2,
            rate_limit=0.1,  # Fast for testing
            timeout=10
        )
    
    @pytest.fixture
    def async_scraper(self, scraper_config):
        """Create async scraper instance."""
        return AsyncDocumentationScraper(scraper_config)
    
    @pytest.mark.asyncio
    async def test_scraper_initialization(self, async_scraper):
        """Test scraper initialization."""
        assert async_scraper.config.max_pages == 5
        assert async_scraper.config.rate_limit == 0.1
        assert async_scraper.session is None
        assert async_scraper.semaphore is None
    
    @pytest.mark.asyncio
    async def test_session_creation(self, async_scraper):
        """Test session creation."""
        await async_scraper._ensure_session()
        
        assert async_scraper.session is not None
        assert not async_scraper.session.closed
        assert async_scraper.semaphore is not None
        
        await async_scraper.close()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, async_scraper):
        """Test async context manager."""
        async with async_scraper as scraper:
            assert scraper.session is not None
        
        # Session should be closed after context
        assert async_scraper.session is None or async_scraper.session.closed
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_scrape_url_success(self, mock_get, async_scraper):
        """Test successful URL scraping."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Title</h1>
                <p>Test content</p>
            </body>
        </html>
        """
        
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with async_scraper:
            result = await async_scraper.scrape_url("https://example.com")
        
        assert result.status == ScrapingStatus.COMPLETED
        assert result.title == "Test Page"
        assert "Test content" in result.content
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_scrape_url_failure(self, mock_get, async_scraper):
        """Test URL scraping failure."""
        mock_response = AsyncMock()
        mock_response.status = 404
        
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with async_scraper:
            result = await async_scraper.scrape_url("https://example.com/notfound")
        
        assert result.status == ScrapingStatus.FAILED
        assert "HTTP 404" in result.error_message
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_scrape_multiple_urls(self, mock_get, async_scraper):
        """Test scraping multiple URLs."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "<html><title>Test</title><body>Content</body></html>"
        
        mock_get.return_value.__aenter__.return_value = mock_response
        
        urls = ["https://example.com/1", "https://example.com/2"]
        
        async with async_scraper:
            results = await async_scraper.scrape_multiple(urls)
        
        assert len(results) == 2
        assert all(result.status == ScrapingStatus.COMPLETED for result in results)


class TestMLContentExtractor:
    """Tests for ML content extractor."""
    
    @pytest.fixture
    def ml_extractor(self):
        """Create ML extractor instance."""
        return MLContentExtractor(use_transformers=False)  # Disable transformers for testing
    
    @pytest.fixture
    def sample_html(self):
        """Sample HTML for testing."""
        return """
        <html>
            <head><title>Python Tutorial</title></head>
            <body>
                <h1>Introduction to Python</h1>
                <p>Python is a programming language...</p>
                
                <h2>Installation</h2>
                <p>To install Python:</p>
                <pre><code>pip install python</code></pre>
                
                <h2>Basic Examples</h2>
                <p>Here are some examples:</p>
                <code>print("Hello")</code>
            </body>
        </html>
        """
    
    def test_extractor_initialization(self, ml_extractor):
        """Test ML extractor initialization."""
        assert ml_extractor.tfidf_vectorizer is not None
        assert ml_extractor.section_classifier is not None
        assert not ml_extractor.is_trained
        assert not ml_extractor.use_transformers
    
    def test_extract_sections(self, ml_extractor, sample_html):
        """Test section extraction."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        sections = ml_extractor.extract_sections(soup, "https://example.com")
        
        assert len(sections) >= 2  # Should find h1 and h2 sections
        
        # Check section properties
        intro_section = next((s for s in sections if "Introduction" in s.title), None)
        assert intro_section is not None
        assert intro_section.level == 1
        assert intro_section.metadata.get('section_type') is not None
    
    def test_extract_content(self, ml_extractor, sample_html):
        """Test content extraction."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        result = ml_extractor.extract(soup, "https://example.com")
        
        assert result['title'] == "Python Tutorial"
        assert 'sections' in result
        assert 'code_examples' in result
        assert 'confidence_score' in result
    
    def test_rule_based_classification(self, ml_extractor):
        """Test rule-based section classification."""
        # Test installation section
        result = ml_extractor._rule_based_section_classification(
            "Installation Guide", 
            "To install, run pip install package"
        )
        assert result == 'installation'
        
        # Test API reference
        result = ml_extractor._rule_based_section_classification(
            "API Reference",
            "Function documentation and methods"
        )
        assert result == 'api_reference'
    
    def test_code_detection(self, ml_extractor):
        """Test code content detection."""
        assert ml_extractor._has_code_content("```python\nprint('hello')\n```")
        assert ml_extractor._has_code_content("Use `print()` function")
        assert ml_extractor._has_code_content("import os")
        assert not ml_extractor._has_code_content("This is just text")
    
    def test_training_with_empty_set(self, ml_extractor):
        """Test training with empty training set."""
        training_set = TrainingSet(
            name="Empty Set",
            description="Empty training set",
            examples=[],
            documentation_type=DocumentationType.GUIDE
        )
        
        with pytest.raises(ValueError, match="Training set is empty"):
            ml_extractor.train(training_set)


class TestLLMContentExtractor:
    """Tests for LLM content extractor."""
    
    @pytest.fixture
    def llm_config(self):
        """Create LLM configuration."""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            api_key="test-key",
            temperature=0.3
        )
    
    @pytest.fixture
    def llm_extractor(self, llm_config):
        """Create LLM extractor instance."""
        return LLMContentExtractor(llm_config=llm_config)
    
    def test_extractor_initialization(self, llm_extractor, llm_config):
        """Test LLM extractor initialization."""
        assert llm_extractor.config.provider == LLMProvider.OPENAI
        assert llm_extractor.config.model_name == "gpt-3.5-turbo"
        assert llm_extractor.client is not None
    
    def test_clean_text_extraction(self, llm_extractor):
        """Test clean text extraction."""
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <script>console.log('remove this');</script>
                <p>Keep this text</p>
                <style>body { color: red; }</style>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        clean_text = llm_extractor._extract_clean_text(soup)
        
        assert "Keep this text" in clean_text
        assert "console.log" not in clean_text
        assert "color: red" not in clean_text
    
    def test_fallback_extraction(self, llm_extractor):
        """Test fallback extraction without LLM."""
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Title</h1>
                <h2>Section 1</h2>
                <p>Content 1</p>
                <h2>Section 2</h2>
                <p>Content 2</p>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Test fallback content extraction
        result = llm_extractor._fallback_extraction(soup, "https://example.com")
        assert result['title'] == "Test Page"
        assert result['document_type'] == 'unknown'
        
        # Test fallback section extraction
        sections = llm_extractor._fallback_section_extraction(soup, "https://example.com")
        assert len(sections) >= 2  # Should find h2 sections
    
    def test_sections_to_text(self, llm_extractor):
        """Test converting sections to text."""
        sections = [
            ContentSection(title="Section 1", content="Content 1", level=1),
            ContentSection(title="Section 2", content="Content 2", level=2)
        ]
        
        text = llm_extractor._sections_to_text(sections)
        assert "# Section 1" in text
        assert "# Section 2" in text
        assert "Content 1" in text
        assert "Content 2" in text


class TestAdaptiveDocumentationScraper:
    """Tests for adaptive documentation scraper."""
    
    @pytest.fixture
    def scraper_config(self):
        """Create test scraper configuration."""
        return ScrapingConfig(
            max_pages=3,
            rate_limit=0.1,
            timeout=10
        )
    
    @pytest.fixture
    def adaptive_scraper(self, scraper_config):
        """Create adaptive scraper instance."""
        return AdaptiveDocumentationScraper(
            config=scraper_config,
            use_ml=True,
            use_llm=False  # Disable LLM for testing
        )
    
    def test_scraper_initialization(self, adaptive_scraper):
        """Test adaptive scraper initialization."""
        assert adaptive_scraper.use_ml is True
        assert adaptive_scraper.use_llm is False
        assert adaptive_scraper.ml_extractor is not None
        assert adaptive_scraper.llm_extractor is None
    
    def test_extraction_stats(self, adaptive_scraper):
        """Test extraction statistics tracking."""
        stats = adaptive_scraper.get_extraction_stats()
        
        assert 'total_extractions' in stats
        assert 'ml_extractions' in stats
        assert 'llm_extractions' in stats
        assert 'fallback_extractions' in stats
        
        # Initially should be zero
        assert stats['total_extractions'] == 0
        assert stats['ml_success_rate'] == 0.0
    
    def test_stats_reset(self, adaptive_scraper):
        """Test resetting extraction statistics."""
        # Simulate some extractions
        adaptive_scraper.extraction_stats['total_extractions'] = 5
        adaptive_scraper.extraction_stats['ml_extractions'] = 3
        
        # Reset stats
        adaptive_scraper.reset_stats()
        
        stats = adaptive_scraper.get_extraction_stats()
        assert stats['total_extractions'] == 0
        assert stats['ml_extractions'] == 0
    
    @pytest.mark.asyncio
    async def test_enhancement_without_html(self, adaptive_scraper):
        """Test enhancement with result that has no HTML."""
        result = ScrapingResult(
            url="https://example.com",
            title="Test",
            content="Test content",
            status=ScrapingStatus.FAILED,
            raw_html=None
        )
        
        # Should return unchanged result
        enhanced = await adaptive_scraper._enhance_with_adaptive_extraction(result)
        assert enhanced == result


# Integration tests
class TestPhase1Integration:
    """Integration tests for Phase 1 components."""
    
    @pytest.mark.asyncio
    async def test_async_to_adaptive_pipeline(self):
        """Test pipeline from async scraper to adaptive extraction."""
        config = ScrapingConfig(max_pages=1, rate_limit=0.1, timeout=10)
        
        # Mock HTML content
        mock_html = """
        <html>
            <head><title>Integration Test</title></head>
            <body>
                <h1>Test Documentation</h1>
                <h2>Installation</h2>
                <p>Install using pip install package</p>
                <h2>Usage Examples</h2>
                <pre><code>import package\npackage.run()</code></pre>
            </body>
        </html>
        """
        
        adaptive_scraper = AdaptiveDocumentationScraper(
            config=config,
            use_ml=True,
            use_llm=False
        )
        
        # Test the enhancement process with mock HTML
        base_result = ScrapingResult(
            url="https://example.com",
            title="Integration Test",
            content="Test content",
            status=ScrapingStatus.COMPLETED,
            raw_html=mock_html
        )
        
        enhanced_result = await adaptive_scraper._enhance_with_adaptive_extraction(base_result)
        
        # Should have ML extraction metadata
        assert enhanced_result.metadata.get('extraction_method') in ['ml', 'fallback']
        assert 'confidence_score' in enhanced_result.metadata
        
        # Should have extracted sections
        assert len(enhanced_result.sections) > 0
        
        await adaptive_scraper.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
