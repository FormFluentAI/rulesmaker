"""
Test basic scraping functionality.
"""

import pytest
from unittest.mock import Mock, patch
import requests

from rules_maker.scrapers import DocumentationScraper
from rules_maker.models import ScrapingConfig, ScrapingStatus


class TestDocumentationScraper:
    """Test cases for DocumentationScraper."""
    
    def test_scraper_initialization(self):
        """Test scraper can be initialized with default config."""
        scraper = DocumentationScraper()
        assert scraper.config is not None
        assert isinstance(scraper.config, ScrapingConfig)
    
    def test_scraper_initialization_with_config(self):
        """Test scraper can be initialized with custom config."""
        config = ScrapingConfig(max_pages=5, timeout=10)
        scraper = DocumentationScraper(config)
        assert scraper.config.max_pages == 5
        assert scraper.config.timeout == 10
    
    @patch('requests.Session.get')
    def test_successful_scraping(self, mock_get):
        """Test successful scraping of a URL."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Test Documentation</title></head>
            <body>
                <h1>API Reference</h1>
                <p>This is test documentation content.</p>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        scraper = DocumentationScraper()
        result = scraper.scrape_url("https://example.com/docs")
        
        assert result.status == ScrapingStatus.COMPLETED
        assert "Test Documentation" in result.title
        assert "API Reference" in result.content
        assert result.url == "https://example.com/docs"
    
    @patch('requests.Session.get')
    def test_failed_scraping(self, mock_get):
        """Test handling of failed HTTP requests."""
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        scraper = DocumentationScraper()
        result = scraper.scrape_url("https://example.com/docs")
        
        assert result.status == ScrapingStatus.FAILED
        assert result.error_message is not None
    
    def test_invalid_url(self):
        """Test handling of invalid URLs."""
        scraper = DocumentationScraper()
        
        with pytest.raises(ValueError):
            scraper.scrape_url("not-a-valid-url")


if __name__ == "__main__":
    pytest.main([__file__])
