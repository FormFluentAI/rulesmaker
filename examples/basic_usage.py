"""
Basic usage example for Rules Maker.
"""

from rules_maker import DocumentationScraper, CursorRuleTransformer
from rules_maker.models import ScrapingConfig, TransformationConfig, RuleFormat


def main():
    """Demonstrate basic usage of Rules Maker."""
    
    # Configure scraping
    scraping_config = ScrapingConfig(
        max_pages=5,
        timeout=30,
        delay=1.0
    )
    
    # Configure transformation  
    transformation_config = TransformationConfig(
        rule_format=RuleFormat.CURSOR,
        max_rules=20,
        include_examples=True
    )
    
    # Initialize scraper and transformer
    scraper = DocumentationScraper(scraping_config)
    transformer = CursorRuleTransformer(transformation_config)
    
    # Example URLs (replace with actual documentation URLs)
    example_urls = [
        "https://docs.python.org/3/library/json.html",
        "https://requests.readthedocs.io/en/latest/user/quickstart/",
        "https://flask.palletsprojects.com/en/2.3.x/quickstart/"
    ]
    
    try:
        # Scrape documentation
        print("Scraping documentation...")
        results = []
        
        for url in example_urls:
            print(f"  Scraping: {url}")
            result = scraper.scrape_url(url)
            results.append(result)
            print(f"    Status: {result.status}")
        
        # Transform to rules
        print("\nTransforming to Cursor rules...")
        cursor_rules = transformer.transform(results)
        
        # Save rules
        with open(".cursorrules", "w") as f:
            f.write(cursor_rules)
        
        print("Rules generated and saved to .cursorrules")
        print("\nPreview:")
        print("="*50)
        print(cursor_rules[:500] + "..." if len(cursor_rules) > 500 else cursor_rules)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
