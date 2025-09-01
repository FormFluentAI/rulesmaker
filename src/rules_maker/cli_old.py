"""
CLI interface for Rules Maker.
"""

import click
import json
import yaml
import asyncio
from pathlib import Path
from typing import Optional

from .scrapers import DocumentationScraper, AsyncDocumentationScraper, AdaptiveDocumentationScraper
from .extractors.llm_extractor import LLMConfig, LLMProvider
from .transformers import CursorRuleTransformer, WindsurfRuleTransformer
from .models import ScrapingConfig, TransformationConfig, RuleFormat
from .utils import setup_logging


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.pass_context
def main(ctx, verbose, config):
    """Rules Maker - Transform web documentation into AI coding assistant rules."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    # Load config if provided
    if config:
        ctx.obj['config_file'] = config


@main.command()
@click.argument('url')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format', type=click.Choice(['cursor', 'windsurf', 'json', 'yaml']), 
              default='cursor', help='Output format')
@click.option('--max-pages', type=int, default=10, help='Maximum pages to scrape')
@click.option('--deep', is_flag=True, help='Enable deep scraping (follow links)')
@click.option('--async-scrape', is_flag=True, help='Use async scraper for better performance')
@click.option('--adaptive', is_flag=True, help='Use adaptive scraper with ML/LLM enhancement')
@click.option('--llm-provider', type=click.Choice(['openai', 'anthropic', 'huggingface', 'local']), 
              help='LLM provider for adaptive scraping')
@click.option('--llm-api-key', help='API key for LLM provider')
@click.option('--llm-model', help='LLM model name')
@click.pass_context
def scrape(ctx, url, output, output_format, max_pages, deep, async_scrape, adaptive, 
           llm_provider, llm_api_key, llm_model):
    """Scrape documentation from a URL and generate rules."""
    click.echo(f"Scraping documentation from: {url}")
    
    # Configure scraping
    scraping_config = ScrapingConfig(max_pages=max_pages)
    
    # Choose scraper type
    if adaptive:
        click.echo("🔮 Using adaptive scraper with ML/LLM enhancement")
        
        # Configure LLM if provided
        llm_config = None
        if llm_provider and llm_api_key:
            llm_config = LLMConfig(
                provider=LLMProvider(llm_provider),
                api_key=llm_api_key,
                model_name=llm_model or "gpt-3.5-turbo"
            )
        
        scraper = AdaptiveDocumentationScraper(
            config=scraping_config,
            use_ml=True,
            use_llm=bool(llm_config),
            llm_config=llm_config
        )
        
        # Use async interface for adaptive scraper
        async def run_adaptive_scrape():
            try:
                async with scraper:
                    if deep:
                        click.echo("Deep scraping enabled - following navigation links...")
                        results = await scraper.scrape_documentation_site(url, max_pages)
                    else:
                        results = [await scraper.scrape_url(url)]
                    
                    # Show extraction statistics
                    stats = scraper.get_extraction_stats()
                    click.echo(f"📊 Extraction Statistics:")
                    click.echo(f"   ML success rate: {stats['ml_success_rate']:.2%}")
                    click.echo(f"   LLM success rate: {stats['llm_success_rate']:.2%}")
                    
                    return results
            finally:
                await scraper.close()
        
        results = asyncio.run(run_adaptive_scrape())
        
    elif async_scrape:
        click.echo("🚀 Using async scraper for high performance")
        scraper = AsyncDocumentationScraper(scraping_config)
        
        async def run_async_scrape():
            async with scraper:
                if deep:
                    click.echo("Deep scraping enabled - following navigation links...")
                    return await scraper.scrape_documentation_site(url, max_pages)
                else:
                    return [await scraper.scrape_url(url)]
        
        results = asyncio.run(run_async_scrape())
        
    else:
        click.echo("📄 Using standard synchronous scraper")
        scraper = DocumentationScraper(scraping_config)
        
        if deep:
            click.echo("Deep scraping enabled - following navigation links...")
            results = scraper.scrape_documentation_site(url, max_pages)
        else:
            results = [scraper.scrape_url(url)]
    
    click.echo(f"Scraped {len(results)} pages successfully")
    
    # Transform results
    transformation_config = TransformationConfig(
        rule_format=RuleFormat(output_format)
    )
    
    if output_format == 'cursor':
        transformer = CursorRuleTransformer(transformation_config)
    elif output_format == 'windsurf':
        transformer = WindsurfRuleTransformer(transformation_config)
    else:
        # Generic transformer for JSON/YAML
        from .transformers import RuleTransformer
        transformer = RuleTransformer(transformation_config)
    
    transformed_content = transformer.transform(results)
    
    # Output results
    if output:
        output_path = Path(output)
        output_path.write_text(transformed_content)
        click.echo(f"Rules saved to: {output_path}")
    else:
        click.echo("\n" + "="*50)
        click.echo("GENERATED RULES:")
        click.echo("="*50)
        click.echo(transformed_content)


@main.command()
@click.argument('urls_file', type=click.Path(exists=True))
@click.option('--output-dir', '-d', type=click.Path(), help='Output directory')
@click.option('--format', 'output_format', type=click.Choice(['cursor', 'windsurf', 'json', 'yaml']), 
              default='cursor', help='Output format')
@click.option('--parallel', is_flag=True, help='Enable parallel scraping (uses async scraper)')
@click.option('--adaptive', is_flag=True, help='Use adaptive scraper with ML enhancement')
def batch(urls_file, output_dir, output_format, parallel, adaptive):
    """Scrape multiple URLs from a file."""
    urls_path = Path(urls_file)
    urls = [line.strip() for line in urls_path.read_text().splitlines() if line.strip()]
    
    click.echo(f"Processing {len(urls)} URLs from {urls_file}")
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    # Choose scraper based on options
    if parallel or adaptive:
        if adaptive:
            click.echo("🔮 Using adaptive scraper for batch processing")
            scraper = AdaptiveDocumentationScraper(use_ml=True, use_llm=False)
        else:
            click.echo("🚀 Using async scraper for parallel processing")
            scraper = AsyncDocumentationScraper()
        
        async def run_batch():
            async with scraper:
                # Process all URLs concurrently
                results = await scraper.scrape_multiple(urls)
                
                # Save results
                for i, (url, result) in enumerate(zip(urls, results), 1):
                    click.echo(f"Processing {i}/{len(urls)}: {url}")
                    
                    if output_dir and result.status.value == "completed":
                        # Generate filename from URL
                        filename = url.replace('https://', '').replace('http://', '').replace('/', '_')
                        filename = f"{filename}.{output_format}"
                        
                        output_file = output_path / filename
                        
                        # Transform and save
                        transformer = CursorRuleTransformer() if output_format == 'cursor' else WindsurfRuleTransformer()
                        content = transformer.transform([result])
                        
                        output_file.write_text(content)
                        click.echo(f"  ✅ Saved to: {output_file}")
                    else:
                        click.echo(f"  ❌ Failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                
                if adaptive and hasattr(scraper, 'get_extraction_stats'):
                    stats = scraper.get_extraction_stats()
                    click.echo(f"\n📊 Batch Extraction Statistics:")
                    click.echo(f"   Total processed: {stats['total_extractions']}")
                    click.echo(f"   ML success rate: {stats['ml_success_rate']:.2%}")
        
        asyncio.run(run_batch())
        
    else:
        # Process URLs sequentially with standard scraper
        scraper = DocumentationScraper()
        
        for i, url in enumerate(urls, 1):
            click.echo(f"Processing {i}/{len(urls)}: {url}")
            
            try:
                result = scraper.scrape_url(url)
                
                if output_dir:
                    # Generate filename from URL
                    filename = url.replace('https://', '').replace('http://', '').replace('/', '_')
                    filename = f"{filename}.{output_format}"
                    
                    output_file = output_path / filename
                    
                    # Transform and save
                    transformer = CursorRuleTransformer() if output_format == 'cursor' else WindsurfRuleTransformer()
                    content = transformer.transform([result])
                    
                    output_file.write_text(content)
                    click.echo(f"  ✅ Saved to: {output_file}")
                
            except Exception as e:
                click.echo(f"  ❌ Error processing {url}: {e}", err=True)


@main.group()
def ml():
    """Machine Learning related commands."""
    pass


@ml.command()
@click.argument('training_data_dir', type=click.Path(exists=True))
@click.option('--model-output', '-o', required=True, help='Output path for trained model')
@click.option('--test-split', type=float, default=0.2, help='Test data split ratio')
def train(training_data_dir, model_output, test_split):
    """Train ML extractor on training data."""
    click.echo(f"🎓 Training ML extractor with data from: {training_data_dir}")
    
    from .extractors.ml_extractor import MLContentExtractor
    from .models import TrainingSet, LearningExample, DocumentationType
    
    # Load training data (you'd implement this based on your data format)
    training_dir = Path(training_data_dir)
    examples = []
    
    # Example: load from JSON files
    for json_file in training_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                
            example = LearningExample(
                input_html=data['html'],
                expected_output=data['expected'],
                url=data['url'],
                documentation_type=DocumentationType(data.get('doc_type', 'unknown'))
            )
            examples.append(example)
            
        except Exception as e:
            click.echo(f"❌ Error loading {json_file}: {e}", err=True)
    
    if not examples:
        click.echo("❌ No training examples found")
        return
    
    click.echo(f"📚 Loaded {len(examples)} training examples")
    
    # Create training set
    training_set = TrainingSet(
        name="CLI Training Set",
        description=f"Training data from {training_data_dir}",
        examples=examples,
        documentation_type=DocumentationType.UNKNOWN
    )
    
    # Train model
    ml_extractor = MLContentExtractor(use_transformers=True)
    
    try:
        performance = ml_extractor.train(training_set)
        click.echo(f"✅ Training completed!")
        click.echo(f"   Accuracy: {performance.get('accuracy', 0):.2%}")
        
        # Save model
        ml_extractor.save_model(model_output)
        click.echo(f"💾 Model saved to: {model_output}")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)


@ml.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('test_url')
def test(model_path, test_url):
    """Test trained ML model on a URL."""
    click.echo(f"🧪 Testing ML model from {model_path} on {test_url}")
    
    from .extractors.ml_extractor import MLContentExtractor
    import requests
    from bs4 import BeautifulSoup
    
    # Load model
    ml_extractor = MLContentExtractor()
    ml_extractor.load_model(model_path)
    
    # Fetch and test URL
    try:
        response = requests.get(test_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        result = ml_extractor.extract(soup, test_url)
        
        click.echo(f"📄 Title: {result.get('title', 'N/A')}")
        click.echo(f"📋 Document Type: {result.get('document_type', 'N/A')}")
        click.echo(f"🎯 Confidence: {result.get('confidence_score', 0):.2f}")
        click.echo(f"📑 Sections: {len(result.get('sections', []))}")
        
        # Show sections
        for i, section in enumerate(result.get('sections', [])[:5], 1):
            click.echo(f"  {i}. {section.get('title', 'Untitled')} (Type: {section.get('metadata', {}).get('section_type', 'unknown')})")
        
        if len(result.get('sections', [])) > 5:
            click.echo(f"  ... and {len(result.get('sections', [])) - 5} more sections")
        
    except Exception as e:
        click.echo(f"❌ Test failed: {e}", err=True)


@main.command()
@click.option('--template', '-t', help='Template name to list')
def templates(template):
    """List available templates or show template content."""
    from .templates import TemplateEngine
    
    engine = TemplateEngine()
    
    if template:
        if engine.template_exists(template):
            template_path = engine.template_dir / template
            content = template_path.read_text()
            click.echo(f"Template: {template}")
            click.echo("="*50)
            click.echo(content)
        else:
            click.echo("Deep scraping enabled - following navigation links...")
            results = scraper.scrape_documentation_site(url, max_pages)
        else:
            results = [scraper.scrape_url(url)]
        
        click.echo(f"Scraped {len(results)} pages successfully")
        
        # Transform results
        transformation_config = TransformationConfig(
            rule_format=RuleFormat(output_format)
        )
        
        if output_format == 'cursor':
            transformer = CursorRuleTransformer(transformation_config)
        elif output_format == 'windsurf':
            transformer = WindsurfRuleTransformer(transformation_config)
        else:
            # Generic transformer for JSON/YAML
            from .transformers import RuleTransformer
            transformer = RuleTransformer(transformation_config)
        
        transformed_content = transformer.transform(results)
        
        # Output results
        if output:
            output_path = Path(output)
            output_path.write_text(transformed_content)
            click.echo(f"Rules saved to: {output_path}")
        else:
            click.echo("\n" + "="*50)
            click.echo("GENERATED RULES:")
            click.echo("="*50)
            click.echo(transformed_content)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument('urls_file', type=click.Path(exists=True))
@click.option('--output-dir', '-d', type=click.Path(), help='Output directory')
@click.option('--format', 'output_format', type=click.Choice(['cursor', 'windsurf', 'json', 'yaml']), 
              default='cursor', help='Output format')
@click.option('--parallel', is_flag=True, help='Enable parallel scraping')
def batch(urls_file, output_dir, output_format, parallel):
    """Scrape multiple URLs from a file."""
    urls_path = Path(urls_file)
    urls = [line.strip() for line in urls_path.read_text().splitlines() if line.strip()]
    
    click.echo(f"Processing {len(urls)} URLs from {urls_file}")
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    # Process URLs
    scraper = DocumentationScraper()
    
    for i, url in enumerate(urls, 1):
        click.echo(f"Processing {i}/{len(urls)}: {url}")
        
        try:
            result = scraper.scrape_url(url)
            
            if output_dir:
                # Generate filename from URL
                filename = url.replace('https://', '').replace('http://', '').replace('/', '_')
                filename = f"{filename}.{output_format}"
                
                output_file = output_path / filename
                
                # Transform and save
                transformer = CursorRuleTransformer() if output_format == 'cursor' else WindsurfRuleTransformer()
                content = transformer.transform([result])
                
                output_file.write_text(content)
                click.echo(f"  Saved to: {output_file}")
            
        except Exception as e:
            click.echo(f"  Error processing {url}: {e}", err=True)


@main.command()
@click.option('--template', '-t', help='Template name to list')
def templates(template):
    """List available templates or show template content."""
    from .templates import TemplateEngine
    
    engine = TemplateEngine()
    
    if template:
        if engine.template_exists(template):
            template_path = engine.template_dir / template
            content = template_path.read_text()
            click.echo(f"Template: {template}")
            click.echo("="*50)
            click.echo(content)
        else:
            click.echo(f"Template '{template}' not found", err=True)
    else:
        available_templates = engine.list_templates()
        click.echo("Available templates:")
        for tmpl in available_templates:
            click.echo(f"  - {tmpl}")


@main.command()
@click.option('--format', 'config_format', type=click.Choice(['yaml', 'json']), 
              default='yaml', help='Config file format')
@click.argument('output_file', type=click.Path())
def init_config(config_format, output_file):
    """Initialize a configuration file."""
    config = {
        'scraping': {
            'max_pages': 50,
            'timeout': 30,
            'delay': 1.0,
            'rate_limit': 1.0,
            'user_agent': 'RulesMaker/0.1.0',
            'follow_links': True,
            'respect_robots_txt': True
        },
        'transformation': {
            'rule_format': 'cursor',
            'max_rules': 50,
            'include_examples': True,
            'include_metadata': True
        }
    }
    
    output_path = Path(output_file)
    
    if config_format == 'yaml':
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(config, indent=2)
    
    output_path.write_text(content)
    click.echo(f"Configuration file created: {output_path}")


if __name__ == '__main__':
    main()
