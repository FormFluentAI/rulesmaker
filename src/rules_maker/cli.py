"""
CLI interface for Rules Maker.
"""

import click
import json
import yaml
from pathlib import Path
from typing import Optional

from .scrapers import DocumentationScraper, AsyncDocumentationScraper
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
@click.pass_context
def scrape(ctx, url, output, output_format, max_pages, deep):
    """Scrape documentation from a URL and generate rules."""
    click.echo(f"Scraping documentation from: {url}")
    
    # Configure scraping
    scraping_config = ScrapingConfig(max_pages=max_pages)
    scraper = DocumentationScraper(scraping_config)
    
    try:
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
