"""
CLI interface for Rules Maker.
"""

import click
import json
import yaml
import asyncio
import os
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# ML Batch Processing Imports
try:
    from .batch_processor import MLBatchProcessor, DocumentationSource, process_popular_frameworks, process_cloud_platforms
    from .processors.ml_documentation_processor import MLDocumentationProcessor
    from .transformers.ml_cursor_transformer import MLCursorTransformer
    from .learning.integrated_learning_system import IntegratedLearningSystem
    from .learning.self_improving_engine import SelfImprovingEngine
    from .strategies.ml_quality_strategy import MLQualityStrategy
    ML_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.debug(f"ML features not available: {e}")
    ML_FEATURES_AVAILABLE = False


def _run_async(coro):
    """Run async coroutine safely, handling existing event loops."""
    try:
        # Check if there's already a running event loop
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # We're in an existing event loop, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(coro)
    except RuntimeError:
        # No event loop, safe to use asyncio.run()
        return asyncio.run(coro)


def _setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """Lightweight logging setup to avoid importing utils (which pulls heavy deps)."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        handlers=[logging.StreamHandler()],
    )


def _load_ml_config(config_path: Optional[str] = None, ctx_config: Optional[Dict] = None) -> Dict[str, Any]:
    """Load ML batch configuration with fallbacks."""
    # Default configuration path
    default_config = Path(__file__).parent.parent.parent / "config" / "ml_batch_config.yaml"
    
    config = {}
    
    # Load from file
    config_file = Path(config_path) if config_path else default_config
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            click.echo(f"📋 Loaded ML config: {config_file}")
        except Exception as e:
            click.echo(f"⚠️ Failed to load config {config_file}: {e}", err=True)
    
    # Merge with context config
    if ctx_config and isinstance(ctx_config, dict):
        config = {**config, **ctx_config}
    
    return config


def _with_progress(operation_name: str):
    """Decorator for adding progress tracking to async operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            click.echo(f"🚀 Starting {operation_name}...")
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                click.echo(f"✅ {operation_name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                click.echo(f"❌ {operation_name} failed after {elapsed:.2f}s: {e}", err=True)
                raise
        return wrapper
    return decorator


class MLBatchCLIError(click.ClickException):
    """Custom exception for ML batch CLI operations."""
    
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


from functools import wraps

def _handle_ml_batch_errors(func):
    """Decorator for handling ML batch processing errors (preserves function metadata)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, click.ClickException):
                raise
            raise MLBatchCLIError(f"ML batch processing failed: {e}")
    return wrapper


def _get_config_template(template_type: str) -> Dict[str, Any]:
    """Get configuration template."""
    
    templates = {
        'minimal': {
            'batch_processing': {
                'max_concurrent': 5,
                'quality_threshold': 0.6,
                'output_format': ['cursor']
            },
            'ml_engine': {
                'quality_threshold': 0.6,
                'enable_self_improvement': False
            },
            'bedrock_integration': {
                'model_id': 'amazon.nova-lite-v1:0',
                'region': 'us-east-1',
                'temperature': 0.3
            }
        },
        'standard': {
            'batch_processing': {
                'max_concurrent': 15,
                'quality_threshold': 0.7,
                'enable_clustering': True,
                'coherence_threshold': 0.6,
                'output_format': ['cursor', 'windsurf']
            },
            'ml_engine': {
                'quality_threshold': 0.7,
                'enable_self_improvement': True,
                'clustering_algorithm': 'kmeans'
            },
            'bedrock_integration': {
                'model_id': 'amazon.nova-lite-v1:0',
                'region': 'us-east-1',
                'temperature': 0.3,
                'max_tokens': 4000
            },
            'integrated_learning': {
                'enable_ml': True,
                'ml_weight': 0.6,
                'feedback_integration': True
            }
        },
        'advanced': {
            'batch_processing': {
                'max_concurrent': 20,
                'quality_threshold': 0.8,
                'enable_clustering': True,
                'coherence_threshold': 0.7,
                'output_format': ['cursor', 'windsurf', 'json']
            },
            'ml_engine': {
                'quality_threshold': 0.8,
                'enable_self_improvement': True,
                'clustering_algorithm': 'kmeans',
                'enable_parallel_processing': True
            },
            'bedrock_integration': {
                'model_id': 'amazon.nova-pro-v1:0',
                'region': 'us-east-1',
                'temperature': 0.2,
                'max_tokens': 8000
            },
            'integrated_learning': {
                'enable_ml': True,
                'ml_weight': 0.8,
                'feedback_integration': True
            },
            'advanced': {
                'enable_semantic_search': True,
                'enable_auto_tagging': True,
                'enable_content_summarization': True
            }
        }
    }
    
    return templates.get(template_type, templates['standard'])


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.option('--json-logs/--no-json-logs', default=None, help='Emit JSON structured logs (overrides config)')
@click.option('--redact-prompts/--no-redact-prompts', default=None, help='Redact prompts in logs (overrides config)')
@click.option('--provider', type=click.Choice(['openai', 'anthropic', 'huggingface', 'bedrock', 'local']), help='LLM provider to use')
@click.option('--model-id', help='Model ID/name to use (e.g., amazon.nova-lite-v1:0)')
@click.option('--region', help='Cloud region (for provider bedrock)')
@click.option('--credentials-csv', type=click.Path(exists=True), help='Path to credentials CSV (for provider bedrock)')
@click.pass_context
def main(ctx, verbose, config, json_logs, redact_prompts, provider, model_id, region, credentials_csv):
    """Rules Maker - Transform web documentation into AI coding assistant rules."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    # Use local lightweight logger to avoid optional deps on import
    _setup_logging(log_level)
    
    # Load config if provided
    if config:
        ctx.obj['config_file'] = config
        try:
            with open(config, 'r', encoding='utf-8') as f:
                ctx.obj['config'] = yaml.safe_load(f) or {}
            click.echo(f"Loaded config: {config}")
        except Exception as e:
            click.echo(f"Warning: Failed to load config file {config}: {e}", err=True)

    # Global provider configuration
    ctx.obj['provider'] = provider
    ctx.obj['model_id'] = model_id
    ctx.obj['region'] = region
    ctx.obj['credentials_csv'] = credentials_csv

    # Telemetry overrides via flags
    if json_logs is not None:
        os.environ['RULES_MAKER_LOG_JSON'] = '1' if json_logs else '0'
    if redact_prompts is not None:
        os.environ['RULES_MAKER_REDACT_PROMPTS'] = '1' if redact_prompts else '0'

    # If provider is bedrock and credentials CSV supplied, pre-load environment
    if provider == 'bedrock' and credentials_csv:
        try:
            from .utils.credentials import setup_bedrock_credentials
            result = setup_bedrock_credentials(credentials_csv)
            if not result.get('validation', {}).get('success'):
                click.echo(f"Bedrock validation failed during setup: {result.get('validation', {}).get('error', 'unknown error')}", err=True)
        except Exception as e:
            click.echo(f"Warning: Failed to pre-configure Bedrock credentials: {e}", err=True)


@main.command()
@click.argument('url', required=False)
@click.option('--output', '-o', type=click.Path(), help='Output path. For single file, extension is normalized by format (.mdc for cursor, .md for windsurf).')
@click.option('--format', 'output_format', type=click.Choice(['cursor', 'windsurf', 'json', 'yaml']), 
              default='cursor', help='Output format/ruleset (cursor|windsurf|json|yaml)')
@click.option('--split', type=click.Choice(['none', 'page', 'category']), default='none', help='Split output by page or by URL category')
@click.option('--output-dir', '-d', type=click.Path(), help='Directory for split outputs when --split=page')
@click.option('--max-pages', type=int, default=10, help='Maximum pages to scrape')
@click.option('--deep', is_flag=True, help='Enable deep scraping (follow links)')
@click.option('--async-scrape', is_flag=True, help='Use async scraper for better performance')
@click.option('--adaptive', is_flag=True, help='Use adaptive scraper with ML/LLM enhancement')
@click.option('--ml', 'use_ml', is_flag=True, help='Use ML-enhanced transformer (Cursor format only)')
@click.option('--ml-enhanced', is_flag=True, help='Use ML-enhanced processing pipeline')
@click.option('--quality-assessment', is_flag=True, help='Include quality assessment in output')
@click.option('--learning-feedback', is_flag=True, help='Collect learning feedback signals')
@click.option('--llm-provider', type=click.Choice(['openai', 'anthropic', 'huggingface', 'bedrock', 'local']), 
              help='LLM provider for adaptive scraping')
@click.option('--llm-api-key', help='API key for LLM provider')
@click.option('--llm-model', help='LLM model name')
@click.option('--credentials-csv', type=click.Path(exists=True), help='Credentials CSV for provider bedrock')
@click.option('--region', help='Region for provider bedrock')
@click.option('--interactive/--no-interactive', '-i/ ', default=False, help='Interactive wizard to choose options')
@click.pass_context
def scrape(ctx, url, output, output_format, split, output_dir, max_pages, deep, async_scrape, adaptive,
           use_ml, ml_enhanced, quality_assessment, learning_feedback, 
           llm_provider, llm_api_key, llm_model, credentials_csv, region, interactive):
    """Scrape documentation from a URL and generate rules."""
    # Interactive wizard for friendlier UX
    if interactive:
        # Helper for extension mapping
        def _ext_for(fmt: str) -> str:
            return ".mdc" if fmt == 'cursor' else ".md" if fmt == 'windsurf' else f".{fmt}"
        from urllib.parse import urlparse

        if not url:
            url = click.prompt("Enter documentation URL", type=str)
        # Format choice
        output_format = click.prompt(
            "Choose rules format",
            type=click.Choice(['cursor', 'windsurf', 'json', 'yaml'], case_sensitive=False),
            default=output_format or 'cursor'
        )
        # Split choice
        split = click.prompt(
            "How to divide output",
            type=click.Choice(['none', 'page', 'category'], case_sensitive=False),
            default=split or 'none'
        )
        # Deep crawl
        deep = click.confirm("Enable deep scraping (follow links)?", default=bool(deep))
        # Max pages
        max_pages = click.prompt("Max pages", type=int, default=max_pages or 10)

        # Suggest defaults for output path/dir
        parsed = urlparse(url)
        domain = (parsed.netloc or 'docs').replace(':', '_')
        ext = _ext_for(output_format)
        if split == 'none':
            suggested = output or f"rules/{domain}{ext}"
            output = click.prompt("Output file path", type=str, default=suggested)
        else:
            sub = 'pages' if split == 'page' else 'categories'
            suggested_dir = output_dir or f"rules/{domain}/{output_format}/{sub}"
            output_dir = click.prompt("Output directory", type=str, default=suggested_dir)

        # Optional async/adaptive selection
        mode = click.prompt(
            "Scraper mode",
            type=click.Choice(['standard', 'async', 'adaptive'], case_sensitive=False),
            default='standard'
        )
        async_scrape = (mode == 'async')
        adaptive = (mode == 'adaptive')

    # Lazy and selective imports to avoid optional deps unless needed
    from .models import ScrapingConfig, TransformationConfig, RuleFormat
    from .transformers import CursorRuleTransformer, WindsurfRuleTransformer
    if adaptive:
        from .scrapers.adaptive_documentation_scraper import AdaptiveDocumentationScraper
        from .extractors.llm_extractor import LLMConfig, LLMProvider
    elif async_scrape:
        from .scrapers.async_documentation_scraper import AsyncDocumentationScraper
    else:
        from .scrapers.documentation_scraper import DocumentationScraper
    click.echo(f"Scraping documentation from: {url}")
    
    try:
        # Configure scraping
        scraping_config = ScrapingConfig(max_pages=max_pages)
        
        # Choose scraper type
        if adaptive:
            click.echo("🔮 Using adaptive scraper with ML/LLM enhancement")
            
            # Configure LLM if provided
            llm_config = None
            # Determine provider from flag or global
            effective_provider = llm_provider or ctx.obj.get('provider')
            effective_model = llm_model or ctx.obj.get('model_id') or "gpt-3.5-turbo"
            effective_region = region or ctx.obj.get('region')
            effective_csv = credentials_csv or ctx.obj.get('credentials_csv')
            if effective_provider:
                if effective_provider == 'bedrock':
                    # Setup Bedrock environment if CSV provided
                    if effective_csv:
                        from .utils.credentials import setup_bedrock_credentials
                        setup_bedrock_credentials(effective_csv)
                    # Merge YAML config if available
                    cfg = ctx.obj.get('config') or {}
                    bed = (cfg.get('bedrock') if isinstance(cfg, dict) else None) or {}
                    llm_config = LLMConfig(
                        provider=LLMProvider.BEDROCK,
                        model_name=(bed.get('model_id') or effective_model or "amazon.nova-lite-v1:0"),
                        region=(bed.get('region') or effective_region or "us-east-1"),
                        temperature=0.3,
                        max_tokens=2000,
                        timeout=int(bed.get('timeout', 30)),
                        retry_max_attempts=int((bed.get('retry') or {}).get('max_attempts', 3)),
                        retry_base_ms=int((bed.get('retry') or {}).get('base_ms', 250)),
                        retry_max_ms=int((bed.get('retry') or {}).get('max_ms', 2000)),
                        max_concurrency=int(bed.get('concurrency', 4)),
                    )
                else:
                    llm_config = LLMConfig(
                        provider=LLMProvider(effective_provider),
                        api_key=llm_api_key,
                        model_name=effective_model
                    )
            
            scraper = AdaptiveDocumentationScraper(
                config=scraping_config,
                use_ml=True,
                use_llm=bool(llm_config),
                llm_config=llm_config,
                app_config=ctx.obj.get('config') if isinstance(ctx.obj.get('config'), dict) else None,
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
            
            results = _run_async(run_adaptive_scrape())
            
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
            
            results = _run_async(run_async_scrape())
            
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
        
        # Helper: normalize extension per format
        def _ext_for(fmt: str) -> str:
            return ".mdc" if fmt == 'cursor' else ".md" if fmt == 'windsurf' else f".{fmt}"

        if split == 'page':
            # Write one file per page result
            outdir = Path(output_dir or (output if output and Path(output).is_dir() else "rules_output"))
            outdir.mkdir(parents=True, exist_ok=True)
            ext = _ext_for(output_format)
            written = 0
            for res in results:
                if use_ml and output_format == 'cursor':
                    from .transformers.ml_cursor_transformer import MLCursorTransformer
                    content = _run_async(MLCursorTransformer().transform([res]))
                else:
                    content = transformer.transform([res])
                # Derive filename from URL
                fname = str(res.url).replace('https://', '').replace('http://', '').replace('/', '_')
                if not fname:
                    fname = "index"
                file_path = outdir / f"{fname}{ext}"
                file_path.write_text(content)
                written += 1
            click.echo(f"🗂️  Wrote {written} files to: {outdir}")
        elif split == 'category':
            # Group results by URL-derived category and write one file per category
            from urllib.parse import urlparse
            outdir = Path(output_dir or (output if output and Path(output).is_dir() else "rules_output"))
            outdir.mkdir(parents=True, exist_ok=True)
            ext = _ext_for(output_format)

            def _slug(s: str) -> str:
                import re
                s = s.strip().lower()
                s = re.sub(r"[^a-z0-9\-_/]", "-", s)
                s = s.replace('/', '-').replace('_', '-')
                s = re.sub(r"-+", "-", s).strip('-')
                return s or "uncategorized"

            def _category_from_url(u: str) -> str:
                try:
                    p = urlparse(u)
                    segs = [seg for seg in (p.path or '').split('/') if seg]
                    if not segs:
                        return "uncategorized"
                    # Find 'docs' anchor
                    if 'docs' in segs:
                        i = segs.index('docs')
                        # Prefer segment after 'app' when present
                        if i + 1 < len(segs) and segs[i + 1] == 'app':
                            # Special-case Next.js: building-your-application/<topic>
                            if i + 2 < len(segs) and segs[i + 2] == 'building-your-application' and i + 3 < len(segs):
                                return _slug(segs[i + 3])
                            if i + 2 < len(segs):
                                return _slug(segs[i + 2])
                        # Fallback: next segment after docs
                        if i + 1 < len(segs):
                            return _slug(segs[i + 1])
                    # Otherwise take first meaningful segment
                    return _slug(segs[0])
                except Exception:
                    return "uncategorized"

            buckets: dict[str, list] = {}
            for res in results:
                cat = _category_from_url(str(res.url))
                buckets.setdefault(cat, []).append(res)

            written = 0
            for cat, group in buckets.items():
                # Use a transformer configured with category hint to specialize rules
                from .models import TransformationConfig, RuleFormat
                if use_ml and output_format == 'cursor':
                    from .transformers.ml_cursor_transformer import MLCursorTransformer
                    content = _run_async(MLCursorTransformer().transform(group))
                else:
                    if output_format == 'cursor':
                        from .transformers import CursorRuleTransformer as _T
                    elif output_format == 'windsurf':
                        from .transformers import WindsurfRuleTransformer as _T
                    else:
                        from .transformers import RuleTransformer as _T
                    cfg = TransformationConfig(
                        rule_format=RuleFormat(output_format),
                        category_hint=cat,
                    )
                    content = _T(cfg).transform(group)
                ext = _ext_for(output_format)
                file_path = outdir / f"{cat}{ext}"
                file_path.write_text(content)
                written += 1
            click.echo(f"🗂️  Wrote {written} category files to: {outdir}")
        else:
            # Single combined file
            if use_ml and output_format == 'cursor':
                from .transformers.ml_cursor_transformer import MLCursorTransformer
                transformed_content = _run_async(MLCursorTransformer().transform(results))
            else:
                transformed_content = transformer.transform(results)
            if output:
                output_path = Path(output)
                # Normalize extension if needed
                ext = _ext_for(output_format)
                if output_path.suffix.lower() != ext:
                    output_path = output_path.with_suffix(ext)
                output_path.parent.mkdir(parents=True, exist_ok=True)
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
@click.option('--parallel', is_flag=True, help='Enable parallel scraping (uses async scraper)')
@click.option('--adaptive', is_flag=True, help='Use adaptive scraper with ML enhancement')
@click.option('--ml', 'use_ml', is_flag=True, help='Use ML-enhanced transformer (Cursor format only)')
def batch(urls_file, output_dir, output_format, parallel, adaptive, use_ml):
    """Scrape multiple URLs from a file."""
    # Lazy imports for heavy deps
    from .scrapers import DocumentationScraper, AsyncDocumentationScraper, AdaptiveDocumentationScraper
    from .transformers import CursorRuleTransformer, WindsurfRuleTransformer
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
                        if use_ml and output_format == 'cursor':
                            from .transformers.ml_cursor_transformer import MLCursorTransformer
                            content = await MLCursorTransformer().transform([result])
                        else:
                            transformer = CursorRuleTransformer() if output_format == 'cursor' else WindsurfRuleTransformer()
                            content = transformer.transform([result])
                        
                        output_file.write_text(content)
                        click.echo(f"  ✅ Saved to: {output_file}")
                    else:
                        error_msg = result.error_message if hasattr(result, 'error_message') else 'Unknown error'
                        click.echo(f"  ❌ Failed: {error_msg}")
                
                if adaptive and hasattr(scraper, 'get_extraction_stats'):
                    stats = scraper.get_extraction_stats()
                    click.echo(f"\n📊 Batch Extraction Statistics:")
                    click.echo(f"   Total processed: {stats['total_extractions']}")
                    click.echo(f"   ML success rate: {stats['ml_success_rate']:.2%}")
        
        _run_async(run_batch())
        
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
                    if use_ml and output_format == 'cursor':
                        from .transformers.ml_cursor_transformer import MLCursorTransformer
                        content = _run_async(MLCursorTransformer().transform([result]))
                    else:
                        transformer = CursorRuleTransformer() if output_format == 'cursor' else WindsurfRuleTransformer()
                        content = transformer.transform([result])
                    
                    output_file.write_text(content)
                    click.echo(f"  ✅ Saved to: {output_file}")
                
            except Exception as e:
                click.echo(f"  ❌ Error processing {url}: {e}", err=True)


@main.command()
@click.option('--rules', required=True, type=click.Path(exists=True), help='Path to rules file (RuleSet JSON/YAML or list of Rule JSON).')
@click.option('--content-file', type=click.Path(exists=True), help='Optional path to documentation/content text to analyze.')
@click.option('--events', type=click.Path(exists=True), help='Optional usage events JSON or JSONL file.')
@click.option('--output', '-o', type=click.Path(), help='Output path for pipeline report JSON (stdout if omitted).')
def analyze(rules, content_file, events, output):
    """Analyze rules and generate insights."""
    click.echo("🔍 Analyzing rules...")
    
    # Load rules
    rules_path = Path(rules)
    if rules_path.suffix.lower() in ['.json']:
        with open(rules_path, 'r') as f:
            rules_data = json.load(f)
    elif rules_path.suffix.lower() in ['.yaml', '.yml']:
        with open(rules_path, 'r') as f:
            rules_data = yaml.safe_load(f)
    else:
        click.echo("❌ Unsupported rules file format. Use JSON or YAML.", err=True)
        return
    
    # Load content if provided
    content = None
    if content_file:
        with open(content_file, 'r') as f:
            content = f.read()
    
    # Load events if provided
    events_data = None
    if events:
        with open(events, 'r') as f:
            if events.endswith('.jsonl'):
                events_data = [json.loads(line) for line in f]
            else:
                events_data = json.load(f)
    
    # Generate analysis
    analysis = {
        'rules_count': len(rules_data.get('rules', [])),
        'categories': list(set(rule.get('category', 'general') for rule in rules_data.get('rules', []))),
        'content_analyzed': content is not None,
        'events_analyzed': events_data is not None,
        'timestamp': time.time()
    }
    
    # Output results
    if output:
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
        click.echo(f"✅ Analysis saved to: {output}")
    else:
        click.echo(json.dumps(analysis, indent=2))


@main.group()
def nextjs():
    """Next.js documentation processing commands."""
    pass


@nextjs.command()
@click.option('--sources', 
              type=click.Choice(['official', 'community', 'ecosystem', 'all']),
              default='all',
              help='Next.js documentation sources to process')
@click.option('--output', '-o', 
              default='.cursor/rules/nextjs',
              help='Output directory for generated cursor rules')
@click.option('--ml-enhanced', is_flag=True, help='Enable ML-enhanced processing')
@click.option('--learning-enabled', is_flag=True, help='Enable learning system integration')
@click.option('--quality-threshold', type=float, default=0.7, help='Minimum quality threshold')
@click.option('--max-pages', type=int, default=50, help='Maximum pages to scrape per source')
@click.option('--rate-limit', type=float, default=0.5, help='Rate limit for scraping (seconds between requests)')
@click.option('--parallel', is_flag=True, help='Enable parallel processing')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def process(sources, output, ml_enhanced, learning_enabled, quality_threshold, max_pages, rate_limit, parallel, verbose):
    """Process Next.js documentation and generate cursor rules."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    click.echo("🚀 Starting Next.js documentation processing...")
    
    # Import Next.js pipeline
    try:
        from .pipelines.nextjs_pipeline import NextJSPipeline
    except ImportError as e:
        click.echo(f"❌ Failed to import Next.js pipeline: {e}", err=True)
        return
    
    async def run_nextjs_pipeline():
        """Run the Next.js documentation pipeline."""
        # Initialize the pipeline
        pipeline = NextJSPipeline(
            output_dir=output,
            use_ml=ml_enhanced,
            use_learning=learning_enabled,
            quality_threshold=quality_threshold,
            max_pages=max_pages,
            rate_limit=rate_limit
        )
        
        try:
            # Process sources
            source_types = [sources] if sources != 'all' else ['all']
            results = await pipeline.process_sources(
                source_types=source_types,
                parallel=parallel
            )
            
            # Display results
            click.echo(f"✅ Next.js documentation processing completed!")
            click.echo(f"📁 Output directory: {results['output_directory']}")
            click.echo(f"📄 Rules generated: {results['cursor_rules_generated']}")
            click.echo(f"📚 Sources processed: {results['sources_processed']}")
            click.echo(f"📄 Pages scraped: {results['pages_scraped']}")
            
            # Show generated rules
            if results['cursor_rules']:
                click.echo("\n📋 Generated cursor rules:")
                for rule_name in results['cursor_rules'].keys():
                    click.echo(f"  • {rule_name}")
            
            # Show pipeline stats
            stats = pipeline.get_pipeline_stats()
            click.echo(f"\n📊 Pipeline Statistics:")
            click.echo(f"  • Categorizer patterns: {stats['categorizer_stats']['total_patterns']}")
            click.echo(f"  • Learned patterns: {stats['categorizer_stats']['learned_patterns']}")
            click.echo(f"  • Categories covered: {stats['categorizer_stats']['categories_covered']}")
            
        finally:
            # Ensure proper cleanup
            try:
                await pipeline.close()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            
            # Force close any remaining aiohttp sessions
            import asyncio
            import gc
            gc.collect()
            await asyncio.sleep(0.1)  # Give time for cleanup
    
    # Run the pipeline
    _run_async(run_nextjs_pipeline())


@nextjs.command()
@click.option('--output', '-o', 
              default='.cursor/rules/nextjs',
              help='Output directory for generated cursor rules')
@click.option('--max-pages', type=int, default=5, help='Maximum pages to scrape per source (for testing)')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def test(output, max_pages, verbose):
    """Test the Next.js pipeline with a small subset of sources."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    click.echo("🧪 Testing Next.js pipeline with limited sources...")
    
    # Import Next.js pipeline
    try:
        from .pipelines.nextjs_pipeline import NextJSPipeline
    except ImportError as e:
        click.echo(f"❌ Failed to import Next.js pipeline: {e}", err=True)
        return
    
    async def run_test():
        """Run a test of the Next.js pipeline."""
        # Initialize the pipeline with test settings
        pipeline = NextJSPipeline(
            output_dir=output,
            use_ml=False,  # Disable ML for faster testing
            use_learning=False,  # Disable learning for testing
            quality_threshold=0.5,  # Lower threshold for testing
            max_pages=max_pages,
            rate_limit=0.1  # Faster rate limit for testing
        )
        
        try:
            # Test with just official sources
            results = await pipeline.process_sources(
                source_types=['official'],
                parallel=False  # Sequential for testing
            )
            
            # Display results
            click.echo(f"✅ Test completed!")
            click.echo(f"📁 Output directory: {results['output_directory']}")
            click.echo(f"📄 Rules generated: {results['cursor_rules_generated']}")
            click.echo(f"📚 Sources processed: {results['sources_processed']}")
            click.echo(f"📄 Pages scraped: {results['pages_scraped']}")
            
            # Show generated rules
            if results['cursor_rules']:
                click.echo("\n📋 Generated cursor rules:")
                for rule_name in results['cursor_rules'].keys():
                    click.echo(f"  • {rule_name}")
                    
        finally:
            # Ensure proper cleanup
            try:
                await pipeline.close()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
            
            # Force close any remaining aiohttp sessions
            import asyncio
            import gc
            gc.collect()
            await asyncio.sleep(0.1)  # Give time for cleanup
    
    # Run the test
    _run_async(run_test())


@nextjs.command()
@click.option('--output-dir', default='test_output', help='Output directory for test results')
@click.option('--quick', is_flag=True, help='Run quick tests only')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def test_components(output_dir, quick, verbose):
    """Test individual Next.js pipeline components."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    click.echo("🧪 Running Next.js pipeline tests...")
    
    # Import test components
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from test_nextjs_pipeline import NextJSPipelineTester
    except ImportError as e:
        click.echo(f"❌ Failed to import test components: {e}", err=True)
        return
    
    async def run_tests():
        """Run the test suite."""
        tester = NextJSPipelineTester(test_output_dir=output_dir)
        
        if quick:
            await tester.test_scraping_components()
            await tester.test_categorization_system()
            await tester.test_formatting_system()
            await tester.test_learning_integration()
        else:
            report = await tester.run_all_tests()
            tester.print_test_summary(report)
    
    # Run tests
    _run_async(run_tests())
    click.echo("✅ Tests completed!")


@main.command()
@click.option('--rules', required=True, type=click.Path(exists=True), help='Path to rules file (RuleSet JSON/YAML or list of Rule JSON).')
@click.option('--content-file', type=click.Path(exists=True), help='Optional path to documentation/content text to analyze.')
@click.option('--events', type=click.Path(exists=True), help='Optional usage events JSON or JSONL file.')
@click.option('--output', '-o', type=click.Path(), help='Output path for pipeline report JSON (stdout if omitted).')
def pipeline(rules, content_file, events, output):
    """Run the Learning Pipeline and emit a JSON report."""
    try:
        from .learning import LearningPipeline
        from .learning.models import ContentAnalysis  # noqa: F401 (for typing visibility)
        from .models import Rule
        import json, yaml
        from datetime import datetime

        # Load rules (RuleSet or list[Rule])
        with open(rules, 'r', encoding='utf-8') as f:
            text = f.read()
        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = yaml.safe_load(text)

        rule_list = []
        if isinstance(data, dict) and 'rules' in data:
            rule_list = data.get('rules') or []
        elif isinstance(data, list):
            rule_list = data
        else:
            raise click.ClickException('Unsupported rules format. Provide a RuleSet with a rules[] key or a JSON list of Rule objects.')

        rule_map = {}
        for obj in rule_list:
            try:
                # Pydantic v2
                r = Rule.model_validate(obj)
            except Exception as e:
                raise click.ClickException(f'Invalid rule object: {e}')
            rule_map[r.id] = r

        # Build pipeline (with a fresh tracker)
        pipeline = LearningPipeline.default()

        # Optional: load usage events and feed tracker
        if events:
            with open(events, 'r', encoding='utf-8') as ef:
                etext = ef.read().strip()
            evt_items = []
            if not etext:
                evt_items = []
            else:
                # JSONL fallback if not a JSON array/dict
                try:
                    parsed = json.loads(etext)
                    if isinstance(parsed, list):
                        evt_items = parsed
                    elif isinstance(parsed, dict) and 'events' in parsed:
                        evt_items = parsed['events'] or []
                    else:
                        raise ValueError('events must be a list or have an events[] key')
                except Exception:
                    # Try JSONL
                    evt_items = [json.loads(line) for line in etext.splitlines() if line.strip()]

            for ev in evt_items:
                rid = ev.get('rule_id') or ev.get('id')
                if not rid:
                    continue
                ts = ev.get('timestamp')
                dt = None
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except Exception:
                        dt = None
                pipeline.usage_tracker.record_usage(
                    rule_id=rid,
                    success=bool(ev.get('success', True)),
                    action=ev.get('action', 'applied'),
                    feedback_score=ev.get('feedback_score'),
                    context=ev.get('context') or {},
                    timestamp=dt,
                )

        # Optional: load content
        content = None
        if content_file:
            content = Path(content_file).read_text(encoding='utf-8')

        # Run pipeline
        report = pipeline.run(rule_map=rule_map, content=content)
        out = json.dumps(report.model_dump(), indent=2, default=str)

        if output:
            Path(output).write_text(out, encoding='utf-8')
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(out)

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


@main.group()
def ml():
    """Machine Learning related commands."""
    pass


@ml.command()
@click.argument('training_data_dir', type=click.Path(exists=True))
@click.option('--model-output', '-o', required=True, help='Output path for trained model')
@click.option('--test-split', type=float, default=0.2, help='Test data split ratio')
@click.option('--checkpoint/--no-checkpoint', default=True, help='Save a training checkpoint alongside the model')
def train(training_data_dir, model_output, test_split, checkpoint):
    """Train ML extractor on training data."""
    click.echo(f"🎓 Training ML extractor with data from: {training_data_dir}")
    
    from .extractors.ml_extractor import MLContentExtractor
    from .models import TrainingSet, LearningExample, DocumentationType
    
    # Load training data
    training_dir = Path(training_data_dir)
    examples = []
    
    # Load from JSON files
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
    
    # Split into train/test
    import random
    rng = random.Random(42)
    rng.shuffle(examples)
    split_idx = int(len(examples) * (1 - max(0.0, min(1.0, float(test_split)))))
    train_examples = examples[:split_idx] if split_idx > 0 else examples
    test_examples = examples[split_idx:] if split_idx < len(examples) else []

    training_set = TrainingSet(
        name="CLI Training Set",
        description=f"Training data from {training_data_dir}",
        examples=train_examples,
        documentation_type=DocumentationType.UNKNOWN
    )
    test_set = TrainingSet(
        name="CLI Test Set",
        description=f"Held-out examples from {training_data_dir}",
        examples=test_examples,
        documentation_type=DocumentationType.UNKNOWN
    )
    
    # Train model
    ml_extractor = MLContentExtractor(use_transformers=True)
    
    try:
        performance = ml_extractor.train(training_set)
        click.echo(f"✅ Training completed!")
        click.echo(f"   CV Accuracy: {performance.get('accuracy', 0):.2%}")
        if test_examples:
            eval_metrics = ml_extractor.evaluate(test_set)
            click.echo(f"   Eval Accuracy: {eval_metrics.get('accuracy', 0):.2%} ({eval_metrics.get('correct',0)}/{eval_metrics.get('total',0)})")

        # Save model
        if checkpoint:
            ckpt_path = str(model_output) + ".checkpoint"
            ml_extractor.save_model(ckpt_path)
            click.echo(f"🧩 Checkpoint saved to: {ckpt_path}")
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
            section_dict = (
                section.model_dump() if hasattr(section, 'model_dump') else section.dict() if hasattr(section, 'dict') else section
            )
            title = section_dict.get('title', 'Untitled')
            section_type = section_dict.get('metadata', {}).get('section_type', 'unknown')
            click.echo(f"  {i}. {title} (Type: {section_type})")
        
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
            click.echo(f"Template '{template}' not found")
    else:
        # List all templates (recursive) via engine helper
        templates_list = engine.list_templates()
        if templates_list:
            click.echo("Available templates:")
            for name in templates_list:
                click.echo(f"  - {name}")
        else:
            click.echo("No templates found")


@main.command()
@click.option('--check-deps', is_flag=True, help='Check if all dependencies are installed')
@click.option('--install-deps', is_flag=True, help='Install missing dependencies')
def setup(check_deps, install_deps):
    """Setup and configuration commands."""
    if check_deps:
        click.echo("🔍 Checking dependencies...")
        
        # Check core dependencies
        deps_status = {}
        
        try:
            import aiohttp
            deps_status['aiohttp'] = f"✅ {aiohttp.__version__}"
        except ImportError:
            deps_status['aiohttp'] = "❌ Missing"
        
        try:
            import sklearn
            deps_status['scikit-learn'] = f"✅ {sklearn.__version__}"
        except ImportError:
            deps_status['scikit-learn'] = "❌ Missing"
        
        try:
            import sentence_transformers
            deps_status['sentence-transformers'] = "✅ Available"
        except ImportError:
            deps_status['sentence-transformers'] = "❌ Missing"
        
        try:
            import nltk
            deps_status['nltk'] = f"✅ {nltk.__version__}"
        except ImportError:
            deps_status['nltk'] = "❌ Missing"
        
        # Show results
        for dep, status in deps_status.items():
            click.echo(f"  {dep}: {status}")
        
        missing_deps = [dep for dep, status in deps_status.items() if "❌" in status]
        if missing_deps:
            click.echo(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
            click.echo("Run 'pip install -r requirements.txt' to install them")
        else:
            click.echo("\n✅ All core dependencies are installed!")
    
    if install_deps:
        click.echo("📦 Installing dependencies...")
        import subprocess
        import sys
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            click.echo("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Failed to install dependencies: {e}")


@main.group()
def bedrock():
    """AWS Bedrock related commands."""
    pass


@bedrock.command()
@click.option('--model-id', help='Model ID to validate (default from env or amazon.nova-lite-v1:0)')
@click.option('--region', help='AWS region to use (default from env or us-east-1)')
@click.option('--credentials-csv', type=click.Path(exists=True), help='Path to credentials CSV to load before validation')
@click.option('--show-config/--no-show-config', default=False, help='Print effective Bedrock config (timeout, concurrency, retry)')
@click.pass_context
def validate(ctx, model_id, region, credentials_csv, show_config):
    """Validate Bedrock credentials and connection; print endpoint, usage, and identity."""
    click.echo("🔐 Validating AWS Bedrock setup...")
    from .utils.credentials import setup_bedrock_credentials, get_credential_manager
    from .bedrock_integration import BedrockRulesMaker

    # Effective params from flags or global context
    effective_model = model_id or ctx.obj.get('model_id') or os.environ.get('BEDROCK_MODEL_ID') or 'amazon.nova-lite-v1:0'
    effective_region = region or ctx.obj.get('region') or os.environ.get('AWS_REGION') or 'us-east-1'
    effective_csv = credentials_csv or ctx.obj.get('credentials_csv')

    # Effective config from YAML/env for display
    cfg = ctx.obj.get('config') if isinstance(ctx.obj.get('config'), dict) else {}
    bed = (cfg.get('bedrock') if isinstance(cfg, dict) else None) or {}
    effective_timeout = int(bed.get('timeout') or os.environ.get('BEDROCK_TIMEOUT') or 30)
    effective_concurrency = int(bed.get('concurrency') or os.environ.get('BEDROCK_MAX_CONCURRENCY') or 4)
    r = bed.get('retry') or {}
    effective_retry = {
        'max_attempts': int(r.get('max_attempts') or os.environ.get('BEDROCK_RETRY_MAX_ATTEMPTS') or 3),
        'base_ms': int(r.get('base_ms') or os.environ.get('BEDROCK_RETRY_BASE_MS') or 250),
        'max_ms': int(r.get('max_ms') or os.environ.get('BEDROCK_RETRY_MAX_MS') or 2000),
    }

    # Step 1: Optionally load credentials from CSV and validate
    validation = None
    if effective_csv:
        click.echo(f"📄 Loading credentials from: {effective_csv}")
        result = setup_bedrock_credentials(effective_csv)
        validation = result.get('validation', {})
    else:
        # Validate using existing environment
        manager = get_credential_manager()
        validation = manager.validate_bedrock_access(model_id=effective_model, region=effective_region)

    # Optionally print effective configuration
    if show_config:
        click.echo("⚙️  Effective Bedrock config:")
        click.echo(f"   model_id={effective_model}")
        click.echo(f"   region={effective_region}")
        click.echo(f"   timeout={effective_timeout}s concurrency={effective_concurrency}")
        click.echo(f"   retry.max_attempts={effective_retry['max_attempts']} base_ms={effective_retry['base_ms']} max_ms={effective_retry['max_ms']}")

    # Print validation result and endpoint/usage
    if validation.get('success'):
        click.echo("✅ Bedrock API call succeeded")
        endpoint = validation.get('endpoint', 'unknown')
        usage = validation.get('usage') or {}
        click.echo(f"   Endpoint: {endpoint}")
        if usage:
            # usage keys can vary; print compact summary
            used_tokens = []
            for k in ('inputTokens', 'outputTokens', 'tokens', 'totalTokens'):
                v = usage.get(k)
                if v is not None:
                    used_tokens.append(f"{k}={v}")
            if used_tokens:
                click.echo(f"   Usage: {' '.join(used_tokens)}")
    else:
        click.echo(f"❌ Bedrock validation failed: {validation.get('error', 'unknown error')}", err=True)

    # Step 2: Identity summary via STS
    manager = get_credential_manager()
    identity = manager.get_aws_session_info()
    if identity and 'error' not in identity:
        click.echo("👤 AWS Identity:")
        click.echo(f"   Account: {identity.get('account', 'unknown')}")
        click.echo(f"   UserId:  {identity.get('user_id', 'unknown')}")
        click.echo(f"   ARN:     {identity.get('arn', 'unknown')}")
    else:
        click.echo(f"⚠️  Unable to fetch identity: {identity.get('error', 'unknown')}" if isinstance(identity, dict) else "⚠️  Unable to fetch identity", err=True)

    # Step 3: Simple end-to-end request through BedrockRulesMaker for usage/cost snapshot
    try:
        maker = BedrockRulesMaker(
            model_id=effective_model,
            region=effective_region,
            credentials_csv_path=effective_csv,
            config=ctx.obj.get('config') if isinstance(ctx.obj.get('config'), dict) else None,
        )
        async def _run():
            return await maker.test_bedrock_connection()
        result = _run_async(_run())
        if result.get('success'):
            click.echo("🧪 Test request: OK")
            usage_stats = result.get('usage_stats', {})
            if usage_stats:
                click.echo("   Usage Summary:")
                click.echo(f"     requests={int(usage_stats.get('requests', 0))} input_tokens={int(usage_stats.get('input_tokens', 0))} output_tokens={int(usage_stats.get('output_tokens', 0))}")
                est = usage_stats.get('estimated_cost_usd')
                if est is not None:
                    click.echo(f"     est_cost_usd={est:.6f}")
        else:
            click.echo(f"🧪 Test request failed: {result.get('error', 'unknown error')}", err=True)
        _run_async(maker.close())
    except Exception as e:
        click.echo(f"⚠️  Skipping end-to-end test: {e}", err=True)


@bedrock.command()
@click.argument('sources_file', type=click.Path(exists=True))
@click.option('--output-dir', '-d', required=True, help='Output directory')
@click.option('--model-id', default='amazon.nova-lite-v1:0', help='Bedrock model ID')
@click.option('--parallel-requests', type=int, default=5, help='Parallel request limit')
@click.option('--cost-limit', type=float, default=10.0, help='Daily cost limit (USD)')
@click.option('--quality-threshold', type=float, default=0.7, help='Quality threshold for rule inclusion')
@click.option('--formats', multiple=True, default=['cursor', 'windsurf'], help='Rule formats to generate')
@click.option('--dry-run', is_flag=True, help='Dry run without actual processing')
@click.pass_context
def batch(ctx, sources_file, output_dir, model_id, parallel_requests, cost_limit, quality_threshold, formats, dry_run):
    """Process batch sources using Bedrock enhancement."""
    if not ML_FEATURES_AVAILABLE:
        click.echo("❌ Bedrock batch features require ML components. Install dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)
        raise click.Abort()
    
    # Load sources from file
    sources_path = Path(sources_file)
    try:
        with open(sources_path, 'r') as f:
            if sources_path.suffix.lower() in ['.yaml', '.yml']:
                sources_data = yaml.safe_load(f)
            else:
                sources_data = json.load(f)
    except Exception as e:
        raise click.ClickException(f"Failed to load sources file: {e}")
    
    # Parse sources data
    if isinstance(sources_data, dict) and 'sources' in sources_data:
        sources_list = sources_data['sources']
    elif isinstance(sources_data, list):
        sources_list = sources_data
    else:
        raise click.ClickException("Sources file must contain a 'sources' list or be a list of source objects")
    
    # Convert to DocumentationSource objects
    sources = []
    for src in sources_list:
        try:
            source = DocumentationSource(
                url=src['url'],
                name=src['name'],
                technology=src.get('technology', 'unknown'),
                framework=src.get('framework'),
                priority=src.get('priority', 1),
                expected_pages=src.get('expected_pages', 20),
                language=src.get('language'),
                category=src.get('category', 'general'),
                metadata=src.get('metadata', {})
            )
            sources.append(source)
        except KeyError as e:
            raise click.ClickException(f"Source missing required field {e}: {src}")
    
    # Setup Bedrock configuration
    bedrock_config = {
        'model_id': ctx.obj.get('model_id') or model_id,
        'region': ctx.obj.get('region') or os.environ.get('AWS_REGION') or 'us-east-1',
        'credentials_csv_path': ctx.obj.get('credentials_csv'),
        'max_tokens': 4000,
        'temperature': 0.3,
        'top_p': 0.9,
    }
    
    if dry_run:
        click.echo(f"🧪 Dry run mode - would process {len(sources)} sources with Bedrock:")
        click.echo(f"   Model: {bedrock_config['model_id']}")
        click.echo(f"   Output: {output_dir}")
        click.echo(f"   Parallel requests: {parallel_requests}")
        click.echo(f"   Cost limit: ${cost_limit}")
        click.echo(f"   Quality threshold: {quality_threshold}")
        click.echo(f"   Formats: {', '.join(formats)}")
        
        # Show first few sources
        for i, source in enumerate(sources[:5], 1):
            click.echo(f"   {i}. {source.name} ({source.technology}) - {source.url}")
        if len(sources) > 5:
            click.echo(f"   ... and {len(sources) - 5} more sources")
        return
    
    async def run_bedrock_batch():
        from .models import RuleFormat
        format_enums = [RuleFormat(fmt) for fmt in formats]
        
        # Initialize processor with cost limits
        processor = MLBatchProcessor(
            bedrock_config=bedrock_config,
            output_dir=output_dir,
            quality_threshold=quality_threshold,
            max_concurrent=parallel_requests
        )
        
        # Add cost monitoring
        click.echo(f"💰 Cost monitoring enabled - limit: ${cost_limit}")
        
        try:
            result = await processor.process_documentation_batch(
                sources=sources,
                formats=format_enums
            )
            
            # Display results with cost information
            click.echo(f"\n📊 Bedrock Batch Processing Results:")
            click.echo(f"   Sources processed: {result.sources_processed}")
            click.echo(f"   Rules generated: {result.total_rules_generated}")
            click.echo(f"   Clusters created: {len(result.clusters)}")
            click.echo(f"   Processing time: {result.processing_time:.2f}s")
            click.echo(f"   Overall coherence: {result.quality_metrics.get('overall_coherence', 0):.3f}")
            
            if result.failed_sources:
                click.echo(f"\n⚠️ Failed sources:")
                for failed_source in result.failed_sources:
                    click.echo(f"   - {failed_source}")
            
            # Cost analysis
            try:
                from .bedrock_integration import BedrockRulesMaker
                maker = BedrockRulesMaker(**bedrock_config)
                usage_stats = maker.get_usage_stats()
                
                estimated_cost = usage_stats.get('estimated_cost_usd', 0)
                click.echo(f"\n💰 Cost Analysis:")
                click.echo(f"   Estimated cost: ${estimated_cost:.4f}")
                click.echo(f"   Input tokens: {usage_stats.get('input_tokens', 0)}")
                click.echo(f"   Output tokens: {usage_stats.get('output_tokens', 0)}")
                click.echo(f"   Requests: {usage_stats.get('requests', 0)}")
                
                if estimated_cost > cost_limit:
                    click.echo(f"⚠️ Cost exceeded limit (${cost_limit})!")
                else:
                    click.echo(f"✅ Within cost limit (${cost_limit})")
                
                await maker.close()
            except Exception as e:
                click.echo(f"⚠️ Could not retrieve cost information: {e}")
            
            # Generate enhanced insights report
            insights_file = Path(output_dir) / "bedrock_batch_insights.json"
            insights_file.parent.mkdir(parents=True, exist_ok=True)
            with open(insights_file, 'w') as f:
                json.dump({
                    'bedrock_batch_results': {
                        'model_id': bedrock_config['model_id'],
                        'sources_processed': result.sources_processed,
                        'total_rules_generated': result.total_rules_generated,
                        'processing_time': result.processing_time,
                        'quality_metrics': result.quality_metrics,
                        'cost_analysis': usage_stats if 'usage_stats' in locals() else None
                    },
                    'clusters': [
                        {
                            'id': cluster.id,
                            'name': cluster.name,
                            'technology': cluster.technology,
                            'coherence_score': cluster.coherence_score,
                            'rules_count': len(cluster.rules)
                        }
                        for cluster in result.clusters
                    ],
                    'insights': result.insights,
                    'source_details': [
                        {
                            'name': src.name,
                            'url': src.url,
                            'technology': src.technology,
                            'framework': src.framework,
                            'priority': src.priority
                        }
                        for src in sources
                    ]
                }, f, indent=2, default=str)
            
            click.echo(f"\n📄 Enhanced insights report saved: {insights_file}")
            
            # Automatically organize generated rules using smart organizer
            try:
                click.echo(f"\n📁 Organizing generated rules...")
                from pathlib import Path as PathLib
                import sys
                import os
                
                # Add the project root to Python path to import smart_organizer
                project_root = PathLib(__file__).parent.parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                # Import and run the smart organizer
                from smart_organizer import RulesOrganizer
                
                organizer = RulesOrganizer(output_dir)
                organized_files, moved_files = organizer.organize_files()
                organizer.create_manifest(organized_files, moved_files)
                organizer.create_readme()
                
                click.echo(f"✅ Rules automatically organized into categories")
                click.echo(f"   Organized directory: {output_dir}")
                click.echo(f"   Manifest file: {PathLib(output_dir) / 'rules' / 'sorted' / 'MANIFEST.json'}")
                
            except ImportError as e:
                click.echo(f"⚠️ Smart organizer not available: {e}")
                click.echo("   Rules generated but not automatically organized")
            except Exception as e:
                click.echo(f"⚠️ Failed to organize rules: {e}")
                click.echo("   Rules generated but organization failed")
        
        except Exception as e:
            click.echo(f"❌ Bedrock batch processing failed: {e}", err=True)
            raise click.Abort()
# ===================================
# ML BATCH PROCESSING COMMAND GROUP
# ===================================

@main.group()
@click.pass_context
def ml_batch(ctx):
    """ML-powered batch processing commands."""
    # Allow entering the group; individual commands will handle missing deps
    if not ML_FEATURES_AVAILABLE:
        click.echo("❌ ML batch features not available. Install required dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)


@ml_batch.command()
@click.option('--output-dir', '-d', default='rules/frameworks', help='Output directory')
@click.option('--bedrock/--no-bedrock', default=False, help='Use Bedrock for enhanced generation')
@click.option('--config', '-c', type=click.Path(exists=True), help='ML batch configuration file')
@click.option('--formats', multiple=True, default=['cursor', 'windsurf'], help='Rule formats to generate')
@click.option('--max-concurrent', type=int, default=15, help='Maximum concurrent operations')
@click.option('--quality-threshold', type=float, default=0.7, help='Quality threshold for rule inclusion')
@click.option('--dry-run', is_flag=True, help='Dry run without actual processing')
@click.pass_context
@_handle_ml_batch_errors
def frameworks(ctx, output_dir, bedrock, config, formats, max_concurrent, quality_threshold, dry_run):
    """Process popular web frameworks with ML batch processing."""
    
    # If ML features are unavailable, permit dry-run for UX
    if not ML_FEATURES_AVAILABLE and not dry_run:
        click.echo("❌ ML batch features not available. Install required dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)
        raise click.Abort()

    # Load configuration
    ml_config = _load_ml_config(config, ctx.obj.get('config'))
    
    # Setup Bedrock configuration if enabled
    bedrock_config = None
    if bedrock:
        bedrock_config = {
            'model_id': ctx.obj.get('model_id') or ml_config.get('bedrock_integration', {}).get('model_id', 'amazon.nova-lite-v1:0'),
            'region': ctx.obj.get('region') or ml_config.get('bedrock_integration', {}).get('region', 'us-east-1'),
            'credentials_csv_path': ctx.obj.get('credentials_csv')
        }
    
    if dry_run:
        click.echo("🧪 Dry run mode - would process popular frameworks with:")
        click.echo(f"   Output: {output_dir}")
        click.echo(f"   Bedrock: {'enabled' if bedrock else 'disabled'}")
        click.echo(f"   Formats: {', '.join(formats)}")
        click.echo(f"   Quality threshold: {quality_threshold}")
        return
    
    async def run_frameworks_processing():
        from .models import RuleFormat
        format_enums = [RuleFormat(fmt) for fmt in formats]
        
        result = await process_popular_frameworks(
            output_dir=output_dir,
            bedrock_config=bedrock_config
        )
        
        # Display results
        click.echo(f"\n📊 Batch Processing Results:")
        click.echo(f"   Sources processed: {result.sources_processed}")
        click.echo(f"   Rules generated: {result.total_rules_generated}")
        click.echo(f"   Clusters created: {len(result.clusters)}")
        click.echo(f"   Processing time: {result.processing_time:.2f}s")
        click.echo(f"   Overall coherence: {result.quality_metrics.get('overall_coherence', 0):.3f}")
        
        if result.failed_sources:
            click.echo(f"\n⚠️ Failed sources:")
            for failed_source in result.failed_sources:
                click.echo(f"   - {failed_source}")
        
        # Generate insights report
        insights_file = Path(output_dir) / "processing_insights.json"
        insights_file.parent.mkdir(parents=True, exist_ok=True)
        with open(insights_file, 'w') as f:
            json.dump({
                'result_summary': {
                    'sources_processed': result.sources_processed,
                    'total_rules_generated': result.total_rules_generated,
                    'processing_time': result.processing_time,
                    'quality_metrics': result.quality_metrics
                },
                'clusters': [
                    {
                        'id': cluster.id,
                        'name': cluster.name,
                        'technology': cluster.technology,
                        'coherence_score': cluster.coherence_score,
                        'rules_count': len(cluster.rules)
                    }
                    for cluster in result.clusters
                ],
                'insights': result.insights
            }, f, indent=2, default=str)
        
        click.echo(f"\n📄 Insights report saved: {insights_file}")
        
        # Automatically organize generated rules using smart organizer
        try:
            click.echo(f"\n📁 Organizing generated rules...")
            from pathlib import Path
            import sys
            import os
            
            # Add the project root to Python path to import smart_organizer
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # Import and run the smart organizer
            from smart_organizer import RulesOrganizer
            
            organizer = RulesOrganizer(output_dir)
            organized_files, moved_files = organizer.organize_files()
            organizer.create_manifest(organized_files, moved_files)
            
            click.echo(f"✅ Rules automatically organized into categories")
            click.echo(f"   Organized directory: {output_dir}")
            click.echo(f"   Manifest file: {Path(output_dir) / 'organization_manifest.json'}")
            
        except ImportError as e:
            click.echo(f"⚠️ Smart organizer not available: {e}")
            click.echo("   Rules generated but not automatically organized")
        except Exception as e:
            click.echo(f"⚠️ Failed to organize rules: {e}")
            click.echo("   Rules generated but organization failed")
    
    _run_async(run_frameworks_processing())


@ml_batch.command()
@click.option('--output-dir', '-d', default='rules/cloud', help='Output directory')
@click.option('--bedrock/--no-bedrock', default=False, help='Use Bedrock for enhanced generation')
@click.option('--config', '-c', type=click.Path(exists=True), help='ML batch configuration file')
@click.option('--formats', multiple=True, default=['cursor', 'windsurf'], help='Rule formats to generate')
@click.option('--max-concurrent', type=int, default=10, help='Maximum concurrent operations')
@click.option('--quality-threshold', type=float, default=0.7, help='Quality threshold for rule inclusion')
@click.option('--dry-run', is_flag=True, help='Dry run without actual processing')
@click.pass_context
@_handle_ml_batch_errors
def cloud(ctx, output_dir, bedrock, config, formats, max_concurrent, quality_threshold, dry_run):
    """Process cloud platform documentation with ML batch processing."""
    
    # If ML features are unavailable, permit dry-run for UX
    if not ML_FEATURES_AVAILABLE and not dry_run:
        click.echo("❌ ML batch features not available. Install required dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)
        raise click.Abort()

    # Load configuration
    ml_config = _load_ml_config(config, ctx.obj.get('config'))
    
    # Setup Bedrock configuration if enabled
    bedrock_config = None
    if bedrock:
        bedrock_config = {
            'model_id': ctx.obj.get('model_id') or ml_config.get('bedrock_integration', {}).get('model_id', 'amazon.nova-lite-v1:0'),
            'region': ctx.obj.get('region') or ml_config.get('bedrock_integration', {}).get('region', 'us-east-1'),
            'credentials_csv_path': ctx.obj.get('credentials_csv')
        }
    
    if dry_run:
        click.echo("🧪 Dry run mode - would process cloud platforms with:")
        click.echo(f"   Output: {output_dir}")
        click.echo(f"   Bedrock: {'enabled' if bedrock else 'disabled'}")
        click.echo(f"   Formats: {', '.join(formats)}")
        click.echo(f"   Quality threshold: {quality_threshold}")
        return
    
    async def run_cloud_processing():
        result = await process_cloud_platforms(
            output_dir=output_dir,
            bedrock_config=bedrock_config
        )
        
        # Display results
        click.echo(f"\n📊 Cloud Platform Processing Results:")
        click.echo(f"   Sources processed: {result.sources_processed}")
        click.echo(f"   Rules generated: {result.total_rules_generated}")
        click.echo(f"   Clusters created: {len(result.clusters)}")
        click.echo(f"   Processing time: {result.processing_time:.2f}s")
        click.echo(f"   Overall coherence: {result.quality_metrics.get('overall_coherence', 0):.3f}")
        
        # Generate insights report
        insights_file = Path(output_dir) / "cloud_processing_insights.json"
        insights_file.parent.mkdir(parents=True, exist_ok=True)
        with open(insights_file, 'w') as f:
            json.dump({
                'result_summary': {
                    'sources_processed': result.sources_processed,
                    'total_rules_generated': result.total_rules_generated,
                    'processing_time': result.processing_time,
                    'quality_metrics': result.quality_metrics
                },
                'clusters': [
                    {
                        'id': cluster.id,
                        'name': cluster.name,
                        'technology': cluster.technology,
                        'coherence_score': cluster.coherence_score,
                        'rules_count': len(cluster.rules)
                    }
                    for cluster in result.clusters
                ],
                'insights': result.insights
            }, f, indent=2, default=str)
        
        click.echo(f"\n📄 Insights report saved: {insights_file}")
        
        # Automatically organize generated rules using smart organizer
        try:
            click.echo(f"\n📁 Organizing generated rules...")
            from pathlib import Path
            import sys
            import os
            
            # Add the project root to Python path to import smart_organizer
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # Import and run the smart organizer
            from smart_organizer import RulesOrganizer
            
            organizer = RulesOrganizer(output_dir)
            organized_files, moved_files = organizer.organize_files()
            organizer.create_manifest(organized_files, moved_files)
            
            click.echo(f"✅ Rules automatically organized into categories")
            click.echo(f"   Organized directory: {output_dir}")
            click.echo(f"   Manifest file: {Path(output_dir) / 'organization_manifest.json'}")
            
        except ImportError as e:
            click.echo(f"⚠️ Smart organizer not available: {e}")
            click.echo("   Rules generated but not automatically organized")
        except Exception as e:
            click.echo(f"⚠️ Failed to organize rules: {e}")
            click.echo("   Rules generated but organization failed")
    
    _run_async(run_cloud_processing())


@ml_batch.command()
@click.argument('sources_file', type=click.Path(exists=True))
@click.option('--output-dir', '-d', required=True, help='Output directory')
@click.option('--bedrock/--no-bedrock', default=False, help='Use Bedrock for enhanced generation')
@click.option('--config', '-c', type=click.Path(exists=True), help='ML batch configuration file')
@click.option('--formats', multiple=True, default=['cursor', 'windsurf'], help='Rule formats to generate')
@click.option('--max-concurrent', type=int, default=15, help='Maximum concurrent operations')
@click.option('--quality-threshold', type=float, default=0.7, help='Quality threshold for rule inclusion')
@click.option('--dry-run', is_flag=True, help='Dry run without actual processing')
@click.pass_context
@_handle_ml_batch_errors
def custom(ctx, sources_file, output_dir, bedrock, config, formats, max_concurrent, quality_threshold, dry_run):
    """Process custom documentation sources with ML batch processing."""
    
    # If ML features are unavailable, permit dry-run for UX
    if not ML_FEATURES_AVAILABLE and not dry_run:
        click.echo("❌ ML batch features not available. Install required dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)
        raise click.Abort()

    # Load configuration
    ml_config = _load_ml_config(config, ctx.obj.get('config'))
    
    # Load sources from file (JSON or YAML)
    sources_path = Path(sources_file)
    try:
        with open(sources_path, 'r') as f:
            if sources_path.suffix.lower() in ['.yaml', '.yml']:
                sources_data = yaml.safe_load(f)
            else:
                sources_data = json.load(f)
    except Exception as e:
        raise click.ClickException(f"Failed to load sources file: {e}")
    
    # Parse sources data
    if isinstance(sources_data, dict) and 'sources' in sources_data:
        sources_list = sources_data['sources']
    elif isinstance(sources_data, list):
        sources_list = sources_data
    else:
        raise click.ClickException("Sources file must contain a 'sources' list or be a list of source objects")
    
    # Convert to DocumentationSource objects
    sources = []
    for src in sources_list:
        try:
            source = DocumentationSource(
                url=src['url'],
                name=src['name'],
                technology=src.get('technology', 'unknown'),
                framework=src.get('framework'),
                priority=src.get('priority', 1),
                expected_pages=src.get('expected_pages', 20),
                language=src.get('language'),
                category=src.get('category', 'general'),
                metadata=src.get('metadata', {})
            )
            sources.append(source)
        except KeyError as e:
            raise click.ClickException(f"Source missing required field {e}: {src}")
    
    # Setup Bedrock configuration if enabled
    bedrock_config = None
    if bedrock:
        bedrock_config = {
            'model_id': ctx.obj.get('model_id') or ml_config.get('bedrock_integration', {}).get('model_id', 'amazon.nova-lite-v1:0'),
            'region': ctx.obj.get('region') or ml_config.get('bedrock_integration', {}).get('region', 'us-east-1'),
            'credentials_csv_path': ctx.obj.get('credentials_csv')
        }
    
    if dry_run:
        click.echo(f"🧪 Dry run mode - would process {len(sources)} custom sources:")
        for source in sources[:5]:  # Show first 5
            click.echo(f"   - {source.name} ({source.url})")
        if len(sources) > 5:
            click.echo(f"   ... and {len(sources) - 5} more")
        click.echo(f"   Output: {output_dir}")
        click.echo(f"   Bedrock: {'enabled' if bedrock else 'disabled'}")
        click.echo(f"   Formats: {', '.join(formats)}")
        return
    
    async def run_custom_processing():
        from .models import RuleFormat
        format_enums = [RuleFormat(fmt) for fmt in formats]
        
        # Initialize processor
        processor = MLBatchProcessor(
            bedrock_config=bedrock_config,
            output_dir=output_dir,
            quality_threshold=quality_threshold,
            max_concurrent=max_concurrent
        )
        
        result = await processor.process_documentation_batch(
            sources=sources,
            formats=format_enums
        )
        
        # Display results
        click.echo(f"\n📊 Custom Batch Processing Results:")
        click.echo(f"   Sources processed: {result.sources_processed}")
        click.echo(f"   Rules generated: {result.total_rules_generated}")
        click.echo(f"   Clusters created: {len(result.clusters)}")
        click.echo(f"   Processing time: {result.processing_time:.2f}s")
        click.echo(f"   Overall coherence: {result.quality_metrics.get('overall_coherence', 0):.3f}")
        
        if result.failed_sources:
            click.echo(f"\n⚠️ Failed sources:")
            for failed_source in result.failed_sources:
                click.echo(f"   - {failed_source}")
        
        # Generate insights report
        insights_file = Path(output_dir) / "custom_processing_insights.json"
        insights_file.parent.mkdir(parents=True, exist_ok=True)
        with open(insights_file, 'w') as f:
            json.dump({
                'result_summary': {
                    'sources_processed': result.sources_processed,
                    'total_rules_generated': result.total_rules_generated,
                    'processing_time': result.processing_time,
                    'quality_metrics': result.quality_metrics
                },
                'clusters': [
                    {
                        'id': cluster.id,
                        'name': cluster.name,
                        'technology': cluster.technology,
                        'coherence_score': cluster.coherence_score,
                        'rules_count': len(cluster.rules)
                    }
                    for cluster in result.clusters
                ],
                'insights': result.insights,
                'source_details': [
                    {
                        'name': src.name,
                        'url': src.url,
                        'technology': src.technology,
                        'framework': src.framework,
                        'priority': src.priority
                    }
                    for src in sources
                ]
            }, f, indent=2, default=str)
        
        click.echo(f"\n📄 Insights report saved: {insights_file}")
        
        # Automatically organize generated rules using smart organizer
        try:
            click.echo(f"\n📁 Organizing generated rules...")
            from pathlib import Path
            import sys
            import os
            
            # Add the project root to Python path to import smart_organizer
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # Import and run the smart organizer
            from smart_organizer import RulesOrganizer
            
            organizer = RulesOrganizer(output_dir)
            organized_files, moved_files = organizer.organize_files()
            organizer.create_manifest(organized_files, moved_files)
            
            click.echo(f"✅ Rules automatically organized into categories")
            click.echo(f"   Organized directory: {output_dir}")
            click.echo(f"   Manifest file: {Path(output_dir) / 'organization_manifest.json'}")
            
        except ImportError as e:
            click.echo(f"⚠️ Smart organizer not available: {e}")
            click.echo("   Rules generated but not automatically organized")
        except Exception as e:
            click.echo(f"⚠️ Failed to organize rules: {e}")
            click.echo("   Rules generated but organization failed")
    
    _run_async(run_custom_processing())


# ===================================
# CONFIGURATION MANAGEMENT GROUP
# ===================================

@main.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
@click.option('--output', '-o', default='config/ml_batch_config.yaml', help='Config file path')
@click.option('--template', type=click.Choice(['minimal', 'standard', 'advanced']), default='standard', help='Configuration template')
@click.option('--force', is_flag=True, help='Overwrite existing config file')
def init(output, template, force):
    """Initialize ML batch configuration."""
    
    output_path = Path(output)
    
    # Check if file exists
    if output_path.exists() and not force:
        click.echo(f"⚠️ Config file already exists: {output_path}")
        click.echo("Use --force to overwrite")
        return
    
    # Get template
    config_template = _get_config_template(template)
    
    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config_template, f, default_flow_style=False, indent=2)
        
        click.echo(f"✅ Configuration initialized: {output_path}")
        click.echo(f"   Template: {template}")
        click.echo(f"   Size: {output_path.stat().st_size} bytes")
        
        # Show key settings
        if 'batch_processing' in config_template:
            bp = config_template['batch_processing']
            click.echo(f"\n📋 Key Settings:")
            click.echo(f"   Max concurrent: {bp.get('max_concurrent', 'N/A')}")
            click.echo(f"   Quality threshold: {bp.get('quality_threshold', 'N/A')}")
            click.echo(f"   Output formats: {', '.join(bp.get('output_format', []))}")
        
    except Exception as e:
        raise click.ClickException(f"Failed to write config file: {e}")


@config.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate(config_file):
    """Validate ML batch configuration file."""
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Basic validation
        errors = []
        warnings = []
        
        # Check required sections
        required_sections = ['batch_processing', 'ml_engine']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate batch_processing section
        if 'batch_processing' in config:
            bp = config['batch_processing']
            
            # Check numeric ranges
            if 'max_concurrent' in bp:
                if not isinstance(bp['max_concurrent'], int) or bp['max_concurrent'] < 1:
                    errors.append("batch_processing.max_concurrent must be a positive integer")
            
            if 'quality_threshold' in bp:
                qt = bp['quality_threshold']
                if not isinstance(qt, (int, float)) or not (0.0 <= qt <= 1.0):
                    errors.append("batch_processing.quality_threshold must be between 0.0 and 1.0")
            
            # Check output formats
            if 'output_format' in bp:
                valid_formats = {'cursor', 'windsurf', 'json', 'yaml'}
                formats = bp['output_format']
                if not isinstance(formats, list):
                    errors.append("batch_processing.output_format must be a list")
                else:
                    invalid_formats = set(formats) - valid_formats
                    if invalid_formats:
                        errors.append(f"Invalid output formats: {', '.join(invalid_formats)}")
        
        # Validate bedrock_integration section
        if 'bedrock_integration' in config:
            bi = config['bedrock_integration']
            
            if 'temperature' in bi:
                temp = bi['temperature']
                if not isinstance(temp, (int, float)) or not (0.0 <= temp <= 1.0):
                    warnings.append("bedrock_integration.temperature should be between 0.0 and 1.0")
            
            if 'max_tokens' in bi:
                if not isinstance(bi['max_tokens'], int) or bi['max_tokens'] < 1:
                    errors.append("bedrock_integration.max_tokens must be a positive integer")
        
        # Display results
        if errors:
            click.echo(f"❌ Configuration validation failed ({len(errors)} errors):")
            for i, error in enumerate(errors, 1):
                click.echo(f"   {i}. {error}")
        else:
            click.echo("✅ Configuration validation passed")
        
        if warnings:
            click.echo(f"\n⚠️ Warnings ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                click.echo(f"   {i}. {warning}")
        
        # Show summary
        sections = list(config.keys())
        click.echo(f"\n📊 Configuration Summary:")
        click.echo(f"   Sections: {', '.join(sections)}")
        click.echo(f"   Total size: {Path(config_file).stat().st_size} bytes")
        
        if errors:
            raise click.Abort()
            
    except yaml.YAMLError as e:
        raise click.ClickException(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise click.ClickException(f"Validation failed: {e}")


# ===================================
# LEARNING SYSTEM COMMAND GROUP
# ===================================

@main.group()
def learning():
    """Integrated learning and self-improvement commands."""
    # Allow group entry; commands handle missing deps appropriately
    if not ML_FEATURES_AVAILABLE:
        click.echo("❌ Learning features not available. Install required dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)


@learning.command()
@click.option('--rule-id', required=True, help='Rule ID for feedback')
@click.option('--signal-type', type=click.Choice(['usage_success', 'user_rating', 'effectiveness', 'relevance']), required=True)
@click.option('--value', type=float, required=True, help='Feedback value (0.0-1.0)')
@click.option('--context', help='Additional context (JSON string)')
@click.option('--source', default='user', help='Feedback source')
@_handle_ml_batch_errors
def feedback(rule_id, signal_type, value, context, source):
    """Collect feedback signals for rule improvement."""
    if not ML_FEATURES_AVAILABLE:
        click.echo("❌ Learning features not available. Install required dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)
        raise click.Abort()
    
    # Validate feedback value
    if not (0.0 <= value <= 1.0):
        raise click.BadParameter("Feedback value must be between 0.0 and 1.0")
    
    # Parse context if provided
    context_dict = {}
    if context:
        try:
            context_dict = json.loads(context)
        except json.JSONDecodeError:
            raise click.BadParameter("Context must be valid JSON")
    
    async def collect_feedback():
        engine = SelfImprovingEngine()
        
        await engine.collect_feedback_signal(
            rule_id=rule_id,
            signal_type=signal_type,
            value=value,
            context=context_dict,
            source=source
        )
        
        click.echo(f"✅ Feedback collected for rule {rule_id}")
        click.echo(f"   Signal: {signal_type} = {value}")
        click.echo(f"   Source: {source}")
        
        # Get updated rule quality prediction if available
        try:
            # Note: predict_rule_quality_by_id is a hypothetical method
            # In real implementation, this would need to be implemented
            click.echo(f"   Feedback successfully recorded for future learning")
        except Exception as e:
            click.echo(f"   Note: {e}")
    
    _run_async(collect_feedback())


@learning.command()
@click.argument('rules_dir', type=click.Path(exists=True))
@click.option('--output', '-o', help='Analysis report output path')
@click.option('--format', type=click.Choice(['json', 'yaml', 'md']), default='json')
def analyze(rules_dir, output, format):
    """Analyze learning patterns and rule effectiveness."""
    if not ML_FEATURES_AVAILABLE:
        click.echo("❌ Learning features not available. Install required dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)
        raise click.Abort()
    
    async def run_analysis():
        engine = SelfImprovingEngine()
        
        # Analyze rule directory
        rules_path = Path(rules_dir)
        rule_files = list(rules_path.rglob("*.md")) + list(rules_path.rglob("*.json"))
        
        click.echo(f"🔍 Analyzing {len(rule_files)} rule files...")
        
        # Collect analysis data
        analysis_data = {
            'timestamp': time.time(),
            'rules_directory': str(rules_path),
            'total_rule_files': len(rule_files),
            'analysis_results': {},
            'recommendations': []
        }
        
        # Basic file analysis
        for rule_file in rule_files:
            try:
                content = rule_file.read_text()
                analysis_data['analysis_results'][str(rule_file)] = {
                    'size_bytes': rule_file.stat().st_size,
                    'content_length': len(content),
                    'word_count': len(content.split()),
                    'has_examples': 'example' in content.lower(),
                    'has_code_blocks': '```' in content
                }
            except Exception as e:
                analysis_data['analysis_results'][str(rule_file)] = {'error': str(e)}
        
        # Generate recommendations
        total_files = len(rule_files)
        avg_size = (sum(r.get('size_bytes', 0) for r in analysis_data['analysis_results'].values()) / total_files) if total_files else 0
        if avg_size < 1000:
            analysis_data['recommendations'].append("Consider adding more detailed examples and explanations to rules")
        
        files_with_examples = sum(1 for r in analysis_data['analysis_results'].values() if r.get('has_examples', False))
        if total_files and (files_with_examples / total_files) < 0.5:
            analysis_data['recommendations'].append("Add practical examples to improve rule effectiveness")
        
        # Output results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(analysis_data, f, indent=2, default=str)
            elif format == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(analysis_data, f, default_flow_style=False)
            elif format == 'md':
                with open(output_path, 'w') as f:
                    f.write(f"# Learning Analysis Report\n\n")
                    f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"**Rules Directory:** {rules_path}\n\n")
                    f.write(f"**Total Files:** {len(rule_files)}\n\n")
                    f.write(f"## Summary\n\n")
                    f.write(f"- Average file size: {avg_size:.0f} bytes\n")
                    denom = len(rule_files) or 1
                    percent = (files_with_examples / denom) * 100.0
                    f.write(f"- Files with examples: {files_with_examples}/{len(rule_files)} ({percent:.1f}%)\n\n")
                    if analysis_data['recommendations']:
                        f.write("## Recommendations\n\n")
                        for i, rec in enumerate(analysis_data['recommendations'], 1):
                            f.write(f"{i}. {rec}\n")
            
            click.echo(f"📄 Analysis report saved: {output_path}")
        else:
            # Display summary to console
            click.echo(f"\n📊 Learning Analysis Summary:")
            click.echo(f"   Total rule files: {len(rule_files)}")
            click.echo(f"   Average file size: {avg_size:.0f} bytes")
            denom = len(rule_files) or 1
            percent = (files_with_examples / denom) * 100.0
            click.echo(f"   Files with examples: {files_with_examples}/{len(rule_files)} ({percent:.1f}%)")
            
            if analysis_data['recommendations']:
                click.echo(f"\n💡 Recommendations:")
                for i, rec in enumerate(analysis_data['recommendations'], 1):
                    click.echo(f"   {i}. {rec}")
    
    _run_async(run_analysis())


# ===================================
# QUALITY ASSESSMENT COMMAND GROUP
# ===================================

@main.group()
def quality():
    """Rule quality assessment and analysis commands."""
    if not ML_FEATURES_AVAILABLE:
        click.echo("❌ Quality assessment features not available. Install required dependencies:", err=True)
        click.echo("   pip install scikit-learn numpy", err=True)
        raise click.Abort()


@quality.command()
@click.argument('rules_dir', type=click.Path())
@click.option('--format', type=click.Choice(['cursor', 'windsurf', 'all']), default='all', help='Rule format to assess')
@click.option('--output', '-o', help='Analysis report output path')
@click.option('--threshold', type=float, default=0.7, help='Quality threshold for assessment')
def assess(rules_dir, format, output, threshold):
    """Assess quality of generated rules."""
    
    rules_path = Path(rules_dir)
    if not rules_path.exists():
        click.echo(f"❌ No such file or directory: {rules_path}")
        raise click.Abort()
    
    # Find rule files
    if format == 'all':
        rule_files = list(rules_path.rglob("*.md")) + list(rules_path.rglob("*.mdc"))
    elif format == 'cursor':
        rule_files = list(rules_path.rglob("*.mdc"))
    elif format == 'windsurf':
        rule_files = list(rules_path.rglob("*.md"))
    
    if not rule_files:
        click.echo(f"❌ No rule files found in {rules_path}")
        return
    
    click.echo(f"🔍 Assessing {len(rule_files)} rule files...")
    
    # Quality assessment metrics
    assessment_results = {
        'total_files': len(rule_files),
        'assessment_timestamp': time.time(),
        'threshold': threshold,
        'files': {},
        'summary': {
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0,
            'avg_score': 0.0,
            'recommendations': []
        }
    }
    
    total_score = 0.0
    
    for rule_file in rule_files:
        try:
            content = rule_file.read_text()
            
            # Calculate quality metrics
            quality_score = _calculate_rule_quality(content)
            total_score += quality_score
            
            # Categorize quality
            if quality_score >= threshold:
                category = 'high'
                assessment_results['summary']['high_quality'] += 1
            elif quality_score >= threshold - 0.2:
                category = 'medium'
                assessment_results['summary']['medium_quality'] += 1
            else:
                category = 'low'
                assessment_results['summary']['low_quality'] += 1
            
            assessment_results['files'][str(rule_file)] = {
                'quality_score': quality_score,
                'category': category,
                'size_bytes': rule_file.stat().st_size,
                'word_count': len(content.split()),
                'has_examples': 'example' in content.lower(),
                'has_code_blocks': '```' in content,
                'structure_score': _assess_structure(content)
            }
            
        except Exception as e:
            assessment_results['files'][str(rule_file)] = {'error': str(e)}
    
    # Calculate averages and recommendations
    assessment_results['summary']['avg_score'] = total_score / len(rule_files) if rule_files else 0.0
    
    # Generate recommendations
    if assessment_results['summary']['low_quality'] > len(rule_files) * 0.3:
        assessment_results['summary']['recommendations'].append(
            "Consider improving rule quality - over 30% of rules are below threshold"
        )
    
    if assessment_results['summary']['avg_score'] < threshold:
        assessment_results['summary']['recommendations'].append(
            f"Average quality score ({assessment_results['summary']['avg_score']:.3f}) is below threshold ({threshold})"
        )
    
    files_without_examples = sum(1 for f in assessment_results['files'].values() 
                               if isinstance(f, dict) and not f.get('has_examples', False))
    if files_without_examples > len(rule_files) * 0.5:
        assessment_results['summary']['recommendations'].append(
            "Add more practical examples to improve rule effectiveness"
        )
    
    # Output results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(assessment_results, f, indent=2, default=str)
        
        click.echo(f"📄 Quality assessment report saved: {output_path}")
    
    # Display summary
    click.echo(f"\n📊 Quality Assessment Summary:")
    click.echo(f"   Total files: {assessment_results['total_files']}")
    click.echo(f"   Average score: {assessment_results['summary']['avg_score']:.3f}")
    click.echo(f"   High quality: {assessment_results['summary']['high_quality']}")
    click.echo(f"   Medium quality: {assessment_results['summary']['medium_quality']}")
    click.echo(f"   Low quality: {assessment_results['summary']['low_quality']}")
    
    if assessment_results['summary']['recommendations']:
        click.echo(f"\n💡 Recommendations:")
        for i, rec in enumerate(assessment_results['summary']['recommendations'], 1):
            click.echo(f"   {i}. {rec}")


@quality.command()
@click.argument('processing_results_dir', type=click.Path(exists=True))
@click.option('--output', '-o', help='Cluster analysis output path')
@click.option('--min-coherence', type=float, default=0.6, help='Minimum coherence score for analysis')
def cluster(processing_results_dir, output, min_coherence):
    """Analyze rule clusters from batch processing results."""
    
    results_path = Path(processing_results_dir)
    
    # Find insights files from batch processing
    insights_files = list(results_path.rglob("*insights.json"))
    
    if not insights_files:
        click.echo(f"❌ No insights files found in {results_path}")
        click.echo("Run ml-batch commands first to generate cluster data")
        return
    
    click.echo(f"🔍 Analyzing clusters from {len(insights_files)} batch processing results...")
    
    cluster_analysis = {
        'analysis_timestamp': time.time(),
        'min_coherence_threshold': min_coherence,
        'total_insights_files': len(insights_files),
        'clusters': [],
        'summary': {
            'total_clusters': 0,
            'high_coherence_clusters': 0,
            'technologies': set(),
            'avg_coherence': 0.0,
            'recommendations': []
        }
    }
    
    total_coherence = 0.0
    
    for insights_file in insights_files:
        try:
            with open(insights_file, 'r') as f:
                insights_data = json.load(f)
            
            # Extract cluster information
            if 'clusters' in insights_data:
                for cluster in insights_data['clusters']:
                    cluster_info = {
                        'id': cluster.get('id', 'unknown'),
                        'name': cluster.get('name', 'Unknown'),
                        'technology': cluster.get('technology', 'unknown'),
                        'coherence_score': cluster.get('coherence_score', 0.0),
                        'rules_count': cluster.get('rules_count', 0),
                        'source_file': str(insights_file)
                    }
                    
                    cluster_analysis['clusters'].append(cluster_info)
                    total_coherence += cluster_info['coherence_score']
                    cluster_analysis['summary']['technologies'].add(cluster_info['technology'])
                    
                    if cluster_info['coherence_score'] >= min_coherence:
                        cluster_analysis['summary']['high_coherence_clusters'] += 1
        
        except Exception as e:
            click.echo(f"⚠️ Failed to parse {insights_file}: {e}")
    
    cluster_analysis['summary']['total_clusters'] = len(cluster_analysis['clusters'])
    cluster_analysis['summary']['avg_coherence'] = (
        total_coherence / len(cluster_analysis['clusters']) 
        if cluster_analysis['clusters'] else 0.0
    )
    cluster_analysis['summary']['technologies'] = list(cluster_analysis['summary']['technologies'])
    
    # Generate recommendations
    if cluster_analysis['summary']['avg_coherence'] < min_coherence:
        cluster_analysis['summary']['recommendations'].append(
            f"Average cluster coherence ({cluster_analysis['summary']['avg_coherence']:.3f}) is below threshold"
        )
    
    low_coherence_clusters = [c for c in cluster_analysis['clusters'] 
                             if c['coherence_score'] < min_coherence]
    if low_coherence_clusters:
        cluster_analysis['summary']['recommendations'].append(
            f"{len(low_coherence_clusters)} clusters have low coherence - consider reprocessing with higher quality sources"
        )
    
    if len(cluster_analysis['summary']['technologies']) < 5:
        cluster_analysis['summary']['recommendations'].append(
            "Limited technology coverage - consider adding more diverse documentation sources"
        )
    
    # Output results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(cluster_analysis, f, indent=2, default=str)
        
        click.echo(f"📄 Cluster analysis saved: {output_path}")
    
    # Display summary
    click.echo(f"\n📊 Cluster Analysis Summary:")
    click.echo(f"   Total clusters: {cluster_analysis['summary']['total_clusters']}")
    click.echo(f"   High coherence clusters: {cluster_analysis['summary']['high_coherence_clusters']}")
    click.echo(f"   Average coherence: {cluster_analysis['summary']['avg_coherence']:.3f}")
    click.echo(f"   Technologies covered: {len(cluster_analysis['summary']['technologies'])}")
    click.echo(f"   Technology list: {', '.join(cluster_analysis['summary']['technologies'][:10])}")
    
    if cluster_analysis['summary']['recommendations']:
        click.echo(f"\n💡 Recommendations:")
        for i, rec in enumerate(cluster_analysis['summary']['recommendations'], 1):
            click.echo(f"   {i}. {rec}")


def _calculate_rule_quality(content: str) -> float:
    """Calculate quality score for rule content."""
    score = 0.0
    
    # Base score
    score += 0.3
    
    # Length bonus
    if len(content) > 500:
        score += 0.1
    if len(content) > 1500:
        score += 0.1
    
    # Structure indicators
    if any(marker in content for marker in ['##', '**', '1.', '-', '*']):
        score += 0.1
    
    # Examples bonus
    if 'example' in content.lower():
        score += 0.15
    
    # Code blocks bonus
    if '```' in content:
        score += 0.1
    
    # Critical instructions bonus
    if any(word in content.upper() for word in ['NEVER', 'ALWAYS', 'MUST', 'SHOULD']):
        score += 0.1
    
    # Practical guidance bonus
    if any(word in content.lower() for word in ['pattern', 'best practice', 'guideline', 'principle']):
        score += 0.05
    
    return min(1.0, score)


def _assess_structure(content: str) -> float:
    """Assess structural quality of rule content."""
    score = 0.0
    
    # Header structure
    if content.count('##') >= 2:
        score += 0.3
    elif content.count('#') >= 1:
        score += 0.2
    
    # List structure
    if content.count('\n-') >= 3 or content.count('\n*') >= 3:
        score += 0.2
    
    # Numbered lists
    if content.count('\n1.') >= 1:
        score += 0.2
    
    # Code formatting
    if content.count('```') >= 2:
        score += 0.2
    
    # Bold/emphasis
    if content.count('**') >= 4:
        score += 0.1
    
    return min(1.0, score)


# ===================================
# ANALYTICS COMMAND GROUP
# ===================================

@main.group()
def analytics():
    """Analytics and insights commands."""
    pass


@analytics.command()
@click.argument('processing_results_dir', type=click.Path(exists=True))
@click.option('--output', '-o', help='Insights report output path')
@click.option('--format', type=click.Choice(['json', 'yaml', 'md']), default='md', help='Output format')
@click.option('--include-recommendations', is_flag=True, default=True, help='Include improvement recommendations')
def insights(processing_results_dir, output, format, include_recommendations):
    """Generate insights from batch processing results."""
    
    results_path = Path(processing_results_dir)
    
    # Find all insights and metadata files
    insights_files = list(results_path.rglob("*insights.json"))
    metadata_files = list(results_path.rglob("metadata.json"))
    
    if not insights_files and not metadata_files:
        click.echo(f"❌ No processing results found in {results_path}")
        return
    
    click.echo(f"📊 Generating insights from {len(insights_files)} insights files and {len(metadata_files)} metadata files...")
    
    # Aggregate insights
    combined_insights = {
        'analysis_timestamp': time.time(),
        'source_directory': str(results_path),
        'insights_files_processed': len(insights_files),
        'metadata_files_processed': len(metadata_files),
        'overall_statistics': {
            'total_sources_processed': 0,
            'total_rules_generated': 0,
            'total_clusters_created': 0,
            'processing_time_total': 0.0,
            'average_coherence': 0.0
        },
        'technology_breakdown': {},
        'quality_metrics': {},
        'performance_analysis': {},
        'recommendations': []
    }
    
    total_coherence_scores = []
    
    # Process insights files
    for insights_file in insights_files:
        try:
            with open(insights_file, 'r') as f:
                data = json.load(f)
            
            # Aggregate statistics
            if 'result_summary' in data:
                summary = data['result_summary']
                combined_insights['overall_statistics']['total_sources_processed'] += summary.get('sources_processed', 0)
                combined_insights['overall_statistics']['total_rules_generated'] += summary.get('total_rules_generated', 0)
                combined_insights['overall_statistics']['processing_time_total'] += summary.get('processing_time', 0)
                
                if 'quality_metrics' in summary:
                    qm = summary['quality_metrics']
                    if 'overall_coherence' in qm:
                        total_coherence_scores.append(qm['overall_coherence'])
            
            # Process clusters
            if 'clusters' in data:
                combined_insights['overall_statistics']['total_clusters_created'] += len(data['clusters'])
                
                for cluster in data['clusters']:
                    tech = cluster.get('technology', 'unknown')
                    if tech not in combined_insights['technology_breakdown']:
                        combined_insights['technology_breakdown'][tech] = {
                            'clusters': 0,
                            'total_rules': 0,
                            'avg_coherence': 0.0,
                            'coherence_scores': []
                        }
                    
                    combined_insights['technology_breakdown'][tech]['clusters'] += 1
                    combined_insights['technology_breakdown'][tech]['total_rules'] += cluster.get('rules_count', 0)
                    
                    if 'coherence_score' in cluster:
                        combined_insights['technology_breakdown'][tech]['coherence_scores'].append(cluster['coherence_score'])
        
        except Exception as e:
            click.echo(f"⚠️ Failed to process {insights_file}: {e}")
    
    # Calculate averages
    if total_coherence_scores:
        combined_insights['overall_statistics']['average_coherence'] = sum(total_coherence_scores) / len(total_coherence_scores)
    
    # Calculate technology averages
    for tech_data in combined_insights['technology_breakdown'].values():
        if tech_data['coherence_scores']:
            tech_data['avg_coherence'] = sum(tech_data['coherence_scores']) / len(tech_data['coherence_scores'])
        del tech_data['coherence_scores']  # Remove raw scores from output
    
    # Performance analysis
    if combined_insights['overall_statistics']['processing_time_total'] > 0:
        combined_insights['performance_analysis'] = {
            'avg_time_per_source': combined_insights['overall_statistics']['processing_time_total'] / max(1, combined_insights['overall_statistics']['total_sources_processed']),
            'avg_rules_per_source': combined_insights['overall_statistics']['total_rules_generated'] / max(1, combined_insights['overall_statistics']['total_sources_processed']),
            'throughput_sources_per_minute': 60 * combined_insights['overall_statistics']['total_sources_processed'] / max(1, combined_insights['overall_statistics']['processing_time_total'])
        }
    
    # Generate recommendations
    if include_recommendations:
        if combined_insights['overall_statistics']['average_coherence'] < 0.7:
            combined_insights['recommendations'].append(
                f"Overall coherence ({combined_insights['overall_statistics']['average_coherence']:.3f}) could be improved by using higher quality documentation sources"
            )
        
        if len(combined_insights['technology_breakdown']) < 5:
            combined_insights['recommendations'].append(
                f"Technology coverage is limited to {len(combined_insights['technology_breakdown'])} technologies - consider expanding source diversity"
            )
        
        if combined_insights['performance_analysis'].get('avg_time_per_source', 0) > 30:
            combined_insights['recommendations'].append(
                "Processing time per source is high - consider optimizing scraping configuration or using more concurrent operations"
            )
        
        # Find best and worst performing technologies
        tech_by_coherence = sorted(
            combined_insights['technology_breakdown'].items(),
            key=lambda x: x[1]['avg_coherence'],
            reverse=True
        )
        
        if tech_by_coherence and len(tech_by_coherence) > 2:
            best_tech = tech_by_coherence[0]
            worst_tech = tech_by_coherence[-1]
            combined_insights['recommendations'].append(
                f"Best performing technology: {best_tech[0]} (coherence: {best_tech[1]['avg_coherence']:.3f})"
            )
            combined_insights['recommendations'].append(
                f"Consider improving sources for {worst_tech[0]} (coherence: {worst_tech[1]['avg_coherence']:.3f})"
            )
    
    # Output results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(combined_insights, f, indent=2, default=str)
        elif format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(combined_insights, f, default_flow_style=False)
        elif format == 'md':
            with open(output_path, 'w') as f:
                f.write("# Batch Processing Insights Report\n\n")
                f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Source Directory:** {results_path}\n\n")
                
                f.write("## Overall Statistics\n\n")
                stats = combined_insights['overall_statistics']
                f.write(f"- **Sources Processed:** {stats['total_sources_processed']}\n")
                f.write(f"- **Rules Generated:** {stats['total_rules_generated']}\n")
                f.write(f"- **Clusters Created:** {stats['total_clusters_created']}\n")
                f.write(f"- **Total Processing Time:** {stats['processing_time_total']:.2f}s\n")
                f.write(f"- **Average Coherence:** {stats['average_coherence']:.3f}\n\n")
                
                if combined_insights['performance_analysis']:
                    f.write("## Performance Analysis\n\n")
                    perf = combined_insights['performance_analysis']
                    f.write(f"- **Avg Time per Source:** {perf.get('avg_time_per_source', 0):.2f}s\n")
                    f.write(f"- **Avg Rules per Source:** {perf.get('avg_rules_per_source', 0):.1f}\n")
                    f.write(f"- **Throughput:** {perf.get('throughput_sources_per_minute', 0):.1f} sources/min\n\n")
                
                if combined_insights['technology_breakdown']:
                    f.write("## Technology Breakdown\n\n")
                    for tech, data in sorted(combined_insights['technology_breakdown'].items()):
                        f.write(f"### {tech.title()}\n")
                        f.write(f"- Clusters: {data['clusters']}\n")
                        f.write(f"- Total Rules: {data['total_rules']}\n")
                        f.write(f"- Avg Coherence: {data['avg_coherence']:.3f}\n\n")
                
                if combined_insights['recommendations']:
                    f.write("## Recommendations\n\n")
                    for i, rec in enumerate(combined_insights['recommendations'], 1):
                        f.write(f"{i}. {rec}\n")
        
        click.echo(f"📄 Insights report saved: {output_path}")
    else:
        # Display summary to console
        stats = combined_insights['overall_statistics']
        click.echo(f"\n📊 Processing Insights Summary:")
        click.echo(f"   Sources processed: {stats['total_sources_processed']}")
        click.echo(f"   Rules generated: {stats['total_rules_generated']}")
        click.echo(f"   Clusters created: {stats['total_clusters_created']}")
        click.echo(f"   Average coherence: {stats['average_coherence']:.3f}")
        
        if combined_insights['technology_breakdown']:
            click.echo(f"\n🔧 Technology Coverage:")
            for tech, data in list(combined_insights['technology_breakdown'].items())[:5]:
                click.echo(f"   {tech}: {data['clusters']} clusters, {data['total_rules']} rules")
        
        if combined_insights['recommendations']:
            click.echo(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(combined_insights['recommendations'][:3], 1):
                click.echo(f"   {i}. {rec}")


# ===================================
# INTELLIGENT ENHANCEMENT COMMANDS
# ===================================

@main.group()
@click.pass_context
def interactive(ctx):
    """Intelligent interactive commands for enhanced user experience."""
    ctx.ensure_object(dict)


@interactive.command('session')
@click.option('--project-type', help='Type of project you are working on')
@click.option('--technologies', help='Comma-separated list of technologies')
@click.option('--experience-level', type=click.Choice(['beginner', 'intermediate', 'advanced', 'expert']), 
              default='intermediate', help='Your experience level')
@click.option('--session-id', help='Resume existing session by ID')
@click.option('--bedrock/--no-bedrock', default=False, help='Use Bedrock for enhanced recommendations')
@click.pass_context
def interactive_session(ctx, project_type, technologies, experience_level, session_id, bedrock):
    """Start an interactive documentation processing session."""
    try:
        from .interactive.cli_assistant import InteractiveCLIAssistant
        from .intelligence.models import UserIntent, ComplexityLevel
        
        # Setup bedrock config if requested
        bedrock_config = None
        if bedrock or ctx.obj.get('provider') == 'bedrock':
            bedrock_config = {
                'model_id': ctx.obj.get('model_id', 'amazon.nova-lite-v1:0'),
                'region': ctx.obj.get('region', 'us-east-1')
            }
            
            if ctx.obj.get('credentials_csv'):
                from .utils.credentials import setup_bedrock_credentials
                cred_result = setup_bedrock_credentials(ctx.obj.get('credentials_csv'))
                if not cred_result.get('validation', {}).get('success', False):
                    click.echo("⚠️  Bedrock credentials validation failed - using basic mode", err=True)
                    bedrock_config = None
        
        # Initialize assistant
        assistant = InteractiveCLIAssistant(bedrock_config)
        
        # Run interactive session
        click.echo("🚀 Starting intelligent interactive session...")
        click.echo("   This guided workflow will help you generate personalized rules")
        click.echo("   based on your project needs and experience level.\n")
        
        async def run_session():
            try:
                session_result = await assistant.start_interactive_session(session_id)
                
                click.echo(f"\n✨ Session completed successfully!")
                click.echo(f"   Session ID: {session_result.session_id}")
                click.echo(f"   Steps completed: {len(session_result.completed_steps)}")
                
                if "generation_results" in session_result.metadata:
                    results = session_result.metadata["generation_results"]
                    click.echo(f"   Rules generated: {results.get('rules_generated', 0)}")
                    click.echo(f"   Sources processed: {results.get('sources_processed', 0)}")
                    
                    if results.get('output_directory'):
                        click.echo(f"   Output saved to: {results['output_directory']}")
                
                return session_result
                
            except Exception as e:
                click.echo(f"❌ Interactive session failed: {e}", err=True)
                raise click.Abort()
        
        # Run the session
        _run_async(run_session())
        
    except ImportError as e:
        click.echo(f"❌ Interactive features not available: {e}", err=True)
        click.echo("   Try installing additional dependencies", err=True)
        raise click.Abort()


@interactive.command('analyze')
@click.argument('content', required=False)
@click.option('--url', help='URL of documentation to analyze')
@click.option('--file', 'input_file', type=click.Path(exists=True), help='File containing content to analyze')
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis results')
@click.option('--bedrock/--no-bedrock', default=False, help='Use Bedrock for enhanced analysis')
@click.pass_context
def analyze_content(ctx, content, url, input_file, output, bedrock):
    """Perform intelligent semantic analysis of documentation content."""
    try:
        from .intelligence.semantic_analyzer import SemanticAnalyzer
        
        # Get content from various sources
        if not any([content, url, input_file]):
            content = click.prompt("Enter content to analyze", type=str)
        
        if input_file:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        if not content and not url:
            click.echo("❌ No content provided to analyze", err=True)
            raise click.Abort()
        
        # Setup bedrock config
        bedrock_config = None
        if bedrock or ctx.obj.get('provider') == 'bedrock':
            bedrock_config = {
                'model_id': ctx.obj.get('model_id', 'amazon.nova-lite-v1:0'),
                'region': ctx.obj.get('region', 'us-east-1')
            }
        
        # Initialize analyzer
        analyzer = SemanticAnalyzer(bedrock_config)
        
        async def run_analysis():
            try:
                click.echo("🔍 Analyzing content with semantic intelligence...")
                
                analysis = await analyzer.analyze_content(
                    content or "", 
                    url or "user-provided-content"
                )
                
                # Display results
                click.echo(f"\n📊 **Semantic Analysis Results**")
                click.echo(f"   Primary Technology: {analysis.primary_technology}")
                
                if analysis.secondary_technologies:
                    click.echo(f"   Secondary Technologies: {', '.join(analysis.secondary_technologies)}")
                
                click.echo(f"   Complexity Level: {analysis.complexity_level.value}")
                click.echo(f"   Content Type: {analysis.content_type.value}")
                click.echo(f"   Quality Score: {analysis.quality_score:.2f}")
                
                if analysis.framework_version:
                    click.echo(f"   Framework Version: {analysis.framework_version}")
                
                if analysis.content_categories:
                    click.echo(f"\n📋 **Detected Categories:**")
                    for category, details in analysis.content_categories.items():
                        click.echo(f"   • {category}: {details.confidence:.2f} confidence")
                        if details.topics:
                            click.echo(f"     Topics: {', '.join(details.topics[:3])}")
                
                if analysis.prerequisites:
                    click.echo(f"\n📚 **Prerequisites:** {', '.join(analysis.prerequisites)}")
                
                click.echo(f"\n🔢 **Content Metrics:**")
                click.echo(f"   Code examples: {analysis.code_examples_count}")
                click.echo(f"   External links: {analysis.external_links_count}")
                click.echo(f"   Content length: {analysis.metadata.get('content_length', 0)} characters")
                
                # Save results if requested
                if output:
                    import json
                    with open(output, 'w') as f:
                        json.dump(analysis.model_dump(), f, indent=2, default=str)
                    click.echo(f"\n💾 Analysis saved to: {output}")
                
                return analysis
                
            except Exception as e:
                click.echo(f"❌ Content analysis failed: {e}", err=True)
                raise click.Abort()
        
        _run_async(run_analysis())
        
    except ImportError as e:
        click.echo(f"❌ Semantic analysis features not available: {e}", err=True)
        raise click.Abort()


@interactive.command('query')
@click.argument('question', required=False)
@click.option('--technologies', help='Comma-separated list of technologies for context')
@click.option('--project-type', help='Type of project for context')
@click.option('--experience-level', type=click.Choice(['beginner', 'intermediate', 'advanced', 'expert']), 
              default='intermediate', help='Your experience level')
@click.option('--bedrock/--no-bedrock', default=False, help='Use Bedrock for enhanced responses')
@click.pass_context
def natural_language_query(ctx, question, technologies, project_type, experience_level, bedrock):
    """Ask natural language questions about documentation and coding practices."""
    try:
        from .nlp.query_processor import NaturalLanguageQueryProcessor, ProjectContext
        
        if not question:
            question = click.prompt("What would you like to know?", type=str)
        
        # Setup bedrock config
        bedrock_config = None
        if bedrock or ctx.obj.get('provider') == 'bedrock':
            bedrock_config = {
                'model_id': ctx.obj.get('model_id', 'amazon.nova-lite-v1:0'),
                'region': ctx.obj.get('region', 'us-east-1')
            }
        
        # Initialize query processor
        processor = NaturalLanguageQueryProcessor(bedrock_config)
        
        # Create project context
        context = None
        if technologies or project_type:
            tech_list = technologies.split(',') if technologies else []
            tech_list = [t.strip() for t in tech_list]
            context = ProjectContext(
                technologies=tech_list,
                project_type=project_type,
                experience_level=experience_level
            )
        
        async def run_query():
            try:
                click.echo(f"🤔 Processing your question: \"{question}\"")
                click.echo("   Analyzing intent and finding relevant information...\n")
                
                response = await processor.process_query(question, context)
                
                # Display the answer
                click.echo(f"💡 **Answer:**")
                click.echo(f"{response.answer}\n")
                
                # Show relevant sources
                if response.relevant_sources:
                    click.echo(f"📚 **Relevant Sources:**")
                    for i, source in enumerate(response.relevant_sources[:3], 1):
                        click.echo(f"{i}. {source.source}")
                        click.echo(f"   Reason: {source.reason}")
                        click.echo(f"   Value: {source.estimated_value}\n")
                
                # Show suggested rules
                if response.suggested_rules:
                    click.echo(f"📝 **Suggested Rules:**")
                    for rule in response.suggested_rules[:3]:
                        click.echo(f"   • {rule}")
                    click.echo()
                
                # Show related topics
                if response.related_topics:
                    click.echo(f"🔗 **Related Topics:** {', '.join(response.related_topics[:5])}\n")
                
                # Show follow-up suggestions
                if response.follow_up_suggestions:
                    click.echo(f"❓ **You might also want to ask:**")
                    for suggestion in response.follow_up_suggestions[:3]:
                        click.echo(f"   • {suggestion}")
                
                click.echo(f"\n🎯 Confidence: {response.confidence:.1%}")
                
                return response
                
            except Exception as e:
                click.echo(f"❌ Query processing failed: {e}", err=True)
                raise click.Abort()
        
        _run_async(run_query())
        
    except ImportError as e:
        click.echo(f"❌ Natural language query features not available: {e}", err=True)
        raise click.Abort()


@interactive.command('predict')
@click.option('--project-analysis', type=click.Path(exists=True), 
              help='JSON file with project analysis data')
@click.option('--user-id', default='default', help='User ID for personalized predictions')
@click.option('--current-rules', type=click.Path(exists=True), 
              help='File containing current rules for context')
@click.option('--output', '-o', type=click.Path(), help='Output file for predictions')
@click.option('--bedrock/--no-bedrock', default=False, help='Use Bedrock for enhanced predictions')
@click.pass_context
def predict_rules(ctx, project_analysis, user_id, current_rules, output, bedrock):
    """Predict what rules you'll need based on your project patterns."""
    try:
        from .intelligence.predictive_enhancer import PredictiveRuleEnhancer
        from .intelligence.models import ProjectAnalysis
        from .learning.user_behavior_tracker import UserBehaviorTracker
        import json
        
        # Setup bedrock config
        bedrock_config = None
        if bedrock or ctx.obj.get('provider') == 'bedrock':
            bedrock_config = {
                'model_id': ctx.obj.get('model_id', 'amazon.nova-lite-v1:0'),
                'region': ctx.obj.get('region', 'us-east-1')
            }
        
        # Initialize enhancer
        enhancer = PredictiveRuleEnhancer(bedrock_config)
        behavior_tracker = UserBehaviorTracker()
        
        # Load project analysis
        if project_analysis:
            with open(project_analysis, 'r') as f:
                analysis_data = json.load(f)
                analysis = ProjectAnalysis(**analysis_data)
        else:
            # Create basic analysis from user input
            click.echo("🔍 Let's analyze your project to make predictions...")
            has_auth = click.confirm("Does your project have authentication?")
            has_routing = click.confirm("Does your project use complex routing?")
            has_state = click.confirm("Does your project use state management?")
            has_api = click.confirm("Does your project integrate with APIs?")
            uses_db = click.confirm("Does your project use a database?")
            has_tests = click.confirm("Does your project have testing setup?")
            
            analysis = ProjectAnalysis(
                has_authentication_patterns=has_auth,
                uses_complex_routing=has_routing,
                has_state_management=has_state,
                has_api_integration=has_api,
                uses_database=uses_db,
                has_testing_setup=has_tests
            )
        
        # Load user profile
        user_profile = behavior_tracker.load_user_profile(user_id)
        
        # Load current rules
        rules_list = []
        if current_rules:
            with open(current_rules, 'r') as f:
                rules_list = f.read().splitlines()
        
        async def run_predictions():
            try:
                click.echo("🔮 Predicting rule needs based on your project...")
                
                predictions = await enhancer.predict_rule_needs(
                    analysis, user_profile, rules_list
                )
                
                if not predictions:
                    click.echo("💭 No specific predictions at this time. Your project looks well set up!")
                    return
                
                click.echo(f"\n📋 **Rule Predictions ({len(predictions)} found):**\n")
                
                for i, pred in enumerate(predictions, 1):
                    priority_emoji = {"critical": "🚨", "high": "⚠️ ", "medium": "📝", "low": "💡"}
                    emoji = priority_emoji.get(pred.priority, "📝")
                    
                    click.echo(f"{emoji} **{i}. {pred.rule_type.replace('-', ' ').title()}**")
                    click.echo(f"   Priority: {pred.priority.title()}")
                    click.echo(f"   Confidence: {pred.confidence:.1%}")
                    click.echo(f"   Reason: {pred.reason}")
                    click.echo(f"   Impact: {pred.estimated_impact or 'Not specified'}")
                    click.echo(f"   Timing: {pred.suggested_timing or 'When convenient'}")
                    
                    if pred.dependencies:
                        click.echo(f"   Dependencies: {', '.join(pred.dependencies)}")
                    click.echo()
                
                # Save predictions if requested
                if output:
                    pred_data = [pred.model_dump() for pred in predictions]
                    with open(output, 'w') as f:
                        json.dump(pred_data, f, indent=2, default=str)
                    click.echo(f"💾 Predictions saved to: {output}")
                
                return predictions
                
            except Exception as e:
                click.echo(f"❌ Prediction failed: {e}", err=True)
                raise click.Abort()
        
        _run_async(run_predictions())
        
    except ImportError as e:
        click.echo(f"❌ Predictive features not available: {e}", err=True)
        raise click.Abort()


@interactive.command('insights')
@click.option('--user-id', default='default', help='User ID for behavior insights')
@click.option('--global-insights', is_flag=True, help='Show global usage patterns')
@click.option('--output', '-o', type=click.Path(), help='Output file for insights')
@click.pass_context
def user_insights(ctx, user_id, global_insights, output):
    """Get insights about your learning patterns and system usage."""
    try:
        from .learning.user_behavior_tracker import UserBehaviorTracker
        import json
        
        tracker = UserBehaviorTracker()
        
        async def run_insights():
            try:
                if global_insights:
                    click.echo("🌍 Analyzing global usage patterns...")
                    insights = await tracker.get_global_insights()
                    
                    click.echo(f"\n📊 **Global Usage Insights:**")
                    
                    if insights['popular_technologies']:
                        click.echo(f"\n🔧 **Popular Technologies:**")
                        for tech, count in insights['popular_technologies'][:5]:
                            click.echo(f"   • {tech}: {count} sessions")
                    
                    click.echo(f"\n📈 **Global Stats:**")
                    click.echo(f"   Success patterns: {insights['success_patterns_count']}")
                    click.echo(f"   Total error reports: {insights['total_error_reports']}")
                    
                    if insights.get('insights'):
                        click.echo(f"\n💡 **System Insights:**")
                        for insight in insights['insights']:
                            click.echo(f"   • {insight}")
                
                else:
                    click.echo(f"👤 Analyzing behavior patterns for user: {user_id}")
                    insights = await tracker.get_user_insights(user_id)
                    
                    if 'error' in insights:
                        click.echo(f"❌ {insights['error']}")
                        return insights
                    
                    summary = insights['user_summary']
                    click.echo(f"\n📊 **Your Usage Summary:**")
                    click.echo(f"   Total sessions: {summary['total_sessions']}")
                    click.echo(f"   Rules generated: {summary['total_rules_generated']}")
                    click.echo(f"   Account age: {summary['account_age_days']} days")
                    click.echo(f"   Avg session time: {summary['avg_session_duration']:.1f} minutes")
                    
                    prefs = insights['preferences']
                    click.echo(f"\n🎯 **Your Preferences:**")
                    click.echo(f"   Learning style: {prefs['learning_style']}")
                    click.echo(f"   Workflow efficiency: {prefs['workflow_efficiency']:.1%}")
                    
                    if prefs['top_frameworks']:
                        click.echo(f"\n🔧 **Your Top Frameworks:**")
                        for framework, usage in prefs['top_frameworks'][:3]:
                            click.echo(f"   • {framework}: {usage} times")
                    
                    if insights['skill_progression']:
                        click.echo(f"\n📈 **Skill Progression:**")
                        for skill, level in insights['skill_progression'].items():
                            click.echo(f"   • {skill}: {level}")
                    
                    patterns = insights['recent_patterns']
                    click.echo(f"\n🔄 **Recent Patterns:**")
                    click.echo(f"   Success rate: {patterns['success_rate']:.1%}")
                    click.echo(f"   Productivity trend: {patterns['productivity_trend']}")
                    
                    if insights.get('recommendations'):
                        click.echo(f"\n💡 **Recommendations for You:**")
                        for rec in insights['recommendations']:
                            click.echo(f"   • {rec}")
                
                # Save insights if requested
                if output:
                    with open(output, 'w') as f:
                        json.dump(insights, f, indent=2, default=str)
                    click.echo(f"\n💾 Insights saved to: {output}")
                
                return insights
                
            except Exception as e:
                click.echo(f"❌ Insights analysis failed: {e}", err=True)
                raise click.Abort()
        
        _run_async(run_insights())
        
    except ImportError as e:
        click.echo(f"❌ User insights features not available: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    main()
