"""
CLI interface for Rules Maker.
"""

import click
import json
import yaml
import asyncio
import os
from pathlib import Path
from typing import Optional
import logging


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
@click.option('--llm-provider', type=click.Choice(['openai', 'anthropic', 'huggingface', 'bedrock', 'local']), 
              help='LLM provider for adaptive scraping')
@click.option('--llm-api-key', help='API key for LLM provider')
@click.option('--llm-model', help='LLM model name')
@click.option('--credentials-csv', type=click.Path(exists=True), help='Credentials CSV for provider bedrock')
@click.option('--region', help='Region for provider bedrock')
@click.option('--interactive/--no-interactive', '-i/ ', default=False, help='Interactive wizard to choose options')
@click.pass_context
def scrape(ctx, url, output, output_format, split, output_dir, max_pages, deep, async_scrape, adaptive, 
           llm_provider, llm_api_key, llm_model, credentials_csv, region, interactive, use_ml):
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


if __name__ == '__main__':
    main()
