#!/usr/bin/env python3
"""
Simple benchmark for AsyncDocumentationScraper concurrency.

Note: This performs real HTTP requests. Use with care.
"""

import asyncio
import time
from rules_maker.scrapers.async_documentation_scraper import AsyncDocumentationScraper
from rules_maker.models import ScrapingConfig


async def main():
    urls = ["https://example.com/"] * 20
    scraper = AsyncDocumentationScraper(ScrapingConfig(rate_limit=0.0, timeout=10))
    start = time.time()
    async with scraper:
        results = await scraper.scrape_multiple(urls)
    dur = time.time() - start
    ok = sum(1 for r in results if r.status.value == "completed")
    print(f"Completed {ok}/{len(results)} requests in {dur:.2f}s (~{len(results)/dur:.2f} rps)")


if __name__ == "__main__":
    asyncio.run(main())

