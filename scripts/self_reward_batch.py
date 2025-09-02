"""
Self-reward batch runner for Rules Maker.

Given a list of documentation URLs, this script:
- Scrapes pages concurrently (async or adaptive LLM-enhanced)
- Generates both Cursor and Windsurf rule outputs
- Computes simple reward scores (heuristics + optional LLM judge)
- Writes outputs under rules/ and a JSONL report under reports/

Usage (from repo root):
  PYTHONPATH=src .venv/bin/python scripts/self_reward_batch.py \
    --urls-file urls.txt --out-base rules --adaptive \
    --provider bedrock --credentials-csv docs/plans/bedrock-long-term-api-key.csv \
    --concurrency 8

Heuristic reward aims to be fast and provider-agnostic; LLM judging is optional
and enabled via --llm-judge and provider flags/budget.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slug_url(url: str) -> str:
    s = url.replace("https://", "").replace("http://", "")
    s = s.replace("/", "_")
    if not s:
        s = "index"
    return s


def _heuristic_reward(content: str) -> Dict[str, Any]:
    """Fast, cheap reward proxies for rule quality."""
    length = len(content or "")
    # Basic structure checks
    hits = {
        "has_principles": ("## Key Principles" in content) or ("## Development Workflow" in content),
        "has_critical": ("Critical Instructions" in content) or ("Quality Gates" in content),
        "has_code": ("```" in content) or ("`" in content),
        "has_role": ("Expert Role" in content) or (content.strip().startswith("You are")),
    }
    # Score: weighted sum with sane length window
    score = 0.0
    # Length window: 1k-15k chars ideal; shape a trapezoid
    if length <= 300:
        len_score = 0.0
    elif length <= 1000:
        len_score = (length - 300) / 700 * 0.4
    elif length <= 15000:
        len_score = 0.4
    elif length <= 30000:
        len_score = 0.4 * (1 - (length - 15000) / 15000)
    else:
        len_score = 0.0

    score += len_score
    score += 0.2 if hits["has_principles"] else 0.0
    score += 0.2 if hits["has_critical"] else 0.0
    score += 0.1 if hits["has_code"] else 0.0
    score += 0.1 if hits["has_role"] else 0.0

    return {
        "length": length,
        "features": hits,
        "heuristic_score": round(score, 4),
    }


async def _llm_judge_score(prompt: str, provider_cfg: Dict[str, Any]) -> Optional[float]:
    """Optional LLM judge: returns a float 0-1 or None on failure.

    Uses rules_maker.extractors.LLMContentExtractor for a lightweight call.
    """
    try:
        from rules_maker.extractors.llm_extractor import LLMContentExtractor, LLMConfig, LLMProvider

        prov = (provider_cfg.get("provider") or "").lower()
        if not prov:
            return None
        if prov == "bedrock":
            cfg = LLMConfig(
                provider=LLMProvider.BEDROCK,
                model_name=provider_cfg.get("model_id") or os.environ.get("BEDROCK_MODEL_ID") or "amazon.nova-lite-v1:0",
                region=provider_cfg.get("region") or os.environ.get("AWS_REGION") or "us-east-1",
                timeout=int(provider_cfg.get("timeout", 20)),
                max_concurrency=int(provider_cfg.get("concurrency", 2)),
            )
        elif prov == "openai":
            cfg = LLMConfig(provider=LLMProvider.OPENAI, api_key=os.environ.get("OPENAI_API_KEY"), model_name=provider_cfg.get("model_id") or "gpt-4o-mini")
        elif prov == "anthropic":
            cfg = LLMConfig(provider=LLMProvider.ANTHROPIC, api_key=os.environ.get("ANTHROPIC_API_KEY"), model_name=provider_cfg.get("model_id") or "claude-3-haiku-20240307")
        else:
            return None

        ex = LLMContentExtractor(llm_config=cfg)
        try:
            sys = (
                "You are a strict reviewer of AI assistant rules. "
                "Return ONLY a float between 0 and 1 (4 decimals), where higher is better."
            )
            user = (
                "Rate the following rules for clarity, structure, actionable guidance, and correctness. "
                "Answer with a single float between 0 and 1.\n\n" + prompt[:6000]
            )
            res = await ex._make_llm_request(user, sys)  # internal but fine for utility
            # Try to extract a float
            txt = json.dumps(res)
            import re as _re

            m = _re.search(r"\b(0\.\d{1,4}|1\.0{1,4}|1)\b", txt)
            if not m:
                return None
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        finally:
            await ex.close()
    except Exception:
        return None


async def run(args: argparse.Namespace) -> None:
    from rules_maker.scrapers.async_documentation_scraper import AsyncDocumentationScraper
    from rules_maker.scrapers.adaptive_documentation_scraper import AdaptiveDocumentationScraper
    from rules_maker.models import ScrapingConfig
    from rules_maker.transformers import CursorRuleTransformer, WindsurfRuleTransformer

    urls = [u.strip() for u in Path(args.urls_file).read_text(encoding="utf-8").splitlines() if u.strip() and not u.strip().startswith("#")]
    if not urls:
        raise SystemExit("No URLs to process.")

    out_base = Path(args.out_base)
    out_cursor = out_base / "cursor"
    out_windsurf = out_base / "windsurf"
    report_dir = Path(args.report_dir)
    _mkdir(out_cursor)
    _mkdir(out_windsurf)
    _mkdir(report_dir)

    # Scraper selection
    scraping_cfg = ScrapingConfig(max_pages=1, rate_limit=max(0.0, args.rate_limit))
    if args.adaptive:
        # LLM is optional for adaptive; pass provider settings via config
        bedrock_cfg = None
        if args.provider == "bedrock":
            bedrock_cfg = {
                "model_id": args.model_id or os.environ.get("BEDROCK_MODEL_ID") or "amazon.nova-lite-v1:0",
                "region": args.region or os.environ.get("AWS_REGION") or "us-east-1",
                "timeout": args.timeout,
                "concurrency": args.concurrency,
            }
        app_cfg: Dict[str, Any] = {"bedrock": bedrock_cfg} if bedrock_cfg else {}
        scraper = AdaptiveDocumentationScraper(
            config=scraping_cfg,
            use_ml=True,
            use_llm=bool(bedrock_cfg) or args.llm_judge,
            llm_config=None,
            app_config=app_cfg or None,
        )
    else:
        scraper = AsyncDocumentationScraper(scraping_cfg)

    cursor_t = CursorRuleTransformer()
    windsurf_t = WindsurfRuleTransformer()

    semaphore = asyncio.Semaphore(args.concurrency)
    results: List[Dict[str, Any]] = []

    async def process_url(u: str) -> None:
        async with semaphore:
            try:
                res = await scraper.scrape_url(u)
                # Transform to both formats
                cursor_text = cursor_t.transform([res])
                windsurf_text = windsurf_t.transform([res])

                slug = _slug_url(u)
                (out_cursor / f"{slug}.mdc").write_text(cursor_text, encoding="utf-8")
                (out_windsurf / f"{slug}.md").write_text(windsurf_text, encoding="utf-8")

                # Rewards
                heur_c = _heuristic_reward(cursor_text)
                heur_w = _heuristic_reward(windsurf_text)
                judge_score: Optional[float] = None
                if args.llm_judge:
                    provider_cfg = {
                        "provider": args.provider,
                        "model_id": args.model_id,
                        "region": args.region,
                        "timeout": args.timeout,
                        "concurrency": args.concurrency,
                    }
                    # Judge on Windsurf rules (workflow-focused) by default
                    judge_score = await _llm_judge_score(windsurf_text, provider_cfg)

                item = {
                    "url": u,
                    "files": {
                        "cursor": str((out_cursor / f"{slug}.mdc").as_posix()),
                        "windsurf": str((out_windsurf / f"{slug}.md").as_posix()),
                    },
                    "rewards": {
                        "cursor": heur_c,
                        "windsurf": heur_w,
                        "llm_judge": judge_score,
                    },
                }
                results.append(item)
            except Exception as e:
                results.append({"url": u, "error": str(e)})

    async with scraper:
        await asyncio.gather(*[process_url(u) for u in urls])
        if hasattr(scraper, "close"):
            await scraper.close()

    # Persist report
    report_path = report_dir / "self_reward_report.jsonl"
    with report_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    succ = [r for r in results if not r.get("error")]
    best = None
    def _total_score(r: Dict[str, Any]) -> float:
        rw = r.get("rewards", {})
        base = (rw.get("windsurf", {}).get("heuristic_score") or 0.0) + (rw.get("cursor", {}).get("heuristic_score") or 0.0)
        if rw.get("llm_judge") is not None:
            base += float(rw["llm_judge"])  # add judge
        return float(base)

    if succ:
        best = max(succ, key=_total_score)
    summary = {
        "total": len(results),
        "succeeded": len(succ),
        "failed": len(results) - len(succ),
        "report": str(report_path.as_posix()),
        "best_example": best,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, **{k: v for k, v in summary.items() if k != "best_example"}}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", required=True, help="Text file with one URL per line")
    ap.add_argument("--out-base", default="rules", help="Base output dir (default: rules)")
    ap.add_argument("--report-dir", default="reports", help="Directory to write JSONL report")
    ap.add_argument("--adaptive", action="store_true", help="Use adaptive scraper (ML/LLM-enhanced)")
    ap.add_argument("--provider", choices=["bedrock", "openai", "anthropic", "local", "none"], default="none", help="LLM provider for judge/adaptive")
    ap.add_argument("--model-id", default=None, help="Provider model ID (e.g., amazon.nova-lite-v1:0)")
    ap.add_argument("--region", default=None, help="Cloud region (for bedrock)")
    ap.add_argument("--credentials-csv", default=None, help="Optional CSV for Bedrock credentials setup")
    ap.add_argument("--concurrency", type=int, default=6, help="Concurrent tasks")
    ap.add_argument("--timeout", type=int, default=20, help="Per-request timeout (LLM judge)")
    ap.add_argument("--rate-limit", type=float, default=0.0, help="Seconds between requests per worker")
    ap.add_argument("--llm-judge", action="store_true", help="Enable LLM judging for reward score")
    args = ap.parse_args()

    # Optional Bedrock credentials setup from CSV (if provided and provider is bedrock)
    if args.provider == "bedrock" and args.credentials-csv:
        try:
            from rules_maker.utils.credentials import setup_bedrock_credentials
            setup_bedrock_credentials(args.credentials_csv)
        except Exception as e:  # non-fatal
            print(f"Warning: failed to setup Bedrock credentials: {e}")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

