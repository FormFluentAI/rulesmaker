"""
Semantic content understanding: pattern recognition, best practices, anti-patterns,
and context-aware custom rule generation.

Implements Phase 2 (1.2) "Intelligent Content Understanding" as a lightweight,
heuristics-based analyzer that requires no heavy ML dependencies.
"""

from __future__ import annotations

import re
import hashlib
from typing import Dict, List, Optional, Tuple

from .models import (
    CodePattern,
    CodePatterns,
    BestPractice,
    BestPractices,
    AntiPattern,
    AntiPatterns,
    ContentAnalysis,
    CustomRules,
)
from ..models import Rule, RuleType


_FENCE_RE = re.compile(r"```(\w+)?\n([\s\S]*?)```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


class SemanticAnalyzer:
    """Advanced content analysis with semantic understanding (heuristic).

    Methods:
      - extract_code_patterns
      - identify_best_practices
      - detect_anti_patterns
      - generate_custom_rules
    """

    def extract_code_patterns(self, content: str) -> CodePatterns:
        """Identify recurring code/documentation patterns from content.

        Heuristics:
          - Parse fenced code blocks and inline code.
          - Detect structural patterns (classes, functions, async, config).
          - Count occurrences and keep representative examples.
        """
        languages, blocks = self._extract_code_blocks(content)

        patterns: Dict[str, CodePattern] = {}

        def _bump(name: str, desc: str, lang: Optional[str], category: str, tag: str, example: Optional[str], score: float = 0.6):
            p = patterns.get(name)
            if not p:
                p = CodePattern(
                    name=name,
                    description=desc,
                    occurrences=0,
                    confidence=score,
                    language=lang,
                    category=category,
                    tags=[tag],
                    examples=[],
                )
                patterns[name] = p
            p.occurrences += 1
            if example and len(p.examples) < 3:
                p.examples.append(example.strip())
            if tag not in p.tags:
                p.tags.append(tag)

        # Analyze blocks
        for lang, code in blocks:
            # Structure
            for m in re.finditer(r"\bclass\s+([A-Za-z_][\w]*)", code):
                _bump("Class Definition", "Object-oriented class usage detected.", lang, "structure", "class", m.group(0))
            for m in re.finditer(r"\bdef\s+([A-Za-z_][\w]*)\s*\(", code):
                _bump("Function Definition", "Function or method definitions present.", lang, "structure", "function", m.group(0))
            for m in re.finditer(r"\basync\b.*?\bdef\b|\bawait\b", code):
                _bump("Async Pattern", "Asynchronous programming constructs detected.", lang, "async", "async", m.group(0), 0.7)
            # Config-like blocks
            if lang in {"json", "yaml", "yml", "toml", "ini"} or re.search(r"\b({|: )", code):
                _bump("Configuration Block", "Configuration structure or snippet found.", lang, "config", "config", code.strip().splitlines()[0] if code.strip().splitlines() else None)
            # Testing
            if re.search(r"\bpytest\b|\bunittest\b|\bdescribe\(|\bit\(|\bexpect\(", code):
                _bump("Testing Pattern", "Test code presence indicates testing practices.", lang, "testing", "tests", None)
            # Logging
            if re.search(r"\blogging\b|\blogn?\(|\bconsole\.log\(", code):
                _bump("Logging Usage", "Logging statements present.", lang, "observability", "logging", None)

        # Inline code often hints at APIs/CLI usage
        inline_hits = _INLINE_CODE_RE.findall(content)
        for snip in inline_hits[:50]:
            if re.search(r"\bGET\b|\bPOST\b|curl\s+-", snip):
                _bump("API Usage", "HTTP/API usage referenced.", None, "api", "http", snip, 0.55)

        categories = sorted(set(p.category or "misc" for p in patterns.values()))
        langs = sorted(set(languages))
        return CodePatterns(patterns=list(patterns.values()), languages=langs, categories=categories)

    def identify_best_practices(self, patterns: CodePatterns) -> BestPractices:
        """Derive best practices from discovered patterns."""
        items: List[BestPractice] = []

        def add_bp(name: str, desc: str, rationale: str, conf: float, tags: List[str]):
            items.append(BestPractice(name=name, description=desc, rationale=rationale, confidence=conf, tags=tags))

        has_async = any(p.name == "Async Pattern" for p in patterns.patterns)
        has_tests = any(p.category == "testing" for p in patterns.patterns)
        has_logging = any(p.name == "Logging Usage" for p in patterns.patterns)
        has_config = any(p.category == "config" for p in patterns.patterns)

        if has_async:
            add_bp(
                "Prefer async/await for I/O",
                "Use async functions and await for concurrent I/O-bound operations.",
                "Improves throughput and responsiveness in network/disk heavy workflows.",
                0.75,
                ["async", "performance"],
            )
        if not has_tests:
            add_bp(
                "Establish testing baseline",
                "Add unit tests for core modules and critical paths.",
                "Prevents regressions and documents expected behaviors.",
                0.7,
                ["testing", "quality"],
            )
        if has_logging:
            add_bp(
                "Use structured logging",
                "Adopt consistent, structured logs rather than ad-hoc prints.",
                "Improves observability and production diagnostics.",
                0.7,
                ["logging", "observability"],
            )
        if has_config:
            add_bp(
                "Externalize configuration",
                "Keep environment- and secret-specific values out of source code.",
                "Supports 12-factor practices and secure deployments.",
                0.8,
                ["config", "security"],
            )

        # Generic guidance when language hints detected
        if patterns.languages:
            add_bp(
                "Document language-specific conventions",
                "Ensure style and idioms are documented for detected languages.",
                "Consistency accelerates onboarding and reduces review churn.",
                0.6,
                ["docs", "consistency"],
            )

        return BestPractices(items=items)

    def detect_anti_patterns(self, content: str) -> AntiPatterns:
        """Detect common anti-patterns using regex-based heuristics."""
        items: List[AntiPattern] = []

        def add_ap(name: str, desc: str, impact: str, remediation: str, severity: str, conf: float, example: Optional[str], tags: List[str]):
            items.append(
                AntiPattern(
                    name=name,
                    description=desc,
                    impact=impact,
                    remediation=remediation,
                    severity=severity,  # type: ignore[arg-type]
                    confidence=conf,
                    examples=[example] if example else [],
                    tags=tags,
                )
            )

        # Common code smells / insecure patterns (language-agnostic heuristics)
        # 1) Python: bare except
        m = re.search(r"except\s*:\s*\n\s*\w+", content)
        if m:
            add_ap(
                "Bare except",
                "Catching all exceptions without handling specific error types.",
                "Masks real failures and complicates debugging.",
                "Catch specific exceptions and handle appropriately.",
                "medium",
                0.7,
                m.group(0),
                ["error-handling", "python"],
            )

        # 2) Python: mutable default argument
        m = re.search(r"def\s+\w+\s*\(.*=\s*\[|\{\}", content)
        if m:
            add_ap(
                "Mutable default argument",
                "Function default uses a mutable value (list/dict).",
                "Unexpected state sharing across calls.",
                "Default to None and create a new object inside the function.",
                "high",
                0.8,
                m.group(0),
                ["python", "correctness"],
            )

        # 3) Wildcard import
        m = re.search(r"from\s+\w+\s+import\s+\*", content)
        if m:
            add_ap(
                "Wildcard import",
                "Importing * pollutes namespace and obscures dependencies.",
                "Hinders readability and can cause collisions.",
                "Import explicit symbols or the module itself.",
                "low",
                0.6,
                m.group(0),
                ["style", "python"],
            )

        # 4) Use of eval/exec
        m = re.search(r"\beval\s*\(|\bexec\s*\(", content)
        if m:
            add_ap(
                "Dynamic eval/exec",
                "Use of eval/exec identified.",
                "Security risks and hard-to-audit behavior.",
                "Avoid dynamic evaluation; use safe parsing or dispatch tables.",
                "high",
                0.8,
                m.group(0),
                ["security"],
            )

        # 5) Hard-coded credentials (heuristic)
        m = re.search(r"AKIA[0-9A-Z]{16}", content)
        if m:
            add_ap(
                "Hard-coded AWS key",
                "Potential AWS Access Key ID found.",
                "Secret leakage risk.",
                "Rotate credentials and use environment managers or secret vaults.",
                "high",
                0.85,
                m.group(0),
                ["security", "secrets"],
            )

        # 6) Debug prints in production-like docs
        m = re.search(r"\bconsole\.log\(|\bprint\(\s*['\"]DEBUG", content)
        if m:
            add_ap(
                "Debug prints",
                "Debug logging via print/console.log detected.",
                "Noisy logs and potential PII exposure.",
                "Use logging frameworks with levels and scrub sensitive data.",
                "low",
                0.55,
                m.group(0),
                ["observability"],
            )

        return AntiPatterns(items=items)

    def generate_custom_rules(self, analysis: ContentAnalysis) -> CustomRules:
        """Generate context-aware rules from the combined analysis.

        Strategy:
          - Convert best practices to BEST_PRACTICE rules.
          - Convert anti-patterns to ERROR_HANDLING rules.
          - Promote salient code patterns as CODE_PATTERN rules.
        """
        rules: List[Rule] = []

        def mk_id(prefix: str, name: str) -> str:
            base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            # Keep IDs stable and short; add hash tail to avoid collisions
            tail = hashlib.sha1(name.encode()).hexdigest()[:6]
            return f"{prefix}-{base}-{tail}"

        # Best practices → BEST_PRACTICE
        for bp in analysis.best_practices.items:
            rules.append(
                Rule(
                    id=mk_id("bp", bp.name),
                    title=bp.name,
                    description=bp.description,
                    content=f"Rationale: {bp.rationale}",
                    type=RuleType.BEST_PRACTICE,
                    category=(analysis.frameworks[0] if analysis.frameworks else "general"),
                    priority=3,
                    confidence_score=min(1.0, max(0.0, bp.confidence)),
                    tags=[*bp.tags, "best-practice"],
                    examples=[],
                )
            )

        # Anti-patterns → ERROR_HANDLING
        for ap in analysis.anti_patterns.items:
            desc = ap.description + (f" Impact: {ap.impact}" if ap.impact else "")
            content = "Remediation: " + ap.remediation if ap.remediation else ""
            rules.append(
                Rule(
                    id=mk_id("ap", ap.name),
                    title=f"Avoid: {ap.name}",
                    description=desc,
                    content=content,
                    type=RuleType.ERROR_HANDLING,
                    category="anti-pattern",
                    priority=4 if ap.severity in ("high",) else 2,
                    confidence_score=min(1.0, max(0.0, ap.confidence)),
                    tags=[*ap.tags, "anti-pattern"],
                    examples=ap.examples[:3],
                    anti_patterns=[ap.name],
                )
            )

        # Code patterns → CODE_PATTERN (top few by occurrences)
        top = sorted(analysis.patterns.patterns, key=lambda p: (p.occurrences, p.confidence), reverse=True)[:5]
        for cp in top:
            rules.append(
                Rule(
                    id=mk_id("cp", cp.name),
                    title=f"Pattern: {cp.name}",
                    description=cp.description,
                    content=("\n".join(cp.examples) if cp.examples else ""),
                    type=RuleType.CODE_PATTERN,
                    category=cp.category or "pattern",
                    priority=2,
                    confidence_score=min(1.0, max(0.0, cp.confidence)),
                    tags=[*cp.tags, *(analysis.languages or [])],
                    examples=cp.examples[:2],
                )
            )

        coverage = {
            "best_practices": (len(analysis.best_practices.items) > 0) * 1.0,
            "anti_patterns": (len(analysis.anti_patterns.items) > 0) * 1.0,
            "code_patterns": (len(analysis.patterns.patterns) > 0) * 1.0,
        }

        return CustomRules(rules=rules, coverage=coverage, notes="Generated from heuristic semantic analysis.")

    # ---- Helpers ----
    def _extract_code_blocks(self, content: str) -> Tuple[List[str], List[Tuple[Optional[str], str]]]:
        """Return (languages, blocks[(lang, code), ...]) parsed from markdown-like content."""
        langs: List[str] = []
        blocks: List[Tuple[Optional[str], str]] = []

        for m in _FENCE_RE.finditer(content):
            lang = (m.group(1) or "").strip() or None
            code = m.group(2)
            blocks.append((lang, code))
            if lang:
                langs.append(lang)

        return langs, blocks

    # ---- One-shot helper ----
    def analyze_content(self, content: str) -> ContentAnalysis:
        """Run a full semantic pass and return ContentAnalysis in one step."""
        patterns = self.extract_code_patterns(content)
        best = self.identify_best_practices(patterns)
        anti = self.detect_anti_patterns(content)

        languages = patterns.languages
        frameworks = self._infer_frameworks(content)
        key_topics = self._extract_topics(content)

        summary = self._summarize_content(content)

        return ContentAnalysis(
            content_summary=summary,
            languages=languages,
            frameworks=frameworks,
            key_topics=key_topics,
            patterns=patterns,
            best_practices=best,
            anti_patterns=anti,
            metadata={"length": len(content)},
        )

    def _infer_frameworks(self, content: str) -> List[str]:
        text = content.lower()
        candidates = {
            "react": ["react", "jsx", "tsx"],
            "nextjs": ["next.js", "nextjs"],
            "vue": ["vue", "nuxt"],
            "angular": ["angular"],
            "svelte": ["svelte"],
            "django": ["django"],
            "flask": ["flask"],
            "fastapi": ["fastapi"],
            "pytorch": ["pytorch"],
            "tensorflow": ["tensorflow"],
            "nodejs": ["node.js", "nodejs", "npm", "express"],
            "spring": ["spring", "springboot", "spring boot"],
            ".net": [".net", "asp.net"],
            "rails": ["rails", "rubyonrails"],
            "laravel": ["laravel"],
            "kubernetes": ["kubernetes", "k8s"],
        }
        found = []
        for fw, keys in candidates.items():
            if any(k in text for k in keys):
                found.append(fw)
        return found

    def _extract_topics(self, content: str) -> List[str]:
        text = content.lower()
        topics = [
            ("testing", ["test", "pytest", "unittest", "describe(", "it("]),
            ("security", ["security", "auth", "oauth", "xss", "csrf", "secrets"]),
            ("performance", ["performance", "optimiz", "latency", "throughput", "cache"]),
            ("api", ["api", "http", "rest", "graphql"]),
            ("configuration", ["config", "configuration", "env", "environment variable"]),
            ("logging", ["logging", "logger", "console.log", "print("]),
            ("error-handling", ["error", "exception", "retry", "backoff"]),
            ("async", ["async", "await", "concurrent", "thread", "event loop"]),
            ("observability", ["metrics", "tracing", "trace", "prometheus", "grafana"]),
        ]
        found = []
        for name, keys in topics:
            if any(k in text for k in keys):
                found.append(name)
        return found

    def _summarize_content(self, content: str) -> str:
        trimmed = content.strip().splitlines()
        head = " ".join(line.strip() for line in trimmed[:5])
        if len(head) > 300:
            head = head[:297] + "..."
        return head
