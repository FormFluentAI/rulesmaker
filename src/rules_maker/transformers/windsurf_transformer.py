"""
Windsurf rules transformer.
"""

from typing import List, Dict, Any
from datetime import datetime
import re

from .rule_transformer import RuleTransformer
from ..models import ScrapingResult


class WindsurfRuleTransformer(RuleTransformer):
    """Transformer for Windsurf rules format."""
    
    def transform(self, results: List[ScrapingResult]) -> str:
        """Transform scraping results into Windsurf rules format."""
        if not results:
            return self._create_empty_windsurf_rules()
        
        # Analyze content similar to Cursor transformer
        tech_stack = self._identify_technology_stack(results)
        domain_info = self._analyze_domain(results)
        
        base = self._generate_windsurf_rules(results, tech_stack, domain_info)
        
        # Optional category-focused prologue mirroring Cursor parity
        cat = getattr(self.config, 'category_hint', None)
        if cat:
            preset = getattr(self.config, 'taxonomy_preset', None)
            canonical = self._canonicalize_category(cat, preset)
            prologue: List[str] = []
            prologue.append(f"# Category: {canonical.replace('-', ' ').title()}")
            prologue.append("")
            prologue.extend(self._category_specific_guidelines(canonical))
            prologue.append("")
            topics = self._collect_focus_topics(results)
            if topics:
                prologue.append("## Focus Topics")
                prologue.append("")
                prologue.extend([f"- {t}" for t in topics])
                prologue.append("")
            snippets = self._extract_top_code_snippets(results, limit=2)
            if snippets:
                prologue.append("## Example Snippets")
                prologue.append("")
                for s in snippets:
                    prologue.append(s)
                    prologue.append("")
            return "\n".join(prologue) + "\n" + base
        
        return base
    
    def _identify_technology_stack(self, results: List[ScrapingResult]) -> Dict[str, Any]:
        """Identify tech stack using regex word boundaries and URL weighting."""
        import re
        tech_indicators = {
            'python': ['python', 'django', 'flask', 'fastapi', 'pytest'],
            'javascript': ['javascript', 'node', 'npm', 'yarn', 'webpack'],
            'typescript': ['typescript', 'ts', 'interface'],
            'react': ['react', 'jsx', 'hooks', 'useeffect', 'usestate'],
            'nextjs': ['next.js', 'nextjs', 'getserversideprops', 'getstaticprops'],
            'vue': ['vue', 'pinia', 'composition api'],
            'angular': ['angular', '@angular', 'rxjs'],
        }
        content = ' '.join([r.content for r in results if r.content]).lower()
        urls = ' '.join([str(r.url) for r in results]).lower()
        scores = {}
        for tech, indicators in tech_indicators.items():
            c = sum(len(re.findall(rf"\b{re.escape(ind)}\b", content)) for ind in indicators)
            u = sum(len(re.findall(rf"\b{re.escape(ind)}\b", urls)) for ind in indicators)
            s = c * 1.0 + u * 2.0
            if s > 0:
                scores[tech] = s
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_domain(self, results: List[ScrapingResult]) -> Dict[str, Any]:
        """Analyze the domain and purpose of the documentation."""
        all_content = ' '.join([result.content for result in results if result.content]).lower()
        
        return {
            'is_framework_docs': any(x in all_content for x in ['installation', 'getting started', 'quickstart']),
            'is_api_docs': any(x in all_content for x in ['endpoint', 'api reference', 'authentication']),
            'is_tutorial': any(x in all_content for x in ['tutorial', 'step by step', 'example']),
            'is_library_docs': any(x in all_content for x in ['import', 'function', 'method', 'class'])
        }
    
    def _generate_windsurf_rules(self, results: List[ScrapingResult], tech_stack: Dict, domain_info: Dict) -> str:
        """Generate Windsurf-specific rules format."""
        primary_tech = list(tech_stack.keys())[0] if tech_stack else 'general'
        
        output = []
        
        # Windsurf-style header
        output.append("# Windsurf Workflow Rules")
        output.append("")
        output.append(f"## Expert Role")
        output.append(self._generate_windsurf_role(primary_tech, tech_stack))
        output.append("")
        
        # Workflow sections
        output.append("## Development Workflow")
        output.append("")
        output.extend(self._generate_workflow_steps(primary_tech, domain_info))
        output.append("")
        
        # Code standards
        output.append("## Code Standards")
        output.append("")
        output.extend(self._generate_windsurf_standards(primary_tech))
        output.append("")
        
        # Project structure
        output.append("## Project Structure")
        output.append("")
        output.extend(self._generate_project_structure(primary_tech))
        output.append("")
        
        # Quality gates
        output.append("## Quality Gates")
        output.append("")
        output.extend(self._generate_quality_gates(primary_tech))
        output.append("")
        
        # Documentation insights
        concepts = self._extract_concepts(results)
        if concepts:
            output.append("## Documentation Context")
            output.append("")
            output.extend(self._format_windsurf_insights(concepts, results))
            output.append("")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output.append("---")
        output.append(f"*Windsurf Rules generated from {len(results)} page(s) on {timestamp}*")
        
        return '\n'.join(output)
    
    def _generate_windsurf_role(self, primary_tech: str, tech_stack: Dict) -> str:
        """Generate Windsurf-style role definition."""
        if primary_tech == 'python':
            return "You are a Python development expert focusing on clean architecture, testing, and maintainable code."
        elif primary_tech in ['react', 'nextjs']:
            return "You are a React/Next.js expert specializing in modern web development and user experience."
        elif primary_tech == 'javascript':
            return "You are a JavaScript expert with deep knowledge of modern ES6+ features and best practices."
        else:
            return f"You are a {primary_tech} development expert focusing on best practices and maintainable solutions."
    
    def _generate_workflow_steps(self, primary_tech: str, domain_info: Dict) -> List[str]:
        """Generate development workflow steps."""
        steps = [
            "1. **Analysis Phase**",
            "   - Understand requirements thoroughly",
            "   - Identify potential challenges and edge cases",
            "   - Plan the implementation approach",
            "",
            "2. **Implementation Phase**",
            "   - Write clean, well-documented code",
            "   - Follow established patterns and conventions",
            "   - Implement proper error handling",
            "",
            "3. **Testing Phase**",
            "   - Write comprehensive tests",
            "   - Test edge cases and error conditions",
            "   - Validate performance requirements",
            "",
            "4. **Review Phase**",
            "   - Code review for quality and standards",
            "   - Documentation review",
            "   - Security review"
        ]
        
        if domain_info['is_api_docs']:
            steps.insert(4, "   - API contract validation")
        
        return steps
    
    def _generate_windsurf_standards(self, primary_tech: str) -> List[str]:
        """Generate Windsurf-style code standards."""
        if primary_tech == 'python':
            return [
                "- **Style**: Follow PEP 8 and use black formatter",
                "- **Types**: Use type hints for all public functions",
                "- **Documentation**: Use docstrings following Google/NumPy style",
                "- **Testing**: Achieve >90% test coverage with pytest",
                "- **Dependencies**: Pin versions in requirements.txt"
            ]
        elif primary_tech in ['javascript', 'typescript', 'react']:
            return [
                "- **Style**: Use ESLint + Prettier configuration",
                "- **Types**: Implement strict TypeScript types",
                "- **Components**: Use functional components with hooks",
                "- **Testing**: Jest + React Testing Library",
                "- **Performance**: Bundle size monitoring and optimization"
            ]
        else:
            return [
                "- **Consistency**: Follow established coding standards",
                "- **Quality**: Maintain high code quality standards",
                "- **Documentation**: Document all public interfaces",
                "- **Testing**: Comprehensive test coverage",
                "- **Performance**: Regular performance monitoring"
            ]
    
    def _generate_project_structure(self, primary_tech: str) -> List[str]:
        """Generate project structure guidelines."""
        if primary_tech == 'python':
            return [
                "```",
                "project/",
                "├── src/",
                "│   └── package_name/",
                "├── tests/",
                "├── docs/",
                "├── requirements.txt",
                "├── pyproject.toml",
                "└── README.md",
                "```"
            ]
        elif primary_tech in ['react', 'nextjs']:
            return [
                "```",
                "project/",
                "├── src/",
                "│   ├── components/",
                "│   ├── pages/ (or app/)",
                "│   ├── hooks/",
                "│   └── utils/",
                "├── public/",
                "├── tests/",
                "├── package.json",
                "└── README.md",
                "```"
            ]
        else:
            return [
                "- Organize code into logical modules",
                "- Separate concerns appropriately",
                "- Maintain clear project structure",
                "- Document architecture decisions"
            ]
    
    def _generate_quality_gates(self, primary_tech: str) -> List[str]:
        """Generate quality gate requirements."""
        return [
            "✅ **Code Quality**",
            "- Linting passes without errors",
            "- Type checking passes (if applicable)",
            "- No code duplication above threshold",
            "",
            "✅ **Testing**",
            "- All tests pass",
            "- Coverage meets minimum requirements",
            "- Integration tests included",
            "",
            "✅ **Security**",
            "- No known vulnerabilities in dependencies",
            "- Input validation implemented",
            "- Authentication/authorization proper",
            "",
            "✅ **Performance**",
            "- Meets performance benchmarks",
            "- Bundle size within limits (web projects)",
            "- Memory usage optimized"
        ]
    
    def _extract_concepts(self, results: List[ScrapingResult]) -> List[str]:
        """Extract key concepts from documentation."""
        concepts = set()
        for result in results:
            if result.sections:
                for section in result.sections:
                    section_dict = (
                        section.model_dump() if hasattr(section, 'model_dump') else section.dict() if hasattr(section, 'dict') else section
                    )
                    title = section_dict.get('title', '')
                    if title:
                        concept_words = re.findall(r'\b[A-Z][a-z]+\b', title)
                        concepts.update(concept_words[:2])
        return list(concepts)[:8]
    
    def _format_windsurf_insights(self, concepts: List[str], results: List[ScrapingResult]) -> List[str]:
        """Format documentation insights for Windsurf."""
        insights = []
        
        if concepts:
            insights.append(f"**Key Concepts**: {', '.join(concepts)}")
        
        urls = [str(result.url) for result in results[:2]]
        if urls:
            insights.append(f"**Source Documentation**: {', '.join(urls)}")
        
        insights.extend([
            "**Implementation Notes**:",
            "- Follow patterns established in the documentation",
            "- Refer to official examples for best practices",
            "- Stay updated with latest framework versions"
        ])
        
        return insights

    # --- Category parity helpers (mirrors Cursor) ---
    def _canonicalize_category(self, category: str, preset: Any) -> str:
        if not preset:
            return category
        cat = category.lower()
        if str(preset).lower() == 'nextjs':
            mapping = {
                'routing': ['routing', 'route', 'linking', 'navigation', 'navigating', 'app', 'pages', 'link', 'router', 'parallel-routes', 'intercepting-routes', 'nested-routes'],
                'data-fetching': ['data', 'fetch', 'fetching', 'rsc', 'server-components', 'getserversideprops', 'getstaticprops', 'loader', 'actions-data'],
                'caching': ['cache', 'caching', 'revalidation', 'revalidate', 'revalidatepath', 'revalidatetag'],
                'server-actions': ['server-actions', 'actions', 'use server', 'form-actions'],
                'errors': ['error', 'errors', 'loading', 'not-found', 'boundary', 'boundaries'],
                'assets-images': ['image', 'images', 'assets', 'static', 'public', 'next/image', 'font', 'next/font'],
                'styling': ['style', 'styles', 'styling', 'css', 'tailwind', 'sass', 'css-modules'],
                'metadata': ['metadata', 'seo', 'og', 'open-graph', 'sitemap', 'robots'],
                'middleware-edge': ['middleware', 'edge', 'edge-runtime', 'matcher'],
                'architecture': ['architecture', 'patterns', 'best-practices', 'structure', 'app-structure'],
                'i18n': ['i18n', 'internationalization', 'localization', 'locale', 'languages'],
            }
            for canonical, synonyms in mapping.items():
                if any(word in cat for word in synonyms):
                    return canonical
        return category

    def _category_specific_guidelines(self, category: str) -> List[str]:
        cat = category.lower()
        bullets: List[str] = []
        def add(title: str, items: List[str]):
            if items:
                bullets.append(f"## {title}")
                bullets.append("")
                bullets.extend([f"- {it}" for it in items])
                bullets.append("")
        if any(k in cat for k in ["getting-started", "installation", "setup"]):
            add("Setup Essentials", [
                "Initialize with `create-next-app` and App Router",
                "Enable strict TypeScript and ESLint core-web-vitals",
                "Structure `app/` with root `layout.tsx`",
            ])
        if "routing" in cat or "linking" in cat or "navigation" in cat or cat == "app":
            add("Routing & Navigation", [
                "Define routes by folders in `app/`",
                "Use `Link` for client navigation, not `<a>`",
                "Provide `default.tsx` for parallel routes fallbacks",
            ])
        if "data" in cat or "fetch" in cat:
            add("Data Fetching", [
                "Fetch in Server Components; prefer static + revalidate",
                "Control cache via `{ next: { revalidate, tags } }`",
                "Use `revalidateTag`/`revalidatePath` post-mutations",
            ])
        if "cach" in cat or "revalid" in cat:
            add("Caching & Revalidation", [
                "Default static; opt into dynamic only when required",
                "Tag critical queries; invalidate narrowly",
                "Avoid broad cache flushes",
            ])
        if "server" in cat and "action" in cat:
            add("Server Actions", [
                "Mark functions `'use server'`; validate inputs",
                "Keep secrets server-side; never expose tokens",
                "Revalidate relevant paths/tags after writes",
            ])
        if "error" in cat or "loading" in cat or "not-found" in cat:
            add("Error/Loading Boundaries", [
                "Provide segment-level `error.tsx`, `loading.tsx`, `not-found.tsx`",
                "Log server errors once; avoid leaking stack traces",
                "Reset state in client error boundaries",
            ])
        if "image" in cat or "assets" in cat:
            add("Assets & Images", [
                "Use `next/image` with proper `sizes`/`fill`",
                "Serve static files from `public/` with `/` paths",
                "Load fonts via `next/font` to prevent CLS",
            ])
        if "css" in cat or "styling" in cat or "tailwind" in cat:
            add("Styling", [
                "Prefer CSS Modules or Tailwind; minimize runtime CSS-in-JS",
                "Scope styles; use layout-level CSS for shells",
                "Avoid hydration flashes with consistent theming",
            ])
        if "metadata" in cat or "seo" in cat:
            add("Metadata & SEO", [
                "Use `export const metadata` or `generateMetadata`",
                "Set canonicals, OpenGraph, and i18n `alternates`",
                "Generate `app/robots.ts` and `app/sitemap.ts` when needed",
            ])
        if "middleware" in cat or "edge" in cat:
            add("Middleware & Edge", [
                "Constrain `config.matcher` to targeted paths",
                "Avoid Node-only APIs in edge runtime",
                "Keep middleware fast and stateless",
            ])
        if "architecture" in cat or "patterns" in cat:
            add("Architecture", [
                "Default to Server Components; isolate client islands",
                "Co-locate by feature within route segments",
                "Extract shared logic into server utilities",
            ])
        if not bullets:
            add("Category Focus", ["Apply framework best practices specific to this topic."])
        return bullets

    def _collect_focus_topics(self, results: List[ScrapingResult]) -> List[str]:
        titles: List[str] = []
        seen = set()
        for r in results:
            for s in (r.sections or []):
                sd = s.model_dump() if hasattr(s, 'model_dump') else s.dict() if hasattr(s, 'dict') else s
                t = str(sd.get('title', '')).strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    titles.append(t)
                if len(titles) >= 12:
                    break
            if len(titles) >= 12:
                break
        return titles

    def _extract_top_code_snippets(self, results: List[ScrapingResult], limit: int = 2) -> List[str]:
        pattern = re.compile(r"```[ \t]*([a-zA-Z0-9+-]*)\n([\s\S]*?)\n```", re.MULTILINE)
        preferred = ["tsx", "ts", "jsx", "js", "bash", "sh"]
        collected = []
        out: List[str] = []
        for res in results:
            content = res.content or ""
            for m in pattern.finditer(content):
                lang = (m.group(1) or "").lower()
                code = (m.group(2) or "").strip()
                if not code:
                    continue
                lines = code.splitlines()
                if len(lines) > 20:
                    code = "\n".join(lines[:20]) + "\n// ..."
                block = f"```{lang}\n{code}\n```" if lang else f"```\n{code}\n```"
                score = (preferred.index(lang) if lang in preferred else len(preferred) + 1)
                collected.append((score, block))
        collected.sort(key=lambda x: x[0])
        for _, block in collected:
            if block not in out:
                out.append(block)
            if len(out) >= limit:
                break
        return out
    
    def _create_empty_windsurf_rules(self) -> str:
        """Create empty Windsurf rules template."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""# Windsurf Workflow Rules

## Expert Role
You are a software development expert focusing on quality, maintainability, and best practices.

## Development Workflow

1. **Analysis Phase**
   - Understand requirements thoroughly
   - Identify potential challenges
   - Plan implementation approach

2. **Implementation Phase**
   - Write clean, documented code
   - Follow established patterns
   - Implement error handling

3. **Testing Phase**
   - Write comprehensive tests
   - Test edge cases
   - Validate performance

4. **Review Phase**
   - Code quality review
   - Documentation review
   - Security review

## Code Standards

- Follow language-specific best practices
- Maintain consistent code style
- Document public interfaces
- Achieve good test coverage

## Quality Gates

✅ **Code Quality**: Linting and type checking pass
✅ **Testing**: All tests pass with good coverage
✅ **Security**: No known vulnerabilities
✅ **Performance**: Meets requirements

---
*Windsurf Rules generated on {timestamp}*"""
