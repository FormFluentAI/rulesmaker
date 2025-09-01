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
        
        return self._generate_windsurf_rules(results, tech_stack, domain_info)
    
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
                    section_dict = section.dict() if hasattr(section, 'dict') else section
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
