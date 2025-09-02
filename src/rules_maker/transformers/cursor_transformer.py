"""
Cursor rules transformer.
"""

from typing import List, Dict, Any, Set
from datetime import datetime
import re

from .rule_transformer import RuleTransformer
from ..models import ScrapingResult
from ..templates import TemplateEngine


class CursorRuleTransformer(RuleTransformer):
    """Transformer for Cursor rules format."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_engine = TemplateEngine()
    
    def transform(self, results: List[ScrapingResult]) -> str:
        """Transform scraping results into Cursor rules format."""
        if not results:
            return self._create_empty_rules()
        
        # Analyze the content to determine the technology stack and domain
        tech_stack = self._identify_technology_stack(results)
        domain_info = self._analyze_domain(results)
        
        # Generate comprehensive rules
        return self._generate_comprehensive_rules(results, tech_stack, domain_info)
    
    def _identify_technology_stack(self, results: List[ScrapingResult]) -> Dict[str, Any]:
        """Identify the technology stack from content and URLs with weighted, regex-based scoring."""
        import re
        tech_indicators = {
            'python':      [r'\bpython\b', r'\bdjango\b', r'\bflask\b', r'\bfastapi\b', r'\bpandas\b', r'\bnumpy\b', r'\bpytest\b', r'\bpip\b'],
            'javascript':  [r'\bjavascript\b', r'\bnode\b', r'\bnpm\b', r'\byarn\b', r'\bwebpack\b', r'\bbabel\b'],
            'typescript':  [r'\btypescript\b', r'\bts\b', r'\binterface\b', r'\bgeneric\b'],
            'react':       [r'\breact\b', r'\bjsx\b', r'\bhook\b', r'\buse(state|effect)\b', r'\bprops\b', r'\bcomponent\b'],
            'nextjs':      [r'\bnext\.?js\b', r'\bnextjs\b', r'get(Server|Static)SideProps', r'\bapp router\b'],
            'vue':         [r'\bvue(js)?\b', r'composition api', r'options api', r'\bpinia\b'],
            'angular':     [r'\bangular\b', r'@angular', r'\brxjs\b', r'\bcomponent\b', r'\bservice\b'],
            'svelte':      [r'\bsvelte\b'],
            'astro':       [r'\bastro\b'],
            'nodejs':      [r'\bnode(\.js)?\b', r'\bexpress\b', r'\brequire\b', r'\bmodule\b'],
            'api':         [r'\bapi\b', r'\bendpoint\b', r'\bgraphql\b', r'\bopenapi\b', r'\bswagger\b'],
            'database':    [r'\b(database|sql|nosql)\b', r'\bpostgres(ql)?\b', r'\bmysql\b', r'\bmongodb\b', r'\bsqlite\b'],
            'web':         [r'\bhtml\b', r'\bcss\b', r'\bdom\b', r'\bhttp\b'],
            'mobile':      [r'\bios\b', r'\bandroid\b', r'\breact native\b', r'\bflutter\b', r'\bexpo\b'],
            'cloud':       [r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bdocker\b', r'\bkubernetes\b', r'\bterraform\b'],
        }
        
        identified: Dict[str, float] = {}
        content = ' '.join([r.content for r in results if r.content]).lower()
        urls = ' '.join([str(r.url) for r in results]).lower()
        
        for tech, patterns in tech_indicators.items():
            c_hits = sum(len(re.findall(p, content, flags=re.IGNORECASE)) for p in patterns)
            u_hits = sum(len(re.findall(p, urls, flags=re.IGNORECASE)) for p in patterns)
            score = c_hits * 1.0 + u_hits * 2.0  # weight URL hints higher
            if score > 0:
                identified[tech] = score
        
        return dict(sorted(identified.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_domain(self, results: List[ScrapingResult]) -> Dict[str, Any]:
        """Analyze the domain and purpose of the documentation."""
        domain_info = {
            'is_framework_docs': False,
            'is_api_docs': False,
            'is_tutorial': False,
            'is_library_docs': False,
            'primary_domain': 'general'
        }
        
        all_content = ' '.join([result.content for result in results if result.content]).lower()
        urls = [str(result.url) for result in results]
        
        # Check for framework documentation
        framework_indicators = ['installation', 'getting started', 'quickstart', 'configuration', 'setup']
        if any(indicator in all_content for indicator in framework_indicators):
            domain_info['is_framework_docs'] = True
        
        # Check for API documentation
        api_indicators = ['endpoint', 'api reference', 'authentication', 'parameters', 'response', 'request']
        if any(indicator in all_content for indicator in api_indicators):
            domain_info['is_api_docs'] = True
        
        # Check for tutorial content
        tutorial_indicators = ['tutorial', 'step by step', 'example', 'walkthrough', 'guide']
        if any(indicator in all_content for indicator in tutorial_indicators):
            domain_info['is_tutorial'] = True
        
        # Check for library documentation
        library_indicators = ['import', 'function', 'method', 'class', 'module', 'package']
        if any(indicator in all_content for indicator in library_indicators):
            domain_info['is_library_docs'] = True
        
        return domain_info
    
    def _generate_comprehensive_rules(self, results: List[ScrapingResult], tech_stack: Dict, domain_info: Dict) -> str:
        """Generate comprehensive Cursor rules based on analyzed content."""
        # Determine primary technology
        primary_tech = list(tech_stack.keys())[0] if tech_stack else 'general'
        
        # Extract key concepts and patterns
        concepts = self._extract_key_concepts(results)
        patterns = self._extract_code_patterns(results)
        best_practices = self._extract_best_practices(results)
        
        # Generate the rules
        output = []
        
        # Header with role definition
        output.append(self._generate_role_definition(primary_tech, tech_stack))
        output.append("")
        
        # Key principles section
        output.append("## Key Principles")
        output.append("")
        output.extend(self._generate_key_principles(domain_info, concepts))
        output.append("")
        
        # Code style and structure
        output.append("## Code Style and Structure")
        output.append("")
        output.extend(self._generate_code_style_rules(primary_tech, patterns))
        output.append("")
        
        # Best practices
        if best_practices:
            output.append("## Best Practices")
            output.append("")
            output.extend(best_practices)
            output.append("")
        
        # Technology-specific guidelines
        if tech_stack:
            output.append("## Technology-Specific Guidelines")
            output.append("")
            output.extend(self._generate_tech_specific_rules(tech_stack, concepts))
            output.append("")
        
        # Error handling and validation
        output.append("## Error Handling and Validation")
        output.append("")
        output.extend(self._generate_error_handling_rules(primary_tech))
        output.append("")
        
        # Performance considerations
        output.append("## Performance Considerations")
        output.append("")
        output.extend(self._generate_performance_rules(primary_tech))
        output.append("")
        
        # Critical instructions
        output.append("## 🚨 Critical Instructions")
        output.append("")
        output.extend(self._generate_critical_instructions(primary_tech, domain_info))
        output.append("")
        
        # Documentation and examples
        if concepts:
            output.append("## Documentation Insights")
            output.append("")
            output.extend(self._format_documentation_insights(concepts, results))
            output.append("")
        
        # Footer
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output.append(f"---")
        output.append(f"*Generated by Rules Maker from {len(results)} documentation page(s) on {timestamp}*")
        
        return '\n'.join(output)
    
    def _generate_role_definition(self, primary_tech: str, tech_stack: Dict) -> str:
        """Generate the role definition based on technology stack."""
        tech_list = list(tech_stack.keys())[:5]  # Top 5 technologies
        
        if 'python' in tech_list:
            return f"You are an expert in Python and {', '.join([t for t in tech_list[1:3] if t != 'python'])} development."
        elif 'nextjs' in tech_list:
            return f"You are an expert in Next.js, React, TypeScript, and full-stack development."
        elif 'react' in tech_list:
            return f"You are an expert in React, TypeScript, and modern web development."
        elif 'javascript' in tech_list or 'typescript' in tech_list:
            return f"You are an expert in JavaScript/TypeScript and {', '.join([t for t in tech_list[1:3] if t not in ['javascript', 'typescript']])} development."
        elif 'vue' in tech_list:
            return f"You are an expert in Vue.js, TypeScript, and modern web development."
        elif 'angular' in tech_list:
            return f"You are an expert in Angular, TypeScript, and enterprise web development."
        else:
            return f"You are an expert in {primary_tech} and modern software development."
    
    def _generate_key_principles(self, domain_info: Dict, concepts: List[str]) -> List[str]:
        """Generate key principles based on domain analysis."""
        principles = []
        
        if domain_info['is_framework_docs']:
            principles.extend([
                "- Follow the framework's established patterns and conventions",
                "- Use framework-specific features and abstractions appropriately",
                "- Maintain consistency with official documentation examples"
            ])
        
        if domain_info['is_api_docs']:
            principles.extend([
                "- Implement proper HTTP status codes and error responses",
                "- Follow RESTful principles for API design",
                "- Include comprehensive request/response validation"
            ])
        
        if domain_info['is_tutorial']:
            principles.extend([
                "- Write clear, step-by-step implementations",
                "- Include practical examples and use cases",
                "- Provide helpful comments and explanations"
            ])
        
        # Generic principles
        principles.extend([
            "- Write clean, readable, and maintainable code",
            "- Use descriptive names for variables, functions, and classes",
            "- Implement proper error handling and logging",
            "- Follow security best practices"
        ])
        
        return principles
    
    def _generate_code_style_rules(self, primary_tech: str, patterns: List[str]) -> List[str]:
        """Generate code style rules based on technology."""
        rules = []
        
        if primary_tech in ['python']:
            rules.extend([
                "- Follow PEP 8 style guidelines",
                "- Use type hints for function parameters and return values",
                "- Prefer list comprehensions and generator expressions when appropriate",
                "- Use context managers for resource management (with statements)"
            ])
        
        elif primary_tech in ['javascript', 'typescript', 'react', 'nextjs']:
            rules.extend([
                "- Use modern ES6+ syntax and features",
                "- Prefer const and let over var",
                "- Use arrow functions for simple expressions",
                "- Implement proper TypeScript types and interfaces"
            ])
        
        elif primary_tech in ['vue']:
            rules.extend([
                "- Use Composition API for complex components",
                "- Follow Vue 3 best practices and patterns",
                "- Use TypeScript with Vue for better type safety",
                "- Implement proper component structure and organization"
            ])
        
        # Add pattern-specific rules
        if patterns:
            rules.append(f"- Follow these documented patterns: {', '.join(patterns[:3])}")
        
        return rules
    
    def _generate_tech_specific_rules(self, tech_stack: Dict, concepts: List[str]) -> List[str]:
        """Generate technology-specific rules."""
        rules = []
        
        for tech in list(tech_stack.keys())[:3]:  # Top 3 technologies
            if tech == 'react':
                rules.extend([
                    "### React Guidelines",
                    "- Use functional components with hooks",
                    "- Implement proper state management (useState, useReducer)",
                    "- Use React.memo for performance optimization",
                    "- Follow component composition patterns",
                    "- Use proper prop types and TypeScript interfaces",
                    ""
                ])
            
            elif tech == 'nextjs':
                rules.extend([
                    "### Next.js Guidelines", 
                    "- Use App Router for new projects (not Pages Router)",
                    "- Implement proper data fetching patterns (Server Components)",
                    "- Optimize for Core Web Vitals",
                    "- Use server and client components appropriately",
                    "- Follow Next.js file-based routing conventions",
                    ""
                ])
            
            elif tech == 'python':
                rules.extend([
                    "### Python Guidelines",
                    "- Use virtual environments for dependency management",
                    "- Follow the Zen of Python principles",
                    "- Implement proper exception handling",
                    "- Use dataclasses or Pydantic for data structures",
                    "- Follow PEP standards and use tools like black, flake8",
                    ""
                ])
            
            elif tech == 'vue':
                rules.extend([
                    "### Vue.js Guidelines",
                    "- Use Composition API for complex logic",
                    "- Implement proper component lifecycle management",
                    "- Use Pinia for state management",
                    "- Follow Vue 3 best practices and patterns",
                    "- Use TypeScript with Vue for better development experience",
                    ""
                ])
        
        return rules
    
    def _generate_error_handling_rules(self, primary_tech: str) -> List[str]:
        """Generate error handling rules."""
        if primary_tech == 'python':
            return [
                "- Use specific exception types rather than generic Exception",
                "- Implement proper logging with appropriate levels",
                "- Use try-except blocks judiciously, not as flow control",
                "- Provide meaningful error messages to users",
                "- Use context managers to ensure proper resource cleanup"
            ]
        elif primary_tech in ['javascript', 'typescript', 'react', 'nextjs']:
            return [
                "- Use try-catch blocks for error handling",
                "- Implement proper error boundaries in React applications",
                "- Use async/await with proper error handling",
                "- Provide user-friendly error messages",
                "- Log errors appropriately for debugging"
            ]
        else:
            return [
                "- Implement comprehensive error handling",
                "- Log errors with sufficient context",
                "- Provide graceful degradation",
                "- Use appropriate error status codes"
            ]
    
    def _generate_performance_rules(self, primary_tech: str) -> List[str]:
        """Generate performance rules."""
        if primary_tech in ['react', 'nextjs']:
            return [
                "- Use React.memo and useMemo for expensive calculations",
                "- Implement code splitting and lazy loading",
                "- Optimize bundle size and loading performance",
                "- Use proper caching strategies",
                "- Minimize re-renders with proper dependency arrays"
            ]
        elif primary_tech == 'python':
            return [
                "- Use generators for memory-efficient iteration",
                "- Profile code to identify bottlenecks",
                "- Use appropriate data structures for the task",
                "- Implement caching where beneficial",
                "- Consider async/await for I/O operations"
            ]
        else:
            return [
                "- Optimize for both time and space complexity",
                "- Use appropriate caching mechanisms",
                "- Minimize unnecessary computations",
                "- Consider scalability in design decisions"
            ]
    
    def _generate_critical_instructions(self, primary_tech: str, domain_info: Dict) -> List[str]:
        """Generate critical do's and don'ts."""
        instructions = [
            "**NEVER:**",
            "- Ignore error handling or edge cases",
            "- Use deprecated APIs or methods", 
            "- Hardcode sensitive information",
            "- Skip input validation and sanitization",
            "- Use any or unknown types without proper justification",
            "",
            "**ALWAYS:**",
            "- Follow security best practices",
            "- Test your code thoroughly", 
            "- Document complex logic and algorithms",
            "- Consider accessibility and user experience",
            "- Use version control with meaningful commit messages"
        ]
        
        if domain_info['is_api_docs']:
            instructions.extend([
                "- Validate all input parameters",
                "- Return appropriate HTTP status codes",
                "- Implement proper authentication/authorization"
            ])
        
        if primary_tech in ['react', 'nextjs']:
            instructions.extend([
                "- Use keys properly in lists and iterations",
                "- Avoid direct DOM manipulation",
                "- Follow React best practices for state updates"
            ])
        
        return instructions
    
    def _extract_key_concepts(self, results: List[ScrapingResult]) -> List[str]:
        """Extract key concepts from the documentation."""
        concepts = set()
        
        for result in results:
            if result.sections:
                for section in result.sections:
                    section_dict = (
                        section.model_dump() if hasattr(section, 'model_dump') else section.dict() if hasattr(section, 'dict') else section
                    )
                    title = section_dict.get('title', '')
                    if title:
                        # Extract meaningful concepts from titles
                        concept_words = re.findall(r'\b[A-Z][a-z]+\b', title)
                        concepts.update(concept_words[:3])  # Limit concepts per section
        
        return list(concepts)[:10]  # Limit total concepts
    
    def _extract_code_patterns(self, results: List[ScrapingResult]) -> List[str]:
        """Extract common code patterns from the documentation."""
        patterns = []
        
        for result in results:
            content = result.content or ''
            
            # Look for common patterns
            if 'import' in content.lower():
                patterns.append('module imports')
            if 'class' in content.lower():
                patterns.append('class definitions')
            if 'function' in content.lower() or 'def ' in content:
                patterns.append('function definitions')
            if 'async' in content.lower():
                patterns.append('async operations')
            if 'component' in content.lower():
                patterns.append('component patterns')
        
        return list(set(patterns))[:5]  # Limit and deduplicate
    
    def _extract_best_practices(self, results: List[ScrapingResult]) -> List[str]:
        """Extract best practices from the documentation."""
        practices = []
        
        for result in results:
            content = result.content or ''
            content_lower = content.lower()
            
            # Look for best practice indicators
            if 'best practice' in content_lower:
                practices.append("- Follow documented best practices")
            if 'recommendation' in content_lower:
                practices.append("- Implement recommended approaches")
            if 'avoid' in content_lower:
                practices.append("- Avoid documented anti-patterns")
            if 'performance' in content_lower:
                practices.append("- Consider performance implications")
            if 'security' in content_lower:
                practices.append("- Implement security considerations")
        
        return list(set(practices))[:5]  # Limit and deduplicate
    
    def _format_documentation_insights(self, concepts: List[str], results: List[ScrapingResult]) -> List[str]:
        """Format documentation insights."""
        insights = []
        
        if concepts:
            insights.append(f"- Key concepts covered: {', '.join(concepts)}")
        
        # Add URL references
        urls = [str(result.url) for result in results[:3]]  # Limit to first 3 URLs
        if urls:
            insights.append(f"- Source documentation: {', '.join(urls)}")
        
        insights.extend([
            "- Refer to the original documentation for detailed examples",
            "- Follow the established patterns shown in the documentation",
            "- Keep up with the latest updates and best practices"
        ])
        
        return insights
    
    def _create_empty_rules(self) -> str:
        """Create empty rules template."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""You are an expert software developer and coding assistant.

## Key Principles

- Write clean, readable, and maintainable code
- Follow established coding standards and best practices  
- Implement proper error handling and validation
- Consider performance and security implications

## Code Style and Structure

- Use descriptive names for variables, functions, and classes
- Keep functions focused and modular
- Add meaningful comments for complex logic
- Follow consistent formatting and indentation

## Best Practices

- Test your code thoroughly
- Handle edge cases appropriately  
- Use version control effectively
- Document your code and APIs

## Error Handling

- Implement comprehensive error handling
- Provide meaningful error messages
- Log errors with sufficient context
- Fail gracefully when possible

---
*Generated by Rules Maker on {timestamp}*"""
