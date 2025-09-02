"""
Adaptive Rule Transformer.

Rules that adapt to user preferences and project context, providing
personalized rule generation based on user behavior and coding style.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
import json
import os

from ..models import ScrapingResult, RuleFormat
from ..transformers.cursor_transformer import CursorRuleTransformer
from ..transformers.windsurf_transformer import WindsurfRuleTransformer
from ..learning.user_behavior_tracker import UserBehaviorTracker, UserProfile
from ..intelligence.models import ContentAnalysis, UserIntent
from ..bedrock_integration import BedrockRulesMaker


class AdaptiveRuleTransformer:
    """Rules that adapt to user preferences and project context."""
    
    def __init__(self, bedrock_config: Optional[Dict] = None, user_id: str = "default"):
        """Initialize the adaptive rule transformer.
        
        Args:
            bedrock_config: Configuration for Bedrock integration
            user_id: User identifier for personalization
        """
        self.bedrock_config = bedrock_config or {}
        self.user_id = user_id
        self.behavior_tracker = UserBehaviorTracker()
        self.cursor_transformer = CursorRuleTransformer()
        self.windsurf_transformer = WindsurfRuleTransformer()
        
        # Load user customization preferences
        self.customization_templates = self._load_customization_templates()
        
    def _load_customization_templates(self) -> Dict:
        """Load rule customization templates."""
        templates_path = "data/customization_templates.json"
        if os.path.exists(templates_path):
            with open(templates_path, 'r') as f:
                return json.load(f)
        return self._get_default_customization_templates()
    
    def _get_default_customization_templates(self) -> Dict:
        """Get default customization templates."""
        return {
            "coding_styles": {
                "functional": {
                    "preferences": ["functional programming", "immutability", "pure functions"],
                    "avoid": ["class-based patterns", "mutable state"],
                    "emphasis": ["data transformation", "composition"]
                },
                "object_oriented": {
                    "preferences": ["classes", "encapsulation", "inheritance", "polymorphism"],
                    "avoid": ["global functions", "procedural patterns"],
                    "emphasis": ["design patterns", "SOLID principles"]
                },
                "minimalist": {
                    "preferences": ["simple solutions", "minimal dependencies", "clean code"],
                    "avoid": ["over-engineering", "complex abstractions"],
                    "emphasis": ["readability", "maintainability"]
                },
                "enterprise": {
                    "preferences": ["type safety", "documentation", "error handling", "testing"],
                    "avoid": ["quick hacks", "undocumented code"],
                    "emphasis": ["scalability", "maintainability", "team collaboration"]
                }
            },
            "experience_adjustments": {
                "beginner": {
                    "add_explanations": True,
                    "include_examples": True,
                    "emphasize_basics": True,
                    "simplify_language": True
                },
                "intermediate": {
                    "add_explanations": True,
                    "include_examples": True,
                    "emphasize_basics": False,
                    "simplify_language": False
                },
                "advanced": {
                    "add_explanations": False,
                    "include_examples": False,
                    "emphasize_basics": False,
                    "simplify_language": False,
                    "include_advanced_patterns": True
                },
                "expert": {
                    "add_explanations": False,
                    "include_examples": False,
                    "emphasize_basics": False,
                    "simplify_language": False,
                    "include_advanced_patterns": True,
                    "focus_on_edge_cases": True
                }
            },
            "framework_customizations": {
                "react": {
                    "functional_components": "Prefer functional components with hooks",
                    "state_management": "Use appropriate state management patterns",
                    "performance": "Emphasize performance optimizations and memoization"
                },
                "nextjs": {
                    "app_router": "Prioritize App Router patterns over Pages Router",
                    "server_components": "Leverage Server Components for optimal performance",
                    "data_fetching": "Use modern data fetching patterns"
                },
                "python": {
                    "type_hints": "Always use type hints for better code clarity",
                    "async": "Use async/await for I/O operations",
                    "error_handling": "Implement proper exception handling"
                }
            }
        }
    
    async def generate_personalized_rules(
        self, 
        content: str, 
        user_profile: UserProfile,
        rule_format: RuleFormat = RuleFormat.CURSOR,
        content_analysis: Optional[ContentAnalysis] = None
    ) -> str:
        """Generate rules adapted to user's coding style and preferences.
        
        Args:
            content: Documentation content to transform
            user_profile: User's profile with behavior patterns
            rule_format: Desired rule format
            content_analysis: Optional pre-computed content analysis
            
        Returns:
            Personalized rules string
        """
        # Analyze user's preferred code patterns
        style_preferences = await self.analyze_user_style(user_profile)
        
        # Adapt rule templates to user preferences
        template_config = self.create_adaptive_template_config(style_preferences, user_profile)
        
        # Generate contextually appropriate rules
        rules = await self.generate_context_aware_rules(
            content, template_config, rule_format, content_analysis
        )
        
        # Apply final personalization touches
        personalized_rules = self.apply_personalization_layer(rules, user_profile, style_preferences)
        
        return personalized_rules
    
    async def analyze_user_style(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze user's preferred code patterns from behavior history.
        
        Args:
            user_profile: User's profile with behavior patterns
            
        Returns:
            Dictionary of style preferences and patterns
        """
        style_preferences = {
            "coding_style": "balanced",  # functional, object_oriented, minimalist, enterprise, balanced
            "complexity_preference": "intermediate",
            "documentation_level": "standard",
            "example_preference": "some",
            "framework_patterns": {},
            "language_patterns": {},
            "personalization_strength": 0.5  # How much to personalize (0.0 to 1.0)
        }
        
        # Analyze framework usage patterns
        framework_preferences = user_profile.behavior_patterns.preferred_frameworks
        if framework_preferences:
            # Determine primary frameworks
            sorted_frameworks = sorted(framework_preferences.items(), key=lambda x: x[1], reverse=True)
            
            for framework, usage_count in sorted_frameworks[:3]:  # Top 3 frameworks
                if usage_count >= 3:  # Significant usage
                    style_preferences["framework_patterns"][framework] = {
                        "usage_level": "high" if usage_count >= 10 else "medium",
                        "customizations": self._get_framework_customizations(framework, user_profile)
                    }
        
        # Infer coding style from user behavior
        style_preferences["coding_style"] = self._infer_coding_style(user_profile)
        
        # Determine complexity preference
        style_preferences["complexity_preference"] = self._infer_complexity_preference(user_profile)
        
        # Determine documentation and example preferences
        style_preferences["documentation_level"] = self._infer_documentation_preference(user_profile)
        style_preferences["example_preference"] = self._infer_example_preference(user_profile)
        
        # Calculate personalization strength based on user data availability
        style_preferences["personalization_strength"] = self._calculate_personalization_strength(user_profile)
        
        return style_preferences
    
    def _infer_coding_style(self, user_profile: UserProfile) -> str:
        """Infer user's preferred coding style from behavior patterns."""
        # Default to balanced
        if user_profile.total_sessions < 3:
            return "balanced"
        
        # Analyze skill progression and framework preferences
        frameworks = user_profile.behavior_patterns.preferred_frameworks
        
        # Check for functional programming indicators
        functional_indicators = ["react", "javascript", "typescript", "haskell", "clojure"]
        functional_score = sum(frameworks.get(fw, 0) for fw in functional_indicators)
        
        # Check for object-oriented indicators
        oo_indicators = ["java", "csharp", "python", "kotlin", "swift"]
        oo_score = sum(frameworks.get(fw, 0) for fw in oo_indicators)
        
        # Check for enterprise patterns
        enterprise_indicators = ["spring", "django", "dotnet", "enterprise"]
        enterprise_score = sum(frameworks.get(fw, 0) for fw in enterprise_indicators)
        
        # Check workflow efficiency for minimalist tendency
        efficiency = user_profile.behavior_patterns.workflow_efficiency.get("overall", 0.5)
        avg_duration = user_profile.behavior_patterns.session_duration_avg
        
        # Decision logic
        if enterprise_score > 0 and efficiency > 0.7:
            return "enterprise"
        elif functional_score > oo_score * 1.5:
            return "functional"
        elif oo_score > functional_score * 1.5:
            return "object_oriented"
        elif avg_duration < 10 and efficiency > 0.8:  # Quick, efficient sessions
            return "minimalist"
        else:
            return "balanced"
    
    def _infer_complexity_preference(self, user_profile: UserProfile) -> str:
        """Infer user's complexity preference from skill progression."""
        skill_levels = user_profile.skill_progression
        
        if not skill_levels:
            return "intermediate"
        
        # Count skill levels
        level_counts = {"beginner": 0, "intermediate": 0, "advanced": 0, "expert": 0}
        for level in skill_levels.values():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Find dominant level
        dominant_level = max(level_counts.items(), key=lambda x: x[1])[0]
        
        # Adjust based on workflow efficiency
        efficiency = user_profile.behavior_patterns.workflow_efficiency.get("overall", 0.5)
        if efficiency > 0.9 and dominant_level != "beginner":
            # High efficiency might indicate readiness for more complexity
            level_order = ["beginner", "intermediate", "advanced", "expert"]
            current_idx = level_order.index(dominant_level)
            if current_idx < len(level_order) - 1:
                return level_order[current_idx + 1]
        
        return dominant_level
    
    def _infer_documentation_preference(self, user_profile: UserProfile) -> str:
        """Infer user's documentation preference."""
        learning_style = user_profile.preferred_learning_style
        
        if learning_style == "structured":
            return "comprehensive"
        elif learning_style == "exploratory":
            return "minimal"
        else:
            return "standard"
    
    def _infer_example_preference(self, user_profile: UserProfile) -> str:
        """Infer user's example preference."""
        learning_style = user_profile.preferred_learning_style
        
        if learning_style == "hands_on":
            return "many"
        elif learning_style == "visual":
            return "some"
        else:
            return "some"
    
    def _calculate_personalization_strength(self, user_profile: UserProfile) -> float:
        """Calculate how much to personalize based on available user data."""
        base_strength = 0.3  # Minimum personalization
        
        # Add strength based on data availability
        if user_profile.total_sessions >= 5:
            base_strength += 0.2
        
        if len(user_profile.behavior_patterns.preferred_frameworks) >= 3:
            base_strength += 0.2
        
        if len(user_profile.skill_progression) >= 2:
            base_strength += 0.2
        
        if user_profile.behavior_patterns.workflow_efficiency.get("overall", 0) > 0.6:
            base_strength += 0.1
        
        return min(base_strength, 1.0)
    
    def _get_framework_customizations(self, framework: str, user_profile: UserProfile) -> Dict:
        """Get specific customizations for a framework based on user behavior."""
        base_customizations = self.customization_templates["framework_customizations"].get(
            framework.lower(), {}
        )
        
        # Add user-specific customizations based on behavior
        user_customizations = base_customizations.copy()
        
        # Example: If user has high efficiency, emphasize performance patterns
        efficiency = user_profile.behavior_patterns.workflow_efficiency.get("overall", 0.5)
        if efficiency > 0.8:
            user_customizations["performance_focus"] = "Emphasize performance optimization patterns"
        
        return user_customizations
    
    def create_adaptive_template_config(
        self, 
        style_preferences: Dict, 
        user_profile: UserProfile
    ) -> Dict:
        """Create adaptive template configuration based on style preferences.
        
        Args:
            style_preferences: Analyzed user style preferences
            user_profile: User profile data
            
        Returns:
            Configuration for template adaptation
        """
        config = {
            "personalization_level": style_preferences["personalization_strength"],
            "coding_style_emphasis": style_preferences["coding_style"],
            "complexity_level": style_preferences["complexity_preference"],
            "documentation_style": style_preferences["documentation_level"],
            "include_examples": style_preferences["example_preference"],
            "framework_specific": style_preferences["framework_patterns"],
            "user_context": {
                "total_experience": user_profile.total_sessions,
                "preferred_frameworks": dict(user_profile.behavior_patterns.preferred_frameworks),
                "learning_style": user_profile.preferred_learning_style,
                "skill_levels": user_profile.skill_progression
            }
        }
        
        # Add style-specific adjustments
        style_config = self.customization_templates["coding_styles"].get(
            style_preferences["coding_style"], {}
        )
        config["style_adjustments"] = style_config
        
        # Add experience-level adjustments
        experience_config = self.customization_templates["experience_adjustments"].get(
            style_preferences["complexity_preference"], {}
        )
        config["experience_adjustments"] = experience_config
        
        return config
    
    async def generate_context_aware_rules(
        self,
        content: str,
        template_config: Dict,
        rule_format: RuleFormat,
        content_analysis: Optional[ContentAnalysis] = None
    ) -> str:
        """Generate contextually appropriate rules based on template configuration.
        
        Args:
            content: Documentation content
            template_config: Adaptive template configuration
            rule_format: Desired rule format
            content_analysis: Optional content analysis
            
        Returns:
            Generated rules string
        """
        # Create scraping result for transformation
        scraping_result = ScrapingResult(
            url="https://adaptive-generation.local",
            title="Adaptive Rule Generation",
            content=content
        )
        
        # Use appropriate base transformer
        if rule_format == RuleFormat.CURSOR:
            base_rules = self.cursor_transformer.transform([scraping_result])
        elif rule_format == RuleFormat.WINDSURF:
            base_rules = self.windsurf_transformer.transform([scraping_result])
        else:
            # Default to cursor format
            base_rules = self.cursor_transformer.transform([scraping_result])
        
        # Apply adaptive modifications
        adapted_rules = self._apply_adaptive_modifications(base_rules, template_config, content_analysis)
        
        # Enhance with LLM if available and personalization level is high
        if self.bedrock_config and template_config["personalization_level"] > 0.7:
            enhanced_rules = await self._enhance_rules_with_llm(
                adapted_rules, template_config, content_analysis
            )
            if enhanced_rules:
                adapted_rules = enhanced_rules
        
        return adapted_rules
    
    def _apply_adaptive_modifications(
        self,
        base_rules: str,
        template_config: Dict,
        content_analysis: Optional[ContentAnalysis]
    ) -> str:
        """Apply adaptive modifications to base rules.
        
        Args:
            base_rules: Base generated rules
            template_config: Template configuration
            content_analysis: Optional content analysis
            
        Returns:
            Modified rules string
        """
        modified_rules = base_rules
        
        # Apply coding style modifications
        modified_rules = self._apply_coding_style_modifications(modified_rules, template_config)
        
        # Apply experience level adjustments
        modified_rules = self._apply_experience_adjustments(modified_rules, template_config)
        
        # Apply framework-specific customizations
        modified_rules = self._apply_framework_customizations(modified_rules, template_config)
        
        # Apply documentation style adjustments
        modified_rules = self._apply_documentation_adjustments(modified_rules, template_config)
        
        return modified_rules
    
    def _apply_coding_style_modifications(self, rules: str, config: Dict) -> str:
        """Apply coding style specific modifications."""
        style = config.get("coding_style_emphasis", "balanced")
        style_adjustments = config.get("style_adjustments", {})
        
        if not style_adjustments:
            return rules
        
        # Add style-specific preferences
        preferences = style_adjustments.get("preferences", [])
        if preferences:
            preference_text = ", ".join(preferences)
            
            # Insert style preferences into rules
            if "## Code Style" in rules:
                rules = rules.replace(
                    "## Code Style",
                    f"## Code Style\\n\\n**Style Preference**: Emphasize {preference_text}\\n"
                )
            else:
                # Add new section
                rules += f"\\n\\n## Personalized Style Preferences\\n\\n- Emphasize: {preference_text}\\n"
        
        # Add things to avoid
        avoid = style_adjustments.get("avoid", [])
        if avoid:
            avoid_text = ", ".join(avoid)
            rules += f"\\n- Avoid: {avoid_text}\\n"
        
        return rules
    
    def _apply_experience_adjustments(self, rules: str, config: Dict) -> str:
        """Apply experience level specific adjustments."""
        experience_config = config.get("experience_adjustments", {})
        
        if not experience_config:
            return rules
        
        # Add explanations for beginners/intermediates
        if experience_config.get("add_explanations", False):
            # Add explanation section
            if "## Additional Context" not in rules:
                rules += "\\n\\n## Additional Context\\n\\n"
                rules += "💡 **Why These Rules Matter**: These guidelines help ensure code quality, "
                rules += "maintainability, and team collaboration.\\n"
        
        # Simplify language for beginners
        if experience_config.get("simplify_language", False):
            # Replace complex terms with simpler alternatives
            replacements = {
                "utilize": "use",
                "implement": "create",
                "instantiate": "create",
                "leverage": "use",
                "optimize": "improve"
            }
            
            for complex_term, simple_term in replacements.items():
                rules = re.sub(
                    f"\\b{complex_term}\\b", 
                    simple_term, 
                    rules, 
                    flags=re.IGNORECASE
                )
        
        # Add advanced patterns for experienced users
        if experience_config.get("include_advanced_patterns", False):
            if "## Advanced Patterns" not in rules:
                rules += "\\n\\n## Advanced Patterns\\n\\n"
                rules += "🚀 Consider implementing advanced patterns like dependency injection, "
                rules += "design patterns, and architectural principles for scalable solutions.\\n"
        
        return rules
    
    def _apply_framework_customizations(self, rules: str, config: Dict) -> str:
        """Apply framework-specific customizations."""
        framework_patterns = config.get("framework_specific", {})
        
        for framework, framework_config in framework_patterns.items():
            customizations = framework_config.get("customizations", {})
            
            if customizations:
                # Add framework-specific section
                section_title = f"## {framework.title()} Specific Guidelines"
                if section_title not in rules:
                    rules += f"\\n\\n{section_title}\\n\\n"
                
                for key, guideline in customizations.items():
                    rules += f"- **{key.replace('_', ' ').title()}**: {guideline}\\n"
        
        return rules
    
    def _apply_documentation_adjustments(self, rules: str, config: Dict) -> str:
        """Apply documentation style adjustments."""
        doc_style = config.get("documentation_style", "standard")
        
        if doc_style == "comprehensive":
            # Add more detailed explanations
            if "## Detailed Guidelines" not in rules:
                rules += "\\n\\n## Detailed Guidelines\\n\\n"
                rules += "📚 For comprehensive understanding, refer to official documentation "
                rules += "and consider edge cases in your implementation.\\n"
        
        elif doc_style == "minimal":
            # Remove excessive explanations (simple regex-based approach)
            # Remove lines that start with explanatory text
            lines = rules.split("\\n")
            filtered_lines = []
            
            skip_patterns = ["💡", "📚", "ℹ️", "**Note:**", "**Tip:**"]
            
            for line in lines:
                if not any(pattern in line for pattern in skip_patterns):
                    filtered_lines.append(line)
                elif len(line.strip()) > 100:  # Keep longer explanatory lines
                    filtered_lines.append(line)
            
            rules = "\\n".join(filtered_lines)
        
        return rules
    
    async def _enhance_rules_with_llm(
        self,
        rules: str,
        template_config: Dict,
        content_analysis: Optional[ContentAnalysis]
    ) -> Optional[str]:
        """Enhance rules using LLM for high personalization scenarios."""
        try:
            bedrock_maker = BedrockRulesMaker(**self.bedrock_config)
            
            # Create enhancement prompt
            user_context = template_config.get("user_context", {})
            
            prompt = f"""
            Enhance these coding rules based on the user's preferences and experience:
            
            User Profile:
            - Experience Level: {template_config.get('complexity_level', 'intermediate')}
            - Preferred Style: {template_config.get('coding_style_emphasis', 'balanced')}
            - Learning Style: {user_context.get('learning_style', 'balanced')}
            - Top Frameworks: {list(user_context.get('preferred_frameworks', {}).keys())[:3]}
            
            Current Rules:
            {rules}
            
            Please enhance these rules by:
            1. Adding personalized recommendations based on the user's framework experience
            2. Adjusting the tone and complexity for their experience level
            3. Including relevant examples for their preferred technologies
            4. Maintaining the original structure but making it more relevant to their needs
            
            Return the enhanced rules in the same format.
            """
            
            enhanced_rules = await bedrock_maker._call_bedrock_async(prompt)
            
            # Basic validation - ensure the response is reasonable
            if enhanced_rules and len(enhanced_rules) > len(rules) * 0.8:
                return enhanced_rules
            
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
        
        return None
    
    def apply_personalization_layer(
        self, 
        rules: str, 
        user_profile: UserProfile, 
        style_preferences: Dict
    ) -> str:
        """Apply final personalization touches to the generated rules.
        
        Args:
            rules: Generated rules string
            user_profile: User profile
            style_preferences: Style preferences
            
        Returns:
            Finalized personalized rules
        """
        personalized_rules = rules
        
        # Add personalized header
        if style_preferences["personalization_strength"] > 0.5:
            header = self._create_personalized_header(user_profile, style_preferences)
            personalized_rules = f"{header}\\n\\n{personalized_rules}"
        
        # Add user-specific footer with next steps
        footer = self._create_personalized_footer(user_profile)
        personalized_rules = f"{personalized_rules}\\n\\n{footer}"
        
        return personalized_rules
    
    def _create_personalized_header(self, user_profile: UserProfile, style_preferences: Dict) -> str:
        """Create a personalized header for the rules."""
        style = style_preferences["coding_style"]
        experience = style_preferences["complexity_preference"]
        top_frameworks = sorted(
            user_profile.behavior_patterns.preferred_frameworks.items(),
            key=lambda x: x[1], reverse=True
        )[:2]
        
        header = "# 🎯 Personalized Coding Rules\\n\\n"
        header += f"**Tailored for**: {experience.title()} developer with {style} coding style\\n"
        
        if top_frameworks:
            frameworks_text = ", ".join([fw[0].title() for fw in top_frameworks])
            header += f"**Optimized for**: {frameworks_text}\\n"
        
        header += f"**Generated**: Based on {user_profile.total_sessions} sessions and your preferences\\n"
        
        return header
    
    def _create_personalized_footer(self, user_profile: UserProfile) -> str:
        """Create a personalized footer with next steps and recommendations."""
        footer = "## 🚀 Next Steps\\n\\n"
        
        # Add recommendations based on user's progression
        recommendations = []
        
        # Skill progression recommendations
        for framework, level in user_profile.skill_progression.items():
            if level == "beginner":
                recommendations.append(f"Practice more {framework} fundamentals")
            elif level == "intermediate":
                recommendations.append(f"Explore advanced {framework} patterns")
        
        # Learning style recommendations
        if user_profile.preferred_learning_style == "hands_on":
            recommendations.append("Try building a project with these rules applied")
        elif user_profile.preferred_learning_style == "structured":
            recommendations.append("Study each rule systematically and create examples")
        
        # Add top 3 recommendations
        for i, rec in enumerate(recommendations[:3], 1):
            footer += f"{i}. {rec}\\n"
        
        footer += "\\n---\\n"
        footer += f"*Rules personalized based on your {user_profile.total_sessions} sessions. "
        footer += "Keep using the system to get even better recommendations!*"
        
        return footer