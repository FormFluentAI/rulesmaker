"""
Interactive CLI Assistant.

Guided workflow assistant for intelligent rule generation with personalized
user experiences and smart recommendations.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os

from ..intelligence.models import (
    UserIntent, InteractiveSession, RecommendedSource,
    ProjectAnalysis, ComplexityLevel
)
from ..intelligence.recommendation_engine import SmartRecommendationEngine
from ..intelligence.semantic_analyzer import SemanticAnalyzer
from ..batch_processor import MLBatchProcessor, DocumentationSource
from .user_interface import UserInterface, ProgressTracker


class InteractiveCLIAssistant:
    """Guided workflow assistant for intelligent rule generation."""
    
    def __init__(self, bedrock_config: Optional[Dict] = None):
        """Initialize the interactive CLI assistant.
        
        Args:
            bedrock_config: Configuration for Bedrock integration
        """
        self.bedrock_config = bedrock_config or {}
        self.recommendation_engine = SmartRecommendationEngine(bedrock_config)
        self.semantic_analyzer = SemanticAnalyzer(bedrock_config)
        self.ui = UserInterface()
        self.sessions_dir = "data/interactive_sessions"
        os.makedirs(self.sessions_dir, exist_ok=True)
    
    async def start_interactive_session(self, session_id: Optional[str] = None) -> InteractiveSession:
        """Launch interactive documentation processing session.
        
        Args:
            session_id: Optional existing session ID to resume
            
        Returns:
            InteractiveSession object with processing results
        """
        # Load or create session
        if session_id:
            session = self._load_session(session_id)
            self.ui.show_message(f"📋 Resuming session: {session_id}", "info")
        else:
            session_id = str(uuid.uuid4())[:8]
            session = InteractiveSession(
                session_id=session_id,
                start_time=datetime.now().isoformat(),
                current_step="initialization"
            )
            self.ui.show_message(f"🚀 Starting new interactive session: {session_id}", "success")
        
        try:
            # Step 1: Understanding user needs
            if "user_context" not in session.completed_steps:
                session.current_step = "gathering_context"
                user_context = await self.gather_user_context(session)
                session.user_context = user_context
                session.completed_steps.append("user_context")
                self._save_session(session)
            
            # Step 2: Project analysis (if applicable)
            if "project_analysis" not in session.completed_steps:
                session.current_step = "project_analysis"
                if await self.ui.confirm("Would you like me to analyze your current project for better recommendations?"):
                    project_analysis = await self.analyze_user_project()
                    session.project_analysis = project_analysis
                session.completed_steps.append("project_analysis")
                self._save_session(session)
            
            # Step 3: Smart source recommendation
            if "source_recommendation" not in session.completed_steps:
                session.current_step = "source_recommendation"
                recommended_sources = await self.recommend_sources(session.user_context, session.project_analysis)
                session.recommended_sources = recommended_sources
                session.completed_steps.append("source_recommendation")
                self._save_session(session)
            
            # Step 4: Guided processing workflow
            if "processing_workflow" not in session.completed_steps:
                session.current_step = "processing_workflow"
                processing_plan = await self.create_processing_plan(session.recommended_sources)
                session.metadata["processing_plan"] = processing_plan
                session.completed_steps.append("processing_workflow")
                self._save_session(session)
            
            # Step 5: Execute guided processing with real-time feedback
            if "rule_generation" not in session.completed_steps:
                session.current_step = "rule_generation"
                results = await self.execute_guided_processing(session)
                session.metadata["generation_results"] = results
                session.completed_steps.append("rule_generation")
                session.current_step = "completed"
                self._save_session(session)
            
            # Step 6: Collect feedback and suggestions
            await self.collect_session_feedback(session)
            
            self.ui.show_message("✨ Interactive session completed successfully!", "success")
            return session
            
        except Exception as e:
            session.current_step = "error"
            session.metadata["error"] = str(e)
            self._save_session(session)
            self.ui.show_message(f"❌ Session error: {e}", "error")
            raise
    
    async def gather_user_context(self, session: InteractiveSession) -> UserIntent:
        """Gather user context and intentions.
        
        Args:
            session: Current interactive session
            
        Returns:
            UserIntent object with user preferences and goals
        """
        self.ui.show_header("🎯 Understanding Your Needs")
        
        # Project type discovery
        self.ui.show_message("Let's start by understanding what you're building:", "info")
        
        project_types = [
            "web-application", "mobile-app", "api-service", "desktop-app",
            "data-science", "machine-learning", "devops", "library", "other"
        ]
        
        project_type = await self.ui.select_option(
            "What type of project are you working on?",
            project_types
        )
        
        if project_type == "other":
            project_type = await self.ui.get_input("Please describe your project type:")
        
        # Technology stack discovery
        self.ui.show_message("\\nNow let's identify your technology stack:", "info")
        
        common_technologies = [
            "JavaScript", "TypeScript", "Python", "Java", "C#", "Go", "Rust",
            "React", "Vue.js", "Angular", "Next.js", "Django", "FastAPI", 
            "Spring Boot", "Node.js", "Express", "Flask"
        ]
        
        technologies = await self.ui.multi_select(
            "Which technologies are you using? (select all that apply)",
            common_technologies
        )
        
        # Add custom technologies
        if await self.ui.confirm("Any other technologies not listed above?"):
            custom_tech = await self.ui.get_input("Please list additional technologies (comma-separated):")
            if custom_tech:
                technologies.extend([tech.strip() for tech in custom_tech.split(",")])
        
        # Experience level
        experience_level = await self.ui.select_option(
            "What's your experience level with these technologies?",
            ["beginner", "intermediate", "advanced", "expert"]
        )
        
        # Goals and objectives
        self.ui.show_message("\\nWhat are your main goals for this project?", "info")
        goals = []
        
        common_goals = [
            "Learn best practices", "Improve code quality", "Increase productivity",
            "Follow industry standards", "Optimize performance", "Enhance security",
            "Better documentation", "Team collaboration", "Rapid prototyping"
        ]
        
        selected_goals = await self.ui.multi_select(
            "Select your primary goals:",
            common_goals
        )
        goals.extend(selected_goals)
        
        # Custom goals
        if await self.ui.confirm("Any other specific goals?"):
            custom_goals = await self.ui.get_input("Describe your other goals:")
            if custom_goals:
                goals.append(custom_goals)
        
        # Time and constraints
        time_budget = None
        if await self.ui.confirm("Do you have any time constraints?"):
            time_budget = await self.ui.get_numeric_input(
                "How much time do you have (in minutes)?",
                min_value=5, max_value=480
            )
        
        # Learning preferences
        learning_prefs = await self.ui.multi_select(
            "How do you prefer to learn?",
            ["examples", "tutorials", "documentation", "best-practices", "troubleshooting"]
        )
        
        return UserIntent(
            project_type=project_type,
            technologies=technologies,
            experience_level=ComplexityLevel(experience_level),
            goals=goals,
            time_budget=time_budget,
            learning_preferences=learning_prefs,
            constraints={"interactive_session": True}
        )
    
    async def analyze_user_project(self) -> Optional[ProjectAnalysis]:
        """Analyze user's current project for better recommendations.
        
        Returns:
            ProjectAnalysis object with detected patterns and technologies
        """
        self.ui.show_header("🔍 Project Analysis")
        
        # Check if user has a project directory they want analyzed
        if await self.ui.confirm("Would you like me to analyze your current project directory?"):
            project_path = await self.ui.get_input(
                "Enter the path to your project directory (or press Enter for current directory):"
            ) or "."
            
            try:
                # Simple project analysis based on file patterns
                analysis = ProjectAnalysis()
                
                # Check for common patterns
                patterns = await self._analyze_project_directory(project_path)
                
                analysis.has_authentication_patterns = patterns.get("auth", False)
                analysis.uses_complex_routing = patterns.get("routing", False)
                analysis.has_state_management = patterns.get("state", False)
                analysis.has_api_integration = patterns.get("api", False)
                analysis.uses_database = patterns.get("database", False)
                analysis.has_testing_setup = patterns.get("testing", False)
                analysis.technologies_detected = patterns.get("technologies", [])
                analysis.architectural_patterns = patterns.get("architecture", [])
                
                self.ui.show_message("📊 Project analysis completed!", "success")
                self._display_project_analysis(analysis)
                
                return analysis
                
            except Exception as e:
                self.ui.show_message(f"⚠️  Could not analyze project: {e}", "warning")
                return None
        
        return None
    
    async def recommend_sources(self, user_context: UserIntent, 
                              project_analysis: Optional[ProjectAnalysis] = None) -> List[RecommendedSource]:
        """Get smart source recommendations.
        
        Args:
            user_context: User's stated intentions and preferences
            project_analysis: Optional project analysis results
            
        Returns:
            List of recommended documentation sources
        """
        self.ui.show_header("🎯 Smart Recommendations")
        
        with ProgressTracker("Generating personalized recommendations...") as progress:
            recommendations = await self.recommendation_engine.recommend_documentation_sources(
                user_context, project_analysis
            )
        
        if not recommendations:
            self.ui.show_message("⚠️  No specific recommendations found. Using default sources.", "warning")
            return []
        
        # Display recommendations
        self.ui.show_message("📚 Here are my recommendations for you:", "info")
        
        for i, rec in enumerate(recommendations[:10], 1):  # Show top 10
            self.ui.show_message(
                f"{i}. **{rec.framework or 'General'}** - Priority: {rec.priority}/5\\n"
                f"   Source: {rec.source}\\n"
                f"   Reason: {rec.reason}\\n"
                f"   Value: {rec.estimated_value}\\n",
                "info"
            )
        
        # Let user customize recommendations
        if await self.ui.confirm("Would you like to customize these recommendations?"):
            recommendations = await self._customize_recommendations(recommendations)
        
        return recommendations
    
    async def create_processing_plan(self, sources: List[RecommendedSource]) -> Dict[str, Any]:
        """Create a processing plan for the recommended sources.
        
        Args:
            sources: List of recommended sources
            
        Returns:
            Processing plan with timeline and priorities
        """
        self.ui.show_header("📋 Processing Plan")
        
        if not sources:
            self.ui.show_message("⚠️  No sources to process.", "warning")
            return {"sources": [], "strategy": "none"}
        
        # Suggest processing strategy based on source count and priorities
        high_priority = [s for s in sources if s.priority >= 4]
        medium_priority = [s for s in sources if 2 <= s.priority < 4]
        
        strategy = "comprehensive" if len(sources) <= 5 else "selective"
        
        plan = {
            "total_sources": len(sources),
            "high_priority_sources": len(high_priority),
            "medium_priority_sources": len(medium_priority),
            "strategy": strategy,
            "estimated_time_minutes": len(sources) * 2,  # Rough estimate
            "processing_order": []
        }
        
        self.ui.show_message(f"📊 **Processing Strategy**: {strategy.title()}", "info")
        self.ui.show_message(f"📈 **Total Sources**: {len(sources)}", "info")
        self.ui.show_message(f"⏱️  **Estimated Time**: ~{plan['estimated_time_minutes']} minutes", "info")
        
        # Create processing order
        if strategy == "selective":
            if await self.ui.confirm("Process only high-priority sources to save time?"):
                plan["processing_order"] = high_priority
                plan["selected_sources"] = len(high_priority)
            else:
                plan["processing_order"] = sorted(sources, key=lambda x: x.priority, reverse=True)
                plan["selected_sources"] = len(sources)
        else:
            plan["processing_order"] = sorted(sources, key=lambda x: x.priority, reverse=True)
            plan["selected_sources"] = len(sources)
        
        return plan
    
    async def execute_guided_processing(self, session: InteractiveSession) -> Dict[str, Any]:
        """Execute the guided processing workflow with real-time feedback.
        
        Args:
            session: Current interactive session
            
        Returns:
            Processing results and generated rules
        """
        self.ui.show_header("⚙️  Processing Documentation")
        
        plan = session.metadata.get("processing_plan", {})
        sources = plan.get("processing_order", [])
        
        if not sources:
            return {"error": "No sources to process"}
        
        # Convert RecommendedSource to DocumentationSource for batch processor
        doc_sources = []
        for rec_source in sources:
            doc_source = DocumentationSource(
                url=rec_source.source,
                title=rec_source.framework or "Documentation",
                language="unknown",
                framework=rec_source.framework,
                priority=rec_source.priority
            )
            doc_sources.append(doc_source)
        
        # Initialize batch processor
        processor = MLBatchProcessor(
            bedrock_config=self.bedrock_config,
            output_dir=f"rules/interactive_session_{session.session_id}",
            max_concurrent=3,  # Gentle processing for interactive use
            quality_threshold=0.6
        )
        
        results = {"rules_generated": 0, "sources_processed": 0, "errors": []}
        
        try:
            with ProgressTracker(f"Processing {len(doc_sources)} documentation sources...") as progress:
                batch_result = await processor.process_documentation_batch(doc_sources)
                
                results["rules_generated"] = batch_result.total_rules_generated
                results["sources_processed"] = batch_result.sources_processed
                results["quality_metrics"] = batch_result.quality_metrics
                results["output_directory"] = processor.output_dir
            
            self.ui.show_message(
                f"✅ Processing completed!\\n"
                f"   📝 Rules generated: {results['rules_generated']}\\n"
                f"   📚 Sources processed: {results['sources_processed']}\\n"
                f"   📁 Output directory: {results['output_directory']}",
                "success"
            )
            
        except Exception as e:
            results["errors"].append(str(e))
            self.ui.show_message(f"❌ Processing error: {e}", "error")
        
        return results
    
    async def collect_session_feedback(self, session: InteractiveSession):
        """Collect user feedback about the session.
        
        Args:
            session: Completed interactive session
        """
        self.ui.show_header("💭 Your Feedback")
        
        # Overall satisfaction
        satisfaction = await self.ui.get_numeric_input(
            "How satisfied are you with this session? (1-5 scale)",
            min_value=1, max_value=5
        )
        
        # Specific feedback
        feedback = {
            "overall_satisfaction": satisfaction,
            "session_id": session.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        if satisfaction >= 4:
            self.ui.show_message("🎉 Great! We're glad you had a positive experience!", "success")
        elif satisfaction <= 2:
            self.ui.show_message("😔 We're sorry this didn't meet your expectations.", "warning")
            improvement = await self.ui.get_input("What could we improve?")
            feedback["improvement_suggestions"] = improvement
        
        # Collect specific feedback on recommendations
        if session.recommended_sources and await self.ui.confirm("Would you like to rate the recommendations?"):
            rec_feedback = {}
            for source in session.recommended_sources[:3]:  # Top 3 only to avoid fatigue
                rating = await self.ui.get_numeric_input(
                    f"How useful was: {source.source}? (1-5)",
                    min_value=1, max_value=5
                )
                rec_feedback[source.source] = rating
            feedback["recommendation_ratings"] = rec_feedback
        
        # Save feedback
        self._save_feedback(feedback)
        
        self.ui.show_message("🙏 Thank you for your feedback! It helps us improve.", "info")
    
    def _save_session(self, session: InteractiveSession):
        """Save session data to disk."""
        session_path = os.path.join(self.sessions_dir, f"session_{session.session_id}.json")
        with open(session_path, 'w') as f:
            # Convert to dict for JSON serialization
            session_dict = session.model_dump()
            json.dump(session_dict, f, indent=2, default=str)
    
    def _load_session(self, session_id: str) -> InteractiveSession:
        """Load session data from disk."""
        session_path = os.path.join(self.sessions_dir, f"session_{session_id}.json")
        if os.path.exists(session_path):
            with open(session_path, 'r') as f:
                session_dict = json.load(f)
                return InteractiveSession(**session_dict)
        
        raise FileNotFoundError(f"Session {session_id} not found")
    
    def _save_feedback(self, feedback: Dict):
        """Save user feedback to disk."""
        feedback_dir = "data/feedback"
        os.makedirs(feedback_dir, exist_ok=True)
        
        feedback_path = os.path.join(
            feedback_dir, 
            f"feedback_{feedback['session_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(feedback_path, 'w') as f:
            json.dump(feedback, f, indent=2, default=str)
    
    async def _analyze_project_directory(self, project_path: str) -> Dict[str, Any]:
        """Analyze a project directory for patterns."""
        import glob
        import os.path
        
        patterns = {
            "auth": False,
            "routing": False,
            "state": False,
            "api": False,
            "database": False,
            "testing": False,
            "technologies": [],
            "architecture": []
        }
        
        try:
            # Check for common file patterns
            files = []
            for ext in ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.json', '*.yaml', '*.yml']:
                files.extend(glob.glob(os.path.join(project_path, '**', ext), recursive=True))
            
            # Analyze file contents and names
            for file_path in files[:50]:  # Limit to avoid performance issues
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        filename = os.path.basename(file_path).lower()
                        
                        # Authentication patterns
                        if any(word in content for word in ['auth', 'login', 'jwt', 'session', 'passport']):
                            patterns["auth"] = True
                        
                        # Routing patterns
                        if any(word in content for word in ['route', 'router', 'path', 'endpoint']):
                            patterns["routing"] = True
                        
                        # State management
                        if any(word in content for word in ['redux', 'vuex', 'state', 'store', 'context']):
                            patterns["state"] = True
                        
                        # API patterns
                        if any(word in content for word in ['api', 'fetch', 'axios', 'request', 'http']):
                            patterns["api"] = True
                        
                        # Database patterns
                        if any(word in content for word in ['database', 'sql', 'mongodb', 'postgres', 'mysql']):
                            patterns["database"] = True
                        
                        # Testing patterns
                        if any(word in content for word in ['test', 'spec', 'jest', 'pytest', 'mocha']):
                            patterns["testing"] = True
                        
                        # Technology detection
                        if '.py' in file_path:
                            patterns["technologies"].append("Python")
                        elif any(ext in file_path for ext in ['.js', '.jsx']):
                            patterns["technologies"].append("JavaScript")
                        elif any(ext in file_path for ext in ['.ts', '.tsx']):
                            patterns["technologies"].append("TypeScript")
                
                except (UnicodeDecodeError, IOError):
                    continue  # Skip files that can't be read
            
            # Remove duplicates
            patterns["technologies"] = list(set(patterns["technologies"]))
            
        except Exception as e:
            self.ui.show_message(f"Error analyzing project: {e}", "warning")
        
        return patterns
    
    def _display_project_analysis(self, analysis: ProjectAnalysis):
        """Display project analysis results."""
        self.ui.show_message("📋 **Project Analysis Results:**", "info")
        
        features = []
        if analysis.has_authentication_patterns:
            features.append("🔐 Authentication")
        if analysis.uses_complex_routing:
            features.append("🛣️  Advanced Routing")
        if analysis.has_state_management:
            features.append("🗃️  State Management")
        if analysis.has_api_integration:
            features.append("🔌 API Integration")
        if analysis.uses_database:
            features.append("🗄️  Database")
        if analysis.has_testing_setup:
            features.append("🧪 Testing")
        
        if features:
            self.ui.show_message("**Detected Features:**", "info")
            for feature in features:
                self.ui.show_message(f"  • {feature}", "info")
        
        if analysis.technologies_detected:
            self.ui.show_message(f"**Technologies:** {', '.join(analysis.technologies_detected)}", "info")
    
    async def _customize_recommendations(self, recommendations: List[RecommendedSource]) -> List[RecommendedSource]:
        """Allow user to customize recommendations."""
        self.ui.show_message("🎛️  **Customization Options:**", "info")
        
        # Filter by priority
        min_priority = await self.ui.get_numeric_input(
            "Minimum priority level to include (1-5):",
            min_value=1, max_value=5, default=2
        )
        
        filtered = [rec for rec in recommendations if rec.priority >= min_priority]
        
        # Limit count if needed
        if len(filtered) > 10:
            max_sources = await self.ui.get_numeric_input(
                f"Limit to how many sources? (current: {len(filtered)}):",
                min_value=1, max_value=len(filtered), default=10
            )
            filtered = filtered[:max_sources]
        
        self.ui.show_message(f"📚 Customized to {len(filtered)} sources", "success")
        return filtered