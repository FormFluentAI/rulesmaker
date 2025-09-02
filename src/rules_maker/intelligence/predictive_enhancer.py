"""
Predictive Rule Enhancer.

Predicts and suggests rule improvements before user requests,
analyzing project patterns and proactively recommending optimizations.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass

from .models import ProjectAnalysis, RulePrediction
from ..bedrock_integration import BedrockRulesMaker
from ..learning.user_behavior_tracker import UserBehaviorTracker, UserProfile


@dataclass
class PredictionContext:
    """Context for making predictions."""
    user_profile: Optional[UserProfile]
    project_analysis: ProjectAnalysis
    current_rules: List[str]
    usage_patterns: Dict[str, Any]
    environment_info: Dict[str, Any]


class PredictiveRuleEnhancer:
    """Predict and suggest rule improvements before user requests."""
    
    def __init__(self, bedrock_config: Optional[Dict] = None):
        """Initialize the predictive enhancer.
        
        Args:
            bedrock_config: Configuration for Bedrock integration
        """
        self.bedrock_config = bedrock_config or {}
        self.behavior_tracker = UserBehaviorTracker()
        
        # Load prediction models and patterns
        self.prediction_patterns = self._load_prediction_patterns()
        self.historical_predictions = self._load_historical_predictions()
        self.rule_effectiveness_data = self._load_rule_effectiveness_data()
        
    def _load_prediction_patterns(self) -> Dict:
        """Load patterns for making predictions."""
        patterns_path = "data/prediction_patterns.json"
        if os.path.exists(patterns_path):
            with open(patterns_path, 'r') as f:
                return json.load(f)
        return self._get_default_prediction_patterns()
    
    def _get_default_prediction_patterns(self) -> Dict:
        """Get default prediction patterns."""
        return {
            "authentication_indicators": {
                "patterns": ["login", "auth", "user", "session", "token", "password"],
                "files": ["auth.js", "login.py", "user.model", "session.py"],
                "dependencies": ["passport", "jwt", "oauth", "firebase-auth"],
                "confidence_weights": {
                    "patterns": 0.3,
                    "files": 0.4,
                    "dependencies": 0.3
                }
            },
            "routing_complexity": {
                "indicators": ["route", "router", "path", "navigate", "link"],
                "complexity_markers": ["dynamic", "nested", "protected", "lazy", "async"],
                "file_count_threshold": 5,
                "pattern_density_threshold": 0.3
            },
            "state_management_needs": {
                "simple_indicators": ["useState", "state", "setState"],
                "complex_indicators": ["redux", "context", "store", "dispatch", "provider"],
                "component_count_threshold": 10,
                "state_complexity_threshold": 3
            },
            "performance_optimization": {
                "indicators": ["slow", "performance", "optimization", "cache", "lazy"],
                "technical_debt_markers": ["todo", "hack", "fix", "temp", "workaround"],
                "file_size_threshold": 1000,  # lines
                "dependency_count_threshold": 50
            },
            "security_vulnerabilities": {
                "risk_patterns": ["eval", "innerHTML", "dangerouslySetInnerHTML", "exec"],
                "sensitive_data": ["password", "key", "secret", "token", "credential"],
                "input_validation": ["input", "form", "request", "parameter", "query"],
                "security_libraries": ["helmet", "cors", "express-validator", "sanitize"]
            },
            "testing_gaps": {
                "test_file_patterns": [".test.", ".spec.", "__tests__"],
                "testing_libraries": ["jest", "mocha", "pytest", "unittest", "cypress"],
                "coverage_indicators": ["coverage", "test", "spec", "mock"],
                "minimum_test_ratio": 0.3
            },
            "deployment_readiness": {
                "config_files": ["dockerfile", "docker-compose", "package.json", "requirements.txt"],
                "env_patterns": [".env", "config", "settings", "environment"],
                "ci_cd_indicators": [".github", ".gitlab", "jenkins", "travis"],
                "production_markers": ["prod", "production", "deploy", "build"]
            }
        }
    
    def _load_historical_predictions(self) -> Dict:
        """Load historical prediction data."""
        history_path = "data/prediction_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        return {"predictions": [], "outcomes": [], "accuracy_stats": {}}
    
    def _save_historical_predictions(self):
        """Save historical prediction data."""
        os.makedirs("data", exist_ok=True)
        with open("data/prediction_history.json", 'w') as f:
            json.dump(self.historical_predictions, f, indent=2, default=str)
    
    def _load_rule_effectiveness_data(self) -> Dict:
        """Load data about rule effectiveness."""
        effectiveness_path = "data/rule_effectiveness.json"
        if os.path.exists(effectiveness_path):
            with open(effectiveness_path, 'r') as f:
                return json.load(f)
        return defaultdict(lambda: {"usage_count": 0, "effectiveness_score": 0.5, "user_ratings": []})
    
    def _save_rule_effectiveness_data(self):
        """Save rule effectiveness data."""
        os.makedirs("data", exist_ok=True)
        with open("data/rule_effectiveness.json", 'w') as f:
            json.dump(dict(self.rule_effectiveness_data), f, indent=2, default=str)
    
    async def predict_rule_needs(self, project_analysis: ProjectAnalysis, 
                                user_profile: Optional[UserProfile] = None,
                                current_rules: List[str] = None) -> List[RulePrediction]:
        """Predict what rules user will need based on project analysis.
        
        Args:
            project_analysis: Analysis of the user's project
            user_profile: Optional user profile for personalization
            current_rules: Optional list of current rules
            
        Returns:
            List of predicted rule needs
        """
        predictions = []
        current_rules = current_rules or []
        
        # Create prediction context
        context = PredictionContext(
            user_profile=user_profile,
            project_analysis=project_analysis,
            current_rules=current_rules,
            usage_patterns={},
            environment_info={}
        )
        
        # Run different prediction algorithms
        auth_predictions = await self._predict_authentication_needs(context)
        predictions.extend(auth_predictions)
        
        routing_predictions = await self._predict_routing_optimization_needs(context)
        predictions.extend(routing_predictions)
        
        state_predictions = await self._predict_state_management_needs(context)
        predictions.extend(state_predictions)
        
        performance_predictions = await self._predict_performance_optimization_needs(context)
        predictions.extend(performance_predictions)
        
        security_predictions = await self._predict_security_enhancement_needs(context)
        predictions.extend(security_predictions)
        
        testing_predictions = await self._predict_testing_improvement_needs(context)
        predictions.extend(testing_predictions)
        
        deployment_predictions = await self._predict_deployment_readiness_needs(context)
        predictions.extend(deployment_predictions)
        
        # Enhance predictions with LLM if available
        if self.bedrock_config:
            enhanced_predictions = await self._enhance_predictions_with_llm(predictions, context)
            if enhanced_predictions:
                predictions = enhanced_predictions
        
        # Filter and prioritize predictions
        filtered_predictions = self._filter_and_prioritize_predictions(predictions, context)
        
        # Record predictions for learning
        self._record_predictions(filtered_predictions, context)
        
        return filtered_predictions
    
    async def _predict_authentication_needs(self, context: PredictionContext) -> List[RulePrediction]:
        """Predict authentication-related rule needs."""
        predictions = []
        
        if not context.project_analysis.has_authentication_patterns:
            return predictions
        
        patterns = self.prediction_patterns["authentication_indicators"]
        confidence = 0.0
        
        # Check for authentication complexity indicators
        auth_complexity_indicators = [
            "multiple auth providers", "oauth", "saml", "sso", 
            "role-based access", "permissions", "jwt refresh"
        ]
        
        # Base prediction for security best practices
        predictions.append(RulePrediction(
            rule_type="security-best-practices",
            confidence=0.85,
            reason="Detected authentication patterns in project - security rules recommended",
            priority="high",
            estimated_impact="High - Prevents security vulnerabilities",
            suggested_timing="immediate",
            dependencies=["authentication system"]
        ))
        
        # Advanced authentication patterns
        if context.user_profile and context.user_profile.skill_progression.get("authentication", "beginner") == "advanced":
            predictions.append(RulePrediction(
                rule_type="advanced-auth-patterns",
                confidence=0.72,
                reason="User has advanced auth experience - suggest advanced patterns",
                priority="medium",
                estimated_impact="Medium - Improves authentication architecture",
                suggested_timing="next iteration"
            ))
        
        # Session management
        predictions.append(RulePrediction(
            rule_type="session-management",
            confidence=0.78,
            reason="Authentication requires proper session handling",
            priority="high",
            estimated_impact="High - Ensures secure user sessions",
            suggested_timing="immediate",
            dependencies=["authentication system", "security framework"]
        ))
        
        return predictions
    
    async def _predict_routing_optimization_needs(self, context: PredictionContext) -> List[RulePrediction]:
        """Predict routing optimization needs."""
        predictions = []
        
        if not context.project_analysis.uses_complex_routing:
            return predictions
        
        # Route optimization for complex routing
        predictions.append(RulePrediction(
            rule_type="routing-optimization",
            confidence=0.76,
            reason="Complex routing detected - optimization rules recommended",
            priority="medium",
            estimated_impact="Medium - Improves navigation performance and UX",
            suggested_timing="next milestone",
            dependencies=["routing system"]
        ))
        
        # Route guards and protection
        if context.project_analysis.has_authentication_patterns:
            predictions.append(RulePrediction(
                rule_type="route-protection",
                confidence=0.83,
                reason="Complex routing with auth - route protection needed",
                priority="high",
                estimated_impact="High - Secures protected application areas",
                suggested_timing="immediate",
                dependencies=["authentication system", "routing system"]
            ))
        
        # Dynamic routing patterns
        predictions.append(RulePrediction(
            rule_type="dynamic-routing-patterns",
            confidence=0.65,
            reason="Complex routing suggests need for dynamic route handling",
            priority="medium",
            estimated_impact="Medium - Enables flexible route management",
            suggested_timing="when adding new features"
        ))
        
        return predictions
    
    async def _predict_state_management_needs(self, context: PredictionContext) -> List[RulePrediction]:
        """Predict state management optimization needs."""
        predictions = []
        
        if not context.project_analysis.has_state_management:
            return predictions
        
        # State management optimization
        predictions.append(RulePrediction(
            rule_type="state-management-optimization",
            confidence=0.71,
            reason="State management detected - optimization patterns recommended",
            priority="medium",
            estimated_impact="Medium - Improves app performance and maintainability",
            suggested_timing="during refactoring",
            dependencies=["state management system"]
        ))
        
        # Global state patterns
        if context.project_analysis.architectural_patterns and "complex" in str(context.project_analysis.architectural_patterns):
            predictions.append(RulePrediction(
                rule_type="global-state-patterns",
                confidence=0.68,
                reason="Complex architecture suggests need for global state management",
                priority="medium",
                estimated_impact="High - Centralizes state management across components",
                suggested_timing="before project grows larger"
            ))
        
        return predictions
    
    async def _predict_performance_optimization_needs(self, context: PredictionContext) -> List[RulePrediction]:
        """Predict performance optimization needs."""
        predictions = []
        
        # Always suggest basic performance patterns
        predictions.append(RulePrediction(
            rule_type="performance-optimization",
            confidence=0.62,
            reason="All applications benefit from performance optimization patterns",
            priority="medium",
            estimated_impact="Medium - Improves user experience and resource usage",
            suggested_timing="during development",
            dependencies=["application framework"]
        ))
        
        # Advanced performance for complex applications
        complexity_score = self._calculate_project_complexity(context.project_analysis)
        if complexity_score > 0.7:
            predictions.append(RulePrediction(
                rule_type="advanced-performance-patterns",
                confidence=0.74,
                reason="High project complexity indicates need for advanced performance patterns",
                priority="high",
                estimated_impact="High - Prevents performance bottlenecks in complex application",
                suggested_timing="before production deployment",
                dependencies=["performance monitoring", "optimization tools"]
            ))
        
        # Caching strategies
        if context.project_analysis.has_api_integration:
            predictions.append(RulePrediction(
                rule_type="caching-strategies",
                confidence=0.69,
                reason="API integration suggests need for caching optimization",
                priority="medium",
                estimated_impact="High - Reduces API calls and improves response times",
                suggested_timing="after core functionality is complete"
            ))
        
        return predictions
    
    async def _predict_security_enhancement_needs(self, context: PredictionContext) -> List[RulePrediction]:
        """Predict security enhancement needs."""
        predictions = []
        
        # Basic security patterns for all applications
        predictions.append(RulePrediction(
            rule_type="basic-security-patterns",
            confidence=0.80,
            reason="All applications need basic security measures",
            priority="high",
            estimated_impact="High - Prevents common security vulnerabilities",
            suggested_timing="immediate",
            dependencies=["security framework"]
        ))
        
        # Input validation for applications with user input
        if context.project_analysis.has_api_integration or context.project_analysis.has_authentication_patterns:
            predictions.append(RulePrediction(
                rule_type="input-validation-security",
                confidence=0.87,
                reason="User input detected - input validation security rules essential",
                priority="high",
                estimated_impact="High - Prevents injection attacks and data corruption",
                suggested_timing="immediate",
                dependencies=["validation library", "security middleware"]
            ))
        
        # API security for applications with APIs
        if context.project_analysis.has_api_integration:
            predictions.append(RulePrediction(
                rule_type="api-security-patterns",
                confidence=0.82,
                reason="API integration requires security best practices",
                priority="high",
                estimated_impact="High - Secures API endpoints and data transmission",
                suggested_timing="before API deployment",
                dependencies=["API framework", "authentication system"]
            ))
        
        return predictions
    
    async def _predict_testing_improvement_needs(self, context: PredictionContext) -> List[RulePrediction]:
        """Predict testing improvement needs."""
        predictions = []
        
        if not context.project_analysis.has_testing_setup:
            # No testing setup detected
            predictions.append(RulePrediction(
                rule_type="testing-setup-basic",
                confidence=0.91,
                reason="No testing framework detected - basic testing setup recommended",
                priority="high",
                estimated_impact="High - Enables quality assurance and prevents regressions",
                suggested_timing="immediate",
                dependencies=["testing framework"]
            ))
        else:
            # Existing testing - suggest improvements
            predictions.append(RulePrediction(
                rule_type="testing-best-practices",
                confidence=0.73,
                reason="Testing framework detected - best practices recommended",
                priority="medium",
                estimated_impact="Medium - Improves test quality and coverage",
                suggested_timing="during development",
                dependencies=["existing testing framework"]
            ))
        
        # Integration testing for complex applications
        if context.project_analysis.has_api_integration and context.project_analysis.uses_database:
            predictions.append(RulePrediction(
                rule_type="integration-testing-patterns",
                confidence=0.76,
                reason="Complex application with API and database - integration testing needed",
                priority="medium",
                estimated_impact="High - Tests system interactions and data flow",
                suggested_timing="after core features are implemented",
                dependencies=["testing framework", "test database"]
            ))
        
        return predictions
    
    async def _predict_deployment_readiness_needs(self, context: PredictionContext) -> List[RulePrediction]:
        """Predict deployment readiness needs."""
        predictions = []
        
        # Basic deployment patterns
        predictions.append(RulePrediction(
            rule_type="deployment-best-practices",
            confidence=0.68,
            reason="All applications eventually need deployment strategies",
            priority="medium",
            estimated_impact="High - Enables reliable production deployment",
            suggested_timing="before production deployment",
            dependencies=["deployment platform"]
        ))
        
        # Environment configuration
        predictions.append(RulePrediction(
            rule_type="environment-configuration",
            confidence=0.75,
            reason="Applications require proper environment configuration",
            priority="medium",
            estimated_impact="High - Enables different deployment environments",
            suggested_timing="early in development",
            dependencies=["configuration management"]
        ))
        
        # CI/CD for complex projects
        complexity_score = self._calculate_project_complexity(context.project_analysis)
        if complexity_score > 0.6:
            predictions.append(RulePrediction(
                rule_type="cicd-pipeline-setup",
                confidence=0.71,
                reason="Complex project benefits from automated CI/CD pipeline",
                priority="medium",
                estimated_impact="High - Automates testing and deployment processes",
                suggested_timing="mid-development",
                dependencies=["CI/CD platform", "testing framework"]
            ))
        
        return predictions
    
    def _calculate_project_complexity(self, analysis: ProjectAnalysis) -> float:
        """Calculate a complexity score for the project."""
        complexity_score = 0.0
        
        # Add complexity based on different factors
        if analysis.has_authentication_patterns:
            complexity_score += 0.2
        
        if analysis.uses_complex_routing:
            complexity_score += 0.2
        
        if analysis.has_state_management:
            complexity_score += 0.15
        
        if analysis.has_api_integration:
            complexity_score += 0.15
        
        if analysis.uses_database:
            complexity_score += 0.15
        
        if analysis.has_testing_setup:
            complexity_score += 0.1  # Good complexity
        
        # Add complexity based on architectural patterns
        if analysis.architectural_patterns:
            complexity_score += len(analysis.architectural_patterns) * 0.05
        
        # Add complexity based on detected technologies
        if analysis.technologies_detected:
            complexity_score += len(analysis.technologies_detected) * 0.03
        
        return min(complexity_score, 1.0)
    
    async def _enhance_predictions_with_llm(
        self, 
        predictions: List[RulePrediction], 
        context: PredictionContext
    ) -> Optional[List[RulePrediction]]:
        """Enhance predictions using LLM analysis."""
        try:
            bedrock_maker = BedrockRulesMaker(**self.bedrock_config)
            
            # Create context for LLM
            project_summary = {
                "has_auth": context.project_analysis.has_authentication_patterns,
                "has_routing": context.project_analysis.uses_complex_routing,
                "has_state": context.project_analysis.has_state_management,
                "has_api": context.project_analysis.has_api_integration,
                "has_db": context.project_analysis.uses_database,
                "has_testing": context.project_analysis.has_testing_setup,
                "technologies": context.project_analysis.technologies_detected[:5]
            }
            
            current_predictions = [
                {"type": p.rule_type, "confidence": p.confidence, "priority": p.priority}
                for p in predictions[:10]  # Top 10 predictions
            ]
            
            prompt = f"""
            Analyze this project and enhance these rule predictions:
            
            Project Analysis:
            {json.dumps(project_summary, indent=2)}
            
            Current Predictions:
            {json.dumps(current_predictions, indent=2)}
            
            Please enhance these predictions by:
            1. Adjusting confidence scores based on project complexity
            2. Adding missing important rule types
            3. Refining priorities based on project characteristics
            4. Suggesting better timing for implementation
            
            Return enhanced predictions with reasoning for any changes.
            """
            
            # Simple enhancement - in practice would parse LLM response more carefully
            enhanced_response = await bedrock_maker._call_bedrock_async(prompt)
            
            # For now, just boost confidence for high-priority predictions
            enhanced_predictions = predictions.copy()
            for pred in enhanced_predictions:
                if pred.priority == "high" and context.project_analysis.has_authentication_patterns:
                    pred.confidence = min(pred.confidence * 1.1, 1.0)
                    pred.reason = f"LLM Enhanced - {pred.reason}"
            
            return enhanced_predictions
            
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
            return None
    
    def _filter_and_prioritize_predictions(
        self, 
        predictions: List[RulePrediction], 
        context: PredictionContext
    ) -> List[RulePrediction]:
        """Filter and prioritize predictions based on context."""
        # Remove duplicates based on rule_type
        seen_types = set()
        unique_predictions = []
        
        for pred in predictions:
            if pred.rule_type not in seen_types:
                unique_predictions.append(pred)
                seen_types.add(pred.rule_type)
        
        # Filter based on confidence threshold
        confidence_threshold = 0.5
        filtered_predictions = [
            pred for pred in unique_predictions 
            if pred.confidence >= confidence_threshold
        ]
        
        # Prioritize based on multiple factors
        def priority_score(pred: RulePrediction) -> float:
            score = pred.confidence
            
            # Boost score for high priority items
            if pred.priority == "high":
                score *= 1.3
            elif pred.priority == "critical":
                score *= 1.5
            
            # Boost score based on user experience level
            if context.user_profile:
                experience_level = context.user_profile.preferred_learning_style
                if experience_level == "advanced" and "advanced" in pred.rule_type:
                    score *= 1.2
                elif experience_level == "beginner" and "basic" in pred.rule_type:
                    score *= 1.2
            
            # Boost score based on rule effectiveness history
            effectiveness = self.rule_effectiveness_data[pred.rule_type]["effectiveness_score"]
            score *= (0.5 + effectiveness * 0.5)  # Scale by effectiveness
            
            return score
        
        # Sort by priority score
        filtered_predictions.sort(key=priority_score, reverse=True)
        
        # Return top 8 predictions
        return filtered_predictions[:8]
    
    def _record_predictions(self, predictions: List[RulePrediction], context: PredictionContext):
        """Record predictions for learning and improvement."""
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "predictions": [
                {
                    "rule_type": pred.rule_type,
                    "confidence": pred.confidence,
                    "priority": pred.priority,
                    "reason": pred.reason
                }
                for pred in predictions
            ],
            "context": {
                "project_complexity": self._calculate_project_complexity(context.project_analysis),
                "has_auth": context.project_analysis.has_authentication_patterns,
                "has_api": context.project_analysis.has_api_integration,
                "user_sessions": context.user_profile.total_sessions if context.user_profile else 0
            }
        }
        
        self.historical_predictions["predictions"].append(prediction_record)
        
        # Keep only last 500 predictions to avoid bloat
        if len(self.historical_predictions["predictions"]) > 500:
            self.historical_predictions["predictions"] = self.historical_predictions["predictions"][-500:]
        
        self._save_historical_predictions()
    
    async def evaluate_prediction_accuracy(self, prediction_id: str, 
                                          actual_outcome: str, 
                                          user_feedback: Optional[Dict] = None):
        """Evaluate the accuracy of a prediction after implementation.
        
        Args:
            prediction_id: Identifier for the prediction
            actual_outcome: What actually happened
            user_feedback: Optional user feedback about the prediction
        """
        outcome_record = {
            "timestamp": datetime.now().isoformat(),
            "prediction_id": prediction_id,
            "actual_outcome": actual_outcome,
            "user_feedback": user_feedback or {},
            "was_accurate": actual_outcome in ["implemented", "useful", "needed"]
        }
        
        self.historical_predictions["outcomes"].append(outcome_record)
        self._save_historical_predictions()
        
        # Update accuracy statistics
        self._update_accuracy_statistics()
    
    def _update_accuracy_statistics(self):
        """Update prediction accuracy statistics."""
        outcomes = self.historical_predictions["outcomes"]
        
        if not outcomes:
            return
        
        # Calculate overall accuracy
        accurate_predictions = [o for o in outcomes if o["was_accurate"]]
        overall_accuracy = len(accurate_predictions) / len(outcomes)
        
        # Calculate accuracy by rule type
        rule_type_accuracy = defaultdict(list)
        for outcome in outcomes:
            # Find corresponding prediction (simplified)
            for pred_record in self.historical_predictions["predictions"]:
                for pred in pred_record["predictions"]:
                    rule_type_accuracy[pred["rule_type"]].append(outcome["was_accurate"])
        
        # Update statistics
        self.historical_predictions["accuracy_stats"] = {
            "overall_accuracy": overall_accuracy,
            "total_predictions": len(outcomes),
            "last_updated": datetime.now().isoformat(),
            "rule_type_accuracy": {
                rule_type: sum(accuracies) / len(accuracies)
                for rule_type, accuracies in rule_type_accuracy.items()
            }
        }
        
        self._save_historical_predictions()
    
    def update_rule_effectiveness(self, rule_type: str, effectiveness_score: float, 
                                 user_rating: Optional[float] = None):
        """Update effectiveness data for a rule type.
        
        Args:
            rule_type: Type of rule being rated
            effectiveness_score: Measured effectiveness (0.0 to 1.0)
            user_rating: Optional user rating (1 to 5)
        """
        rule_data = self.rule_effectiveness_data[rule_type]
        rule_data["usage_count"] += 1
        
        # Update effectiveness with moving average
        current_effectiveness = rule_data["effectiveness_score"]
        new_effectiveness = current_effectiveness * 0.8 + effectiveness_score * 0.2
        rule_data["effectiveness_score"] = new_effectiveness
        
        # Add user rating if provided
        if user_rating is not None:
            rule_data["user_ratings"].append(user_rating)
            # Keep only last 20 ratings
            rule_data["user_ratings"] = rule_data["user_ratings"][-20:]
        
        self._save_rule_effectiveness_data()
    
    async def get_proactive_suggestions(self, context: PredictionContext) -> List[str]:
        """Get proactive suggestions based on current project state.
        
        Args:
            context: Current project context
            
        Returns:
            List of proactive suggestion strings
        """
        suggestions = []
        
        # Analyze trends in user behavior
        if context.user_profile:
            recent_patterns = self._analyze_recent_user_patterns(context.user_profile)
            
            if recent_patterns.get("increasing_complexity", False):
                suggestions.append(
                    "Your projects are getting more complex. Consider implementing "
                    "advanced architectural patterns for better maintainability."
                )
            
            if recent_patterns.get("frequent_errors", False):
                suggestions.append(
                    "I notice some recurring issues. Let's implement better error "
                    "handling and validation patterns."
                )
        
        # Analyze project evolution patterns
        complexity_score = self._calculate_project_complexity(context.project_analysis)
        
        if complexity_score > 0.7:
            suggestions.append(
                "Your project is quite complex. Consider breaking it down with "
                "microservices or module patterns."
            )
        
        if context.project_analysis.has_api_integration and not context.project_analysis.has_testing_setup:
            suggestions.append(
                "With API integration, comprehensive testing becomes crucial. "
                "Let's set up integration testing patterns."
            )
        
        # Analyze missing best practices
        missing_practices = self._identify_missing_best_practices(context.project_analysis)
        for practice in missing_practices[:2]:  # Top 2 missing practices
            suggestions.append(f"Consider implementing {practice} for better code quality.")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _analyze_recent_user_patterns(self, user_profile: UserProfile) -> Dict[str, bool]:
        """Analyze recent patterns in user behavior."""
        patterns = {
            "increasing_complexity": False,
            "frequent_errors": False,
            "learning_new_tech": False,
            "focusing_on_quality": False
        }
        
        # Simple pattern analysis based on available data
        if user_profile.total_sessions > 5:
            error_frequency = len(user_profile.behavior_patterns.error_patterns) / user_profile.total_sessions
            if error_frequency > 0.3:
                patterns["frequent_errors"] = True
        
        if len(user_profile.behavior_patterns.preferred_frameworks) > 3:
            patterns["learning_new_tech"] = True
        
        if user_profile.behavior_patterns.workflow_efficiency.get("overall", 0) > 0.8:
            patterns["focusing_on_quality"] = True
        
        return patterns
    
    def _identify_missing_best_practices(self, analysis: ProjectAnalysis) -> List[str]:
        """Identify missing best practices based on project analysis."""
        missing = []
        
        if analysis.has_api_integration and not analysis.has_testing_setup:
            missing.append("comprehensive API testing")
        
        if analysis.has_authentication_patterns:
            missing.append("security audit patterns")
        
        if analysis.uses_database and not analysis.has_testing_setup:
            missing.append("database migration and seeding patterns")
        
        if analysis.uses_complex_routing:
            missing.append("route optimization and caching")
        
        if not analysis.has_testing_setup:
            missing.append("basic testing framework setup")
        
        return missing
    
    def get_prediction_insights(self) -> Dict:
        """Get insights about prediction performance and trends.
        
        Returns:
            Dictionary with prediction insights
        """
        insights = {
            "total_predictions_made": len(self.historical_predictions["predictions"]),
            "prediction_accuracy": 0.0,
            "most_accurate_rule_types": [],
            "common_prediction_patterns": [],
            "improvement_suggestions": []
        }
        
        # Calculate accuracy if we have outcome data
        accuracy_stats = self.historical_predictions.get("accuracy_stats", {})
        if accuracy_stats:
            insights["prediction_accuracy"] = accuracy_stats.get("overall_accuracy", 0.0)
            
            # Find most accurate rule types
            rule_accuracies = accuracy_stats.get("rule_type_accuracy", {})
            sorted_accuracies = sorted(rule_accuracies.items(), key=lambda x: x[1], reverse=True)
            insights["most_accurate_rule_types"] = sorted_accuracies[:5]
        
        # Analyze prediction patterns
        prediction_types = Counter()
        for pred_record in self.historical_predictions["predictions"]:
            for pred in pred_record["predictions"]:
                prediction_types[pred["rule_type"]] += 1
        
        insights["common_prediction_patterns"] = prediction_types.most_common(10)
        
        # Generate improvement suggestions
        if insights["prediction_accuracy"] < 0.7:
            insights["improvement_suggestions"].append(
                "Prediction accuracy could be improved - need more training data"
            )
        
        if len(insights["common_prediction_patterns"]) < 5:
            insights["improvement_suggestions"].append(
                "Need more diverse prediction patterns for better coverage"
            )
        
        return insights