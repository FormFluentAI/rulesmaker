"""
User Behavior Learning System.

Tracks user interactions and preferences to continuously improve the system's
recommendations and rule generation capabilities.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

from ..intelligence.models import InteractiveSession, UserIntent, ProjectAnalysis


@dataclass
class UserBehaviorPattern:
    """Represents learned user behavior patterns."""
    preferred_frameworks: Dict[str, float]
    rule_usage_patterns: Dict[str, int]
    content_preferences: Dict[str, float]
    workflow_efficiency: Dict[str, float]
    interaction_frequency: float
    session_duration_avg: float
    feature_adoption: Dict[str, bool]
    error_patterns: List[str]
    success_indicators: Dict[str, int]


@dataclass
class UserProfile:
    """Comprehensive user profile with learned preferences."""
    user_id: str
    creation_date: str
    last_updated: str
    total_sessions: int
    total_rules_generated: int
    behavior_patterns: UserBehaviorPattern
    skill_progression: Dict[str, str]  # technology -> skill_level
    preferred_learning_style: str
    customization_preferences: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]


class UserBehaviorTracker:
    """Learn from user interactions and preferences."""
    
    def __init__(self, data_dir: str = "data/user_behavior"):
        """Initialize the behavior tracker.
        
        Args:
            data_dir: Directory to store user behavior data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.current_session_data = {}
        self.user_profiles = {}
        self.global_patterns = self._load_global_patterns()
        
    def _load_global_patterns(self) -> Dict:
        """Load global usage patterns across all users."""
        patterns_path = os.path.join(self.data_dir, "global_patterns.json")
        if os.path.exists(patterns_path):
            with open(patterns_path, 'r') as f:
                return json.load(f)
        return {
            "popular_frameworks": defaultdict(int),
            "common_workflows": defaultdict(int),
            "successful_patterns": defaultdict(list),
            "problem_areas": defaultdict(list)
        }
    
    def _save_global_patterns(self):
        """Save global usage patterns."""
        patterns_path = os.path.join(self.data_dir, "global_patterns.json")
        # Convert defaultdicts to regular dicts for JSON serialization
        patterns_data = {
            "popular_frameworks": dict(self.global_patterns["popular_frameworks"]),
            "common_workflows": dict(self.global_patterns["common_workflows"]),
            "successful_patterns": dict(self.global_patterns["successful_patterns"]),
            "problem_areas": dict(self.global_patterns["problem_areas"])
        }
        with open(patterns_path, 'w') as f:
            json.dump(patterns_data, f, indent=2)
    
    def load_user_profile(self, user_id: str = "default") -> Optional[UserProfile]:
        """Load user profile from storage.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            UserProfile object or None if not found
        """
        profile_path = os.path.join(self.data_dir, f"profile_{user_id}.json")
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                data = json.load(f)
                # Convert dict back to UserBehaviorPattern
                if 'behavior_patterns' in data:
                    data['behavior_patterns'] = UserBehaviorPattern(**data['behavior_patterns'])
                return UserProfile(**data)
        return None
    
    def save_user_profile(self, profile: UserProfile):
        """Save user profile to storage.
        
        Args:
            profile: UserProfile object to save
        """
        profile_path = os.path.join(self.data_dir, f"profile_{profile.user_id}.json")
        # Convert to dict for JSON serialization
        profile_dict = asdict(profile)
        with open(profile_path, 'w') as f:
            json.dump(profile_dict, f, indent=2, default=str)
    
    async def track_user_session(self, session: InteractiveSession, user_id: str = "default"):
        """Capture user behavior patterns for system improvement.
        
        Args:
            session: Interactive session to analyze
            user_id: User identifier
        """
        # Load or create user profile
        profile = self.load_user_profile(user_id)
        if not profile:
            profile = self._create_new_user_profile(user_id)
        
        # Analyze session data
        session_patterns = await self._analyze_session_patterns(session)
        
        # Update user profile with new patterns
        self._update_user_profile(profile, session_patterns, session)
        
        # Update global patterns
        self._update_global_patterns(session_patterns, session)
        
        # Save updates
        self.save_user_profile(profile)
        self._save_global_patterns()
    
    def _create_new_user_profile(self, user_id: str) -> UserProfile:
        """Create a new user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            New UserProfile object
        """
        return UserProfile(
            user_id=user_id,
            creation_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            total_sessions=0,
            total_rules_generated=0,
            behavior_patterns=UserBehaviorPattern(
                preferred_frameworks={},
                rule_usage_patterns={},
                content_preferences={},
                workflow_efficiency={},
                interaction_frequency=0.0,
                session_duration_avg=0.0,
                feature_adoption={},
                error_patterns=[],
                success_indicators={}
            ),
            skill_progression={},
            preferred_learning_style="balanced",
            customization_preferences={},
            feedback_history=[]
        )
    
    async def _analyze_session_patterns(self, session: InteractiveSession) -> Dict:
        """Analyze patterns from a user session.
        
        Args:
            session: Interactive session to analyze
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            "frameworks_used": [],
            "workflow_steps": session.completed_steps,
            "session_duration": self._calculate_session_duration(session),
            "features_used": [],
            "errors_encountered": [],
            "success_metrics": {},
            "interaction_count": len(session.completed_steps),
            "user_preferences": {}
        }
        
        # Analyze user context for framework preferences
        if session.user_context:
            patterns["frameworks_used"] = session.user_context.technologies
            patterns["user_preferences"]["project_type"] = session.user_context.project_type
            patterns["user_preferences"]["experience_level"] = session.user_context.experience_level.value
            patterns["user_preferences"]["goals"] = session.user_context.goals
        
        # Analyze recommendations interaction
        if session.recommended_sources:
            patterns["features_used"].append("smart_recommendations")
            patterns["success_metrics"]["recommendations_accepted"] = len(session.recommended_sources)
        
        # Analyze processing results
        if "generation_results" in session.metadata:
            results = session.metadata["generation_results"]
            patterns["success_metrics"]["rules_generated"] = results.get("rules_generated", 0)
            patterns["success_metrics"]["sources_processed"] = results.get("sources_processed", 0)
            
            if results.get("errors"):
                patterns["errors_encountered"].extend(results["errors"])
        
        # Analyze workflow efficiency
        expected_steps = ["user_context", "source_recommendation", "processing_workflow", "rule_generation"]
        completed_ratio = len([step for step in expected_steps if step in session.completed_steps]) / len(expected_steps)
        patterns["workflow_efficiency"] = completed_ratio
        
        return patterns
    
    def _calculate_session_duration(self, session: InteractiveSession) -> float:
        """Calculate session duration in minutes.
        
        Args:
            session: Interactive session
            
        Returns:
            Duration in minutes
        """
        if session.start_time:
            try:
                start_time = datetime.fromisoformat(session.start_time)
                # Approximate end time based on last update or current time
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() / 60
                return max(duration, 0.1)  # Minimum 0.1 minutes
            except ValueError:
                pass
        
        # Fallback estimation based on completed steps
        return len(session.completed_steps) * 2.0  # ~2 minutes per step
    
    def _update_user_profile(self, profile: UserProfile, patterns: Dict, session: InteractiveSession):
        """Update user profile with new patterns.
        
        Args:
            profile: User profile to update
            patterns: Detected patterns from session
            session: Interactive session data
        """
        profile.last_updated = datetime.now().isoformat()
        profile.total_sessions += 1
        
        # Update rules generated count
        profile.total_rules_generated += patterns["success_metrics"].get("rules_generated", 0)
        
        # Update framework preferences
        for framework in patterns["frameworks_used"]:
            current_count = profile.behavior_patterns.preferred_frameworks.get(framework, 0.0)
            profile.behavior_patterns.preferred_frameworks[framework] = current_count + 1.0
        
        # Update workflow efficiency (moving average)
        current_efficiency = profile.behavior_patterns.workflow_efficiency.get("overall", 0.0)
        new_efficiency = patterns["workflow_efficiency"]
        
        # Weighted moving average (give more weight to recent sessions)
        weight = 0.3  # Weight for new session
        profile.behavior_patterns.workflow_efficiency["overall"] = (
            current_efficiency * (1 - weight) + new_efficiency * weight
        )
        
        # Update session duration average
        current_avg = profile.behavior_patterns.session_duration_avg
        new_duration = patterns["session_duration"]
        
        if profile.total_sessions == 1:
            profile.behavior_patterns.session_duration_avg = new_duration
        else:
            # Moving average
            profile.behavior_patterns.session_duration_avg = (
                current_avg * 0.7 + new_duration * 0.3
            )
        
        # Update interaction frequency (sessions per week, estimated)
        now = datetime.now()
        creation_date = datetime.fromisoformat(profile.creation_date)
        weeks_since_creation = max((now - creation_date).days / 7.0, 0.1)
        profile.behavior_patterns.interaction_frequency = profile.total_sessions / weeks_since_creation
        
        # Update feature adoption
        for feature in patterns["features_used"]:
            profile.behavior_patterns.feature_adoption[feature] = True
        
        # Update error patterns (keep last 10 errors)
        if patterns["errors_encountered"]:
            profile.behavior_patterns.error_patterns.extend(patterns["errors_encountered"])
            profile.behavior_patterns.error_patterns = profile.behavior_patterns.error_patterns[-10:]
        
        # Update success indicators
        for metric, value in patterns["success_metrics"].items():
            current_total = profile.behavior_patterns.success_indicators.get(metric, 0)
            profile.behavior_patterns.success_indicators[metric] = current_total + value
        
        # Update skill progression based on framework usage and session complexity
        self._update_skill_progression(profile, patterns)
        
        # Update learning style preferences
        self._infer_learning_style(profile, patterns, session)
    
    def _update_skill_progression(self, profile: UserProfile, patterns: Dict):
        """Update skill progression tracking.
        
        Args:
            profile: User profile to update
            patterns: Detected session patterns
        """
        for framework in patterns["frameworks_used"]:
            current_level = profile.skill_progression.get(framework, "beginner")
            usage_count = profile.behavior_patterns.preferred_frameworks.get(framework, 0)
            
            # Simple progression logic based on usage frequency and success
            success_rate = patterns.get("workflow_efficiency", 0.0)
            
            if usage_count >= 10 and success_rate > 0.8:
                new_level = "expert"
            elif usage_count >= 5 and success_rate > 0.6:
                new_level = "advanced"  
            elif usage_count >= 2 and success_rate > 0.4:
                new_level = "intermediate"
            else:
                new_level = "beginner"
            
            # Only progress forward, never regress
            level_order = ["beginner", "intermediate", "advanced", "expert"]
            current_idx = level_order.index(current_level)
            new_idx = level_order.index(new_level)
            
            if new_idx > current_idx:
                profile.skill_progression[framework] = new_level
    
    def _infer_learning_style(self, profile: UserProfile, patterns: Dict, session: InteractiveSession):
        """Infer user's preferred learning style.
        
        Args:
            profile: User profile to update
            patterns: Session patterns
            session: Interactive session data
        """
        # Analyze user behavior to infer learning preferences
        style_indicators = {
            "visual": 0,
            "hands_on": 0,
            "structured": 0,
            "exploratory": 0
        }
        
        # Check for structured workflow preference
        if len(session.completed_steps) >= 4:  # Completed most steps
            style_indicators["structured"] += 2
        
        # Check for exploration vs. guided approach
        if session.user_context and len(session.user_context.technologies) > 3:
            style_indicators["exploratory"] += 1
        else:
            style_indicators["structured"] += 1
        
        # Check for hands-on preference (project analysis usage)
        if session.project_analysis:
            style_indicators["hands_on"] += 2
        
        # Check session duration (longer sessions might indicate thorough/structured approach)
        if patterns["session_duration"] > 20:  # >20 minutes
            style_indicators["structured"] += 1
        elif patterns["session_duration"] < 5:  # <5 minutes  
            style_indicators["exploratory"] += 1
        
        # Determine dominant style
        dominant_style = max(style_indicators.items(), key=lambda x: x[1])[0]
        
        # Update profile with weighted average
        current_style = profile.preferred_learning_style
        if current_style == "balanced" or style_indicators[dominant_style] >= 3:
            profile.preferred_learning_style = dominant_style
    
    def _update_global_patterns(self, patterns: Dict, session: InteractiveSession):
        """Update global usage patterns.
        
        Args:
            patterns: Session patterns
            session: Interactive session data
        """
        # Update popular frameworks
        for framework in patterns["frameworks_used"]:
            self.global_patterns["popular_frameworks"][framework] += 1
        
        # Update common workflows
        workflow_signature = "_".join(sorted(patterns["workflow_steps"]))
        self.global_patterns["common_workflows"][workflow_signature] += 1
        
        # Record successful patterns
        if patterns["workflow_efficiency"] > 0.8:
            success_key = f"{patterns.get('user_preferences', {}).get('project_type', 'unknown')}_success"
            self.global_patterns["successful_patterns"][success_key].append({
                "frameworks": patterns["frameworks_used"],
                "workflow": patterns["workflow_steps"],
                "duration": patterns["session_duration"],
                "efficiency": patterns["workflow_efficiency"]
            })
        
        # Record problem areas
        if patterns["errors_encountered"]:
            for error in patterns["errors_encountered"]:
                self.global_patterns["problem_areas"]["errors"].append(error)
    
    async def get_user_insights(self, user_id: str = "default") -> Dict:
        """Get insights about user behavior and preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user insights
        """
        profile = self.load_user_profile(user_id)
        if not profile:
            return {"error": "User profile not found"}
        
        insights = {
            "user_summary": {
                "total_sessions": profile.total_sessions,
                "total_rules_generated": profile.total_rules_generated,
                "account_age_days": self._calculate_account_age_days(profile),
                "avg_session_duration": profile.behavior_patterns.session_duration_avg,
                "interaction_frequency": profile.behavior_patterns.interaction_frequency
            },
            "preferences": {
                "top_frameworks": sorted(
                    profile.behavior_patterns.preferred_frameworks.items(),
                    key=lambda x: x[1], reverse=True
                )[:5],
                "learning_style": profile.preferred_learning_style,
                "workflow_efficiency": profile.behavior_patterns.workflow_efficiency.get("overall", 0.0)
            },
            "skill_progression": profile.skill_progression,
            "feature_adoption": profile.behavior_patterns.feature_adoption,
            "recent_patterns": {
                "success_rate": self._calculate_recent_success_rate(profile),
                "error_frequency": len(profile.behavior_patterns.error_patterns) / max(profile.total_sessions, 1),
                "productivity_trend": self._calculate_productivity_trend(profile)
            },
            "recommendations": self._generate_user_recommendations(profile)
        }
        
        return insights
    
    def _calculate_account_age_days(self, profile: UserProfile) -> int:
        """Calculate account age in days."""
        try:
            creation_date = datetime.fromisoformat(profile.creation_date)
            return (datetime.now() - creation_date).days
        except ValueError:
            return 0
    
    def _calculate_recent_success_rate(self, profile: UserProfile) -> float:
        """Calculate recent success rate based on error patterns."""
        recent_errors = len(profile.behavior_patterns.error_patterns)
        recent_sessions = min(profile.total_sessions, 10)  # Last 10 sessions
        
        if recent_sessions == 0:
            return 0.0
        
        return max(0.0, 1.0 - (recent_errors / recent_sessions))
    
    def _calculate_productivity_trend(self, profile: UserProfile) -> str:
        """Calculate productivity trend."""
        if profile.total_sessions < 3:
            return "insufficient_data"
        
        efficiency = profile.behavior_patterns.workflow_efficiency.get("overall", 0.0)
        avg_duration = profile.behavior_patterns.session_duration_avg
        
        # Simple heuristic: shorter sessions with high efficiency = improving productivity
        if efficiency > 0.8 and avg_duration < 15:
            return "improving"
        elif efficiency > 0.6:
            return "stable"
        else:
            return "needs_attention"
    
    def _generate_user_recommendations(self, profile: UserProfile) -> List[str]:
        """Generate personalized recommendations for user improvement.
        
        Args:
            profile: User profile
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Efficiency recommendations
        efficiency = profile.behavior_patterns.workflow_efficiency.get("overall", 0.0)
        if efficiency < 0.5:
            recommendations.append("Try using the guided workflow feature to improve session efficiency")
        
        # Feature adoption recommendations
        if not profile.behavior_patterns.feature_adoption.get("smart_recommendations", False):
            recommendations.append("Enable smart recommendations to discover more relevant documentation")
        
        # Session frequency recommendations
        if profile.behavior_patterns.interaction_frequency < 0.5:  # Less than 0.5 sessions per week
            recommendations.append("Regular practice sessions can help improve your development workflow")
        
        # Skill progression recommendations
        stagnant_skills = [
            skill for skill, level in profile.skill_progression.items()
            if level == "beginner" and profile.behavior_patterns.preferred_frameworks.get(skill, 0) >= 3
        ]
        
        if stagnant_skills:
            recommendations.append(f"Consider exploring advanced topics in: {', '.join(stagnant_skills[:3])}")
        
        # Learning style recommendations
        if profile.preferred_learning_style == "hands_on":
            recommendations.append("Try the project analysis feature to get more targeted recommendations")
        elif profile.preferred_learning_style == "structured":
            recommendations.append("Use the complete workflow steps for best results")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def get_global_insights(self) -> Dict:
        """Get insights about global usage patterns.
        
        Returns:
            Dictionary with global insights
        """
        return {
            "popular_technologies": sorted(
                self.global_patterns["popular_frameworks"].items(),
                key=lambda x: x[1], reverse=True
            )[:10],
            "common_workflows": sorted(
                self.global_patterns["common_workflows"].items(),
                key=lambda x: x[1], reverse=True
            )[:5],
            "success_patterns_count": len(self.global_patterns["successful_patterns"]),
            "total_error_reports": len(self.global_patterns["problem_areas"].get("errors", [])),
            "insights": self._generate_global_insights()
        }
    
    def _generate_global_insights(self) -> List[str]:
        """Generate insights from global usage patterns."""
        insights = []
        
        # Most popular technology
        if self.global_patterns["popular_frameworks"]:
            top_tech = max(self.global_patterns["popular_frameworks"].items(), key=lambda x: x[1])
            insights.append(f"Most popular technology: {top_tech[0]} ({top_tech[1]} sessions)")
        
        # Workflow efficiency
        successful_count = sum(len(patterns) for patterns in self.global_patterns["successful_patterns"].values())
        total_workflows = sum(self.global_patterns["common_workflows"].values())
        
        if total_workflows > 0:
            success_rate = successful_count / total_workflows
            insights.append(f"Global workflow success rate: {success_rate:.1%}")
        
        # Common problem areas
        error_count = len(self.global_patterns["problem_areas"].get("errors", []))
        if error_count > 10:
            insights.append("Consider reviewing documentation for common error patterns")
        
        return insights
    
    async def start_learning_session(self, user_id: str = "default"):
        """Start a learning session to track user behavior.
        
        Args:
            user_id: User identifier
        """
        self.current_session_data[user_id] = {
            "start_time": datetime.now(),
            "actions": [],
            "context": {}
        }
    
    def record_user_action(self, user_id: str, action: str, context: Dict = None):
        """Record a user action during a session.
        
        Args:
            user_id: User identifier
            action: Action performed by user
            context: Additional context data
        """
        if user_id in self.current_session_data:
            self.current_session_data[user_id]["actions"].append({
                "action": action,
                "timestamp": datetime.now(),
                "context": context or {}
            })
    
    async def end_learning_session(self, user_id: str = "default"):
        """End a learning session and analyze patterns.
        
        Args:
            user_id: User identifier
        """
        if user_id not in self.current_session_data:
            return
        
        session_data = self.current_session_data[user_id]
        
        # Create a mock InteractiveSession for analysis
        mock_session = InteractiveSession(
            session_id=f"learning_{user_id}_{int(datetime.now().timestamp())}",
            start_time=session_data["start_time"].isoformat(),
            completed_steps=[action["action"] for action in session_data["actions"]],
            metadata={"learning_session": True}
        )
        
        # Track the session
        await self.track_user_session(mock_session, user_id)
        
        # Clean up current session data
        del self.current_session_data[user_id]