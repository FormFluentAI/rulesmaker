"""
Next.js Learning Integration

Specialized learning system integration for Next.js documentation processing
that learns from user interactions, feedback, and usage patterns to continuously
improve categorization and rule generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

from ..models import ScrapingResult
from ..intelligence.nextjs_categorizer import NextJSCategorizer, NextJSCategory
from ..formatters.cursor_rules_formatter import CursorRulesFormatter
from .integrated_learning_system import IntegratedLearningSystem

logger = logging.getLogger(__name__)


@dataclass
class NextJSLearningEvent:
    """Learning event for Next.js documentation processing."""
    event_type: str  # 'categorization', 'rule_generation', 'user_feedback', 'usage'
    category: str
    confidence: float
    content_hash: str
    url: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    user_feedback: Optional[Dict[str, Any]] = None


@dataclass
class NextJSLearningMetrics:
    """Learning metrics for Next.js processing."""
    total_events: int
    categorization_accuracy: float
    rule_quality_score: float
    user_satisfaction: float
    category_distribution: Dict[str, int]
    improvement_trends: Dict[str, float]
    last_updated: datetime


class NextJSLearningIntegration:
    """Learning integration system for Next.js documentation processing."""
    
    def __init__(
        self,
        learning_system: Optional[IntegratedLearningSystem] = None,
        data_dir: str = "data/nextjs_learning"
    ):
        """Initialize the Next.js learning integration.
        
        Args:
            learning_system: Optional integrated learning system
            data_dir: Directory for storing learning data
        """
        self.learning_system = learning_system
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.categorizer = NextJSCategorizer()
        self.formatter = CursorRulesFormatter(self.categorizer)
        
        # Learning data storage
        self.events_file = self.data_dir / "learning_events.jsonl"
        self.metrics_file = self.data_dir / "learning_metrics.json"
        self.patterns_file = self.data_dir / "learned_patterns.json"
        
        # Load existing data
        self.learning_events = self._load_learning_events()
        self.learning_metrics = self._load_learning_metrics()
        self.learned_patterns = self._load_learned_patterns()
        
        # Learning configuration
        self.learning_config = {
            'min_events_for_learning': 10,
            'feedback_weight': 0.3,
            'usage_weight': 0.2,
            'quality_weight': 0.5,
            'learning_rate': 0.1,
            'retention_days': 90
        }
    
    def _load_learning_events(self) -> List[NextJSLearningEvent]:
        """Load learning events from storage."""
        events = []
        if self.events_file.exists():
            with open(self.events_file, 'r') as f:
                for line in f:
                    if line.strip():
                        event_data = json.loads(line)
                        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                        events.append(NextJSLearningEvent(**event_data))
        return events
    
    def _save_learning_events(self):
        """Save learning events to storage."""
        with open(self.events_file, 'w') as f:
            for event in self.learning_events:
                event_dict = asdict(event)
                event_dict['timestamp'] = event.timestamp.isoformat()
                f.write(json.dumps(event_dict) + '\n')
    
    def _load_learning_metrics(self) -> Optional[NextJSLearningMetrics]:
        """Load learning metrics from storage."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                return NextJSLearningMetrics(**data)
        return None
    
    def _save_learning_metrics(self, metrics: NextJSLearningMetrics):
        """Save learning metrics to storage."""
        metrics_dict = asdict(metrics)
        metrics_dict['last_updated'] = metrics.last_updated.isoformat()
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def _load_learned_patterns(self) -> Dict[str, Any]:
        """Load learned patterns from storage."""
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_learned_patterns(self):
        """Save learned patterns to storage."""
        with open(self.patterns_file, 'w') as f:
            json.dump(self.learned_patterns, f, indent=2)
    
    async def record_categorization_event(
        self,
        content: str,
        url: str,
        categories: Dict[str, Any],
        confidence: float
    ):
        """Record a categorization event for learning."""
        event = NextJSLearningEvent(
            event_type='categorization',
            category=max(categories.items(), key=lambda x: x[1].confidence)[0] if categories else 'unknown',
            confidence=confidence,
            content_hash=hash(content),
            url=url,
            timestamp=datetime.now(),
            metadata={
                'categories': {k: v.confidence for k, v in categories.items()} if categories else {},
                'content_length': len(content),
                'url_domain': url.split('/')[2] if '://' in url else 'unknown'
            }
        )
        
        self.learning_events.append(event)
        self._save_learning_events()
        
        logger.debug(f"Recorded categorization event: {event.category} (confidence: {confidence:.2f})")
    
    async def record_rule_generation_event(
        self,
        results: List[ScrapingResult],
        generated_rules: Dict[str, str],
        quality_score: float
    ):
        """Record a rule generation event for learning."""
        for result in results:
            event = NextJSLearningEvent(
                event_type='rule_generation',
                category='rule_generation',
                confidence=quality_score,
                content_hash=hash(result.content),
                url=result.url,
                timestamp=datetime.now(),
                metadata={
                    'rules_generated': len(generated_rules),
                    'content_length': len(result.content),
                    'quality_score': quality_score,
                    'rule_categories': list(generated_rules.keys())
                }
            )
            
            self.learning_events.append(event)
        
        self._save_learning_events()
        logger.debug(f"Recorded rule generation event: {len(generated_rules)} rules (quality: {quality_score:.2f})")
    
    async def record_user_feedback(
        self,
        content_hash: str,
        url: str,
        feedback: Dict[str, Any]
    ):
        """Record user feedback for learning."""
        event = NextJSLearningEvent(
            event_type='user_feedback',
            category=feedback.get('category', 'unknown'),
            confidence=feedback.get('satisfaction', 0.5),
            content_hash=content_hash,
            url=url,
            timestamp=datetime.now(),
            user_feedback=feedback
        )
        
        self.learning_events.append(event)
        self._save_learning_events()
        
        logger.info(f"Recorded user feedback: {feedback.get('satisfaction', 0.5):.2f} satisfaction")
    
    async def record_usage_event(
        self,
        rule_category: str,
        usage_context: Dict[str, Any]
    ):
        """Record usage event for learning."""
        event = NextJSLearningEvent(
            event_type='usage',
            category=rule_category,
            confidence=usage_context.get('effectiveness', 0.5),
            content_hash=hash(str(usage_context)),
            url=usage_context.get('url', ''),
            timestamp=datetime.now(),
            metadata=usage_context
        )
        
        self.learning_events.append(event)
        self._save_learning_events()
        
        logger.debug(f"Recorded usage event: {rule_category}")
    
    async def analyze_learning_patterns(self) -> Dict[str, Any]:
        """Analyze learning patterns from recorded events."""
        if not self.learning_events:
            return {'status': 'no_data', 'message': 'No learning events available'}
        
        # Filter recent events (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_events = [e for e in self.learning_events if e.timestamp > cutoff_date]
        
        if not recent_events:
            return {'status': 'no_recent_data', 'message': 'No recent learning events'}
        
        # Analyze patterns
        patterns = {
            'total_events': len(recent_events),
            'event_types': Counter(e.event_type for e in recent_events),
            'category_distribution': Counter(e.category for e in recent_events),
            'confidence_trends': self._analyze_confidence_trends(recent_events),
            'feedback_analysis': self._analyze_feedback_patterns(recent_events),
            'usage_patterns': self._analyze_usage_patterns(recent_events),
            'quality_improvements': self._analyze_quality_improvements(recent_events)
        }
        
        return patterns
    
    def _analyze_confidence_trends(self, events: List[NextJSLearningEvent]) -> Dict[str, Any]:
        """Analyze confidence trends over time."""
        # Group events by day
        daily_confidence = defaultdict(list)
        for event in events:
            day = event.timestamp.date()
            daily_confidence[day].append(event.confidence)
        
        # Calculate trends
        trends = {}
        for day, confidences in daily_confidence.items():
            trends[str(day)] = {
                'average': np.mean(confidences),
                'count': len(confidences),
                'std': np.std(confidences)
            }
        
        return trends
    
    def _analyze_feedback_patterns(self, events: List[NextJSLearningEvent]) -> Dict[str, Any]:
        """Analyze user feedback patterns."""
        feedback_events = [e for e in events if e.event_type == 'user_feedback']
        
        if not feedback_events:
            return {'status': 'no_feedback'}
        
        satisfaction_scores = [e.confidence for e in feedback_events]
        feedback_by_category = defaultdict(list)
        
        for event in feedback_events:
            feedback_by_category[event.category].append(event.confidence)
        
        return {
            'average_satisfaction': np.mean(satisfaction_scores),
            'satisfaction_by_category': {
                category: np.mean(scores) 
                for category, scores in feedback_by_category.items()
            },
            'total_feedback_events': len(feedback_events)
        }
    
    def _analyze_usage_patterns(self, events: List[NextJSLearningEvent]) -> Dict[str, Any]:
        """Analyze usage patterns."""
        usage_events = [e for e in events if e.event_type == 'usage']
        
        if not usage_events:
            return {'status': 'no_usage_data'}
        
        usage_by_category = Counter(e.category for e in usage_events)
        effectiveness_scores = [e.confidence for e in usage_events]
        
        return {
            'usage_by_category': dict(usage_by_category),
            'average_effectiveness': np.mean(effectiveness_scores),
            'total_usage_events': len(usage_events)
        }
    
    def _analyze_quality_improvements(self, events: List[NextJSLearningEvent]) -> Dict[str, Any]:
        """Analyze quality improvements over time."""
        rule_events = [e for e in events if e.event_type == 'rule_generation']
        
        if len(rule_events) < 2:
            return {'status': 'insufficient_data'}
        
        # Sort by timestamp
        rule_events.sort(key=lambda x: x.timestamp)
        
        # Calculate quality trend
        quality_scores = [e.confidence for e in rule_events]
        time_points = [(e.timestamp - rule_events[0].timestamp).days for e in rule_events]
        
        # Simple linear trend
        if len(quality_scores) > 1:
            trend = np.polyfit(time_points, quality_scores, 1)[0]
        else:
            trend = 0
        
        return {
            'quality_trend': trend,
            'initial_quality': quality_scores[0],
            'current_quality': quality_scores[-1],
            'improvement': quality_scores[-1] - quality_scores[0]
        }
    
    async def update_learning_models(self):
        """Update learning models based on recent events."""
        if not self.learning_events:
            logger.info("No learning events available for model updates")
            return
        
        # Filter recent events
        cutoff_date = datetime.now() - timedelta(days=self.learning_config['retention_days'])
        recent_events = [e for e in self.learning_events if e.timestamp > cutoff_date]
        
        if len(recent_events) < self.learning_config['min_events_for_learning']:
            logger.info(f"Insufficient events for learning: {len(recent_events)} < {self.learning_config['min_events_for_learning']}")
            return
        
        logger.info(f"Updating learning models with {len(recent_events)} events")
        
        # Update categorizer patterns
        await self._update_categorizer_patterns(recent_events)
        
        # Update formatter preferences
        await self._update_formatter_preferences(recent_events)
        
        # Update integrated learning system if available
        if self.learning_system:
            await self._update_integrated_learning_system(recent_events)
        
        # Clean up old events
        self._cleanup_old_events()
        
        logger.info("Learning models updated successfully")
    
    async def _update_categorizer_patterns(self, events: List[NextJSLearningEvent]):
        """Update categorizer patterns based on learning events."""
        # Group events by category
        category_events = defaultdict(list)
        for event in events:
            if event.event_type in ['categorization', 'user_feedback']:
                category_events[event.category].append(event)
        
        # Update patterns for each category
        for category, category_events_list in category_events.items():
            # Calculate average confidence
            avg_confidence = np.mean([e.confidence for e in category_events_list])
            
            # Extract successful patterns
            successful_events = [e for e in category_events_list if e.confidence > 0.7]
            
            if successful_events:
                # Update learned patterns
                if category not in self.learned_patterns:
                    self.learned_patterns[category] = []
                
                # Add new patterns from successful events
                for event in successful_events:
                    if event.metadata and 'content_length' in event.metadata:
                        pattern = {
                            'pattern': f"content_length_{event.metadata['content_length']}",
                            'confidence': event.confidence,
                            'source': 'learning_integration',
                            'timestamp': event.timestamp.isoformat()
                        }
                        
                        # Check if pattern already exists
                        existing = any(
                            p.get('pattern') == pattern['pattern'] 
                            for p in self.learned_patterns[category]
                        )
                        
                        if not existing:
                            self.learned_patterns[category].append(pattern)
        
        # Save updated patterns
        self._save_learned_patterns()
    
    async def _update_formatter_preferences(self, events: List[NextJSLearningEvent]):
        """Update formatter preferences based on learning events."""
        # Analyze user feedback for formatting preferences
        feedback_events = [e for e in events if e.event_type == 'user_feedback']
        
        if not feedback_events:
            return
        
        # Extract formatting preferences from feedback
        formatting_preferences = defaultdict(list)
        
        for event in feedback_events:
            if event.user_feedback:
                feedback = event.user_feedback
                
                # Extract preferences
                if 'format_preference' in feedback:
                    formatting_preferences['format'].append(feedback['format_preference'])
                
                if 'content_structure' in feedback:
                    formatting_preferences['structure'].append(feedback['content_structure'])
                
                if 'example_preference' in feedback:
                    formatting_preferences['examples'].append(feedback['example_preference'])
        
        # Update formatter configuration
        if formatting_preferences:
            # Calculate most preferred options
            preferred_format = Counter(formatting_preferences.get('format', [])).most_common(1)
            preferred_structure = Counter(formatting_preferences.get('structure', [])).most_common(1)
            preferred_examples = Counter(formatting_preferences.get('examples', [])).most_common(1)
            
            # Update formatter preferences (this would be implemented in the formatter)
            logger.info(f"Updated formatting preferences: {preferred_format}, {preferred_structure}, {preferred_examples}")
    
    async def _update_integrated_learning_system(self, events: List[NextJSLearningEvent]):
        """Update integrated learning system with new data."""
        if not self.learning_system:
            return
        
        # Convert events to learning system format
        learning_data = []
        for event in events:
            learning_data.append({
                'category': event.category,
                'confidence': event.confidence,
                'event_type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'metadata': event.metadata or {}
            })
        
        # Update learning system
        try:
            await self.learning_system.update_with_new_data(learning_data)
            logger.info("Updated integrated learning system")
        except Exception as e:
            logger.warning(f"Failed to update integrated learning system: {e}")
    
    def _cleanup_old_events(self):
        """Clean up old learning events."""
        cutoff_date = datetime.now() - timedelta(days=self.learning_config['retention_days'])
        original_count = len(self.learning_events)
        
        self.learning_events = [e for e in self.learning_events if e.timestamp > cutoff_date]
        
        removed_count = original_count - len(self.learning_events)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old learning events")
            self._save_learning_events()
    
    async def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report."""
        patterns = await self.analyze_learning_patterns()
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'learning_summary': {
                'total_events': len(self.learning_events),
                'recent_events': len([e for e in self.learning_events if e.timestamp > datetime.now() - timedelta(days=30)]),
                'learning_active': len(self.learning_events) >= self.learning_config['min_events_for_learning']
            },
            'patterns_analysis': patterns,
            'learned_patterns_summary': {
                'total_categories': len(self.learned_patterns),
                'total_patterns': sum(len(patterns) for patterns in self.learned_patterns.values()),
                'categories': list(self.learned_patterns.keys())
            },
            'recommendations': await self._generate_recommendations(patterns)
        }
        
        return report
    
    async def _generate_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on learning patterns."""
        recommendations = []
        
        # Analyze confidence trends
        if 'confidence_trends' in patterns:
            confidence_trends = patterns['confidence_trends']
            if confidence_trends:
                recent_confidence = list(confidence_trends.values())[-1]['average']
                if recent_confidence < 0.6:
                    recommendations.append("Consider improving categorization accuracy - recent confidence scores are low")
        
        # Analyze feedback patterns
        if 'feedback_analysis' in patterns and patterns['feedback_analysis'].get('status') != 'no_feedback':
            feedback_analysis = patterns['feedback_analysis']
            if feedback_analysis.get('average_satisfaction', 0) < 0.7:
                recommendations.append("User satisfaction is below target - review rule quality and formatting")
        
        # Analyze usage patterns
        if 'usage_patterns' in patterns and patterns['usage_patterns'].get('status') != 'no_usage_data':
            usage_patterns = patterns['usage_patterns']
            if usage_patterns.get('average_effectiveness', 0) < 0.6:
                recommendations.append("Rule effectiveness is low - consider updating patterns and examples")
        
        # Analyze quality improvements
        if 'quality_improvements' in patterns and patterns['quality_improvements'].get('status') != 'insufficient_data':
            quality_improvements = patterns['quality_improvements']
            if quality_improvements.get('trend', 0) < 0:
                recommendations.append("Quality trend is declining - review learning algorithms and data quality")
        
        if not recommendations:
            recommendations.append("Learning system is performing well - continue current approach")
        
        return recommendations
    
    async def get_learning_metrics(self) -> NextJSLearningMetrics:
        """Get current learning metrics."""
        if not self.learning_events:
            return NextJSLearningMetrics(
                total_events=0,
                categorization_accuracy=0.0,
                rule_quality_score=0.0,
                user_satisfaction=0.0,
                category_distribution={},
                improvement_trends={},
                last_updated=datetime.now()
            )
        
        # Calculate metrics
        total_events = len(self.learning_events)
        
        # Categorization accuracy (from categorization events)
        categorization_events = [e for e in self.learning_events if e.event_type == 'categorization']
        categorization_accuracy = np.mean([e.confidence for e in categorization_events]) if categorization_events else 0.0
        
        # Rule quality score (from rule generation events)
        rule_events = [e for e in self.learning_events if e.event_type == 'rule_generation']
        rule_quality_score = np.mean([e.confidence for e in rule_events]) if rule_events else 0.0
        
        # User satisfaction (from feedback events)
        feedback_events = [e for e in self.learning_events if e.event_type == 'user_feedback']
        user_satisfaction = np.mean([e.confidence for e in feedback_events]) if feedback_events else 0.0
        
        # Category distribution
        category_distribution = Counter(e.category for e in self.learning_events)
        
        # Improvement trends (simplified)
        improvement_trends = {}
        if len(self.learning_events) > 10:
            recent_events = self.learning_events[-10:]
            older_events = self.learning_events[-20:-10] if len(self.learning_events) > 20 else self.learning_events[:-10]
            
            if older_events:
                recent_avg = np.mean([e.confidence for e in recent_events])
                older_avg = np.mean([e.confidence for e in older_events])
                improvement_trends['confidence'] = recent_avg - older_avg
        
        metrics = NextJSLearningMetrics(
            total_events=total_events,
            categorization_accuracy=categorization_accuracy,
            rule_quality_score=rule_quality_score,
            user_satisfaction=user_satisfaction,
            category_distribution=dict(category_distribution),
            improvement_trends=improvement_trends,
            last_updated=datetime.now()
        )
        
        # Save metrics
        self._save_learning_metrics(metrics)
        
        return metrics
