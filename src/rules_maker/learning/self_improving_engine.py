"""
Self-Improving ML Engine for Rules Maker

Implements a sophisticated feedback loop system that learns from rule usage,
performance metrics, and user feedback to continuously improve rule quality.
Features self-awarding mechanisms and adaptive quality thresholds.
"""

import logging
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from .models import (
    GeneratedRule, UsageEvent, UsageInsights, 
    RuleEffectiveness, QualityMetrics
)
from ..models import Rule, RuleType

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSignal:
    """Represents a feedback signal for rule quality assessment."""
    rule_id: str
    signal_type: str  # 'usage_success', 'user_rating', 'performance_metric', 'coherence_score'
    value: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"  # 'user', 'system', 'ml_model'
    confidence: float = 1.0


@dataclass
class QualityPrediction:
    """Prediction of rule quality with confidence intervals."""
    rule_id: str
    predicted_quality: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    prediction_timestamp: float


@dataclass
class ImprovementRecommendation:
    """Recommendation for improving rule quality."""
    rule_id: str
    recommendation_type: str
    description: str
    expected_improvement: float
    confidence: float
    implementation_priority: int


class SelfImprovingEngine:
    """ML-powered self-improving engine for rule quality optimization."""
    
    def __init__(
        self,
        feedback_window_hours: int = 168,  # 1 week
        min_feedback_signals: int = 5,
        quality_threshold: float = 0.7,
        learning_rate: float = 0.1,
        model_update_interval_hours: int = 24
    ):
        """Initialize the self-improving engine."""
        self.feedback_window_hours = feedback_window_hours
        self.min_feedback_signals = min_feedback_signals
        self.quality_threshold = quality_threshold
        self.learning_rate = learning_rate
        self.model_update_interval_hours = model_update_interval_hours
        
        # Storage for feedback signals and metrics
        self.feedback_signals: List[FeedbackSignal] = []
        self.quality_history: Dict[str, List[float]] = defaultdict(list)
        self.rule_features: Dict[str, Dict[str, float]] = {}
        
        # ML models for quality prediction and improvement
        self.quality_predictor: Optional[GradientBoostingRegressor] = None
        self.success_classifier: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = []
        self.last_model_update: float = 0
        
        # Adaptive thresholds
        self.adaptive_thresholds: Dict[str, float] = {
            'quality_threshold': quality_threshold,
            'success_threshold': 0.6,
            'improvement_threshold': 0.1
        }
        
        # Performance tracking
        self.model_performance: Dict[str, float] = {}
        self.improvement_history: List[Dict[str, Any]] = []
        
    async def collect_feedback_signal(
        self,
        rule_id: str,
        signal_type: str,
        value: float,
        context: Dict[str, Any] = None,
        source: str = "system",
        confidence: float = 1.0
    ):
        """Collect a feedback signal for a rule."""
        signal = FeedbackSignal(
            rule_id=rule_id,
            signal_type=signal_type,
            value=value,
            timestamp=time.time(),
            context=context or {},
            source=source,
            confidence=confidence
        )
        
        self.feedback_signals.append(signal)
        logger.debug(f"Collected {signal_type} feedback for rule {rule_id}: {value}")
        
        # Update quality history
        if signal_type in ['quality_score', 'user_rating', 'performance_metric']:
            self.quality_history[rule_id].append(value)
        
        # Trigger model update if needed
        if self._should_update_models():
            await self._update_ml_models()
    
    async def analyze_rule_performance(
        self,
        rules: List[GeneratedRule],
        time_window_hours: Optional[int] = None
    ) -> Dict[str, RuleEffectiveness]:
        """Analyze rule performance using collected feedback signals."""
        time_window = time_window_hours or self.feedback_window_hours
        cutoff_time = time.time() - (time_window * 3600)
        
        # Filter recent feedback signals
        recent_signals = [
            signal for signal in self.feedback_signals
            if signal.timestamp > cutoff_time
        ]
        
        # Group signals by rule ID
        signals_by_rule = defaultdict(list)
        for signal in recent_signals:
            signals_by_rule[signal.rule_id].append(signal)
        
        effectiveness_results = {}
        
        for rule in rules:
            rule_id = rule.rule.id
            rule_signals = signals_by_rule.get(rule_id, [])
            
            # Calculate effectiveness metrics
            effectiveness = await self._calculate_rule_effectiveness(
                rule, rule_signals
            )
            effectiveness_results[rule_id] = effectiveness
        
        return effectiveness_results
    
    async def predict_rule_quality(
        self,
        rule: Rule,
        features: Dict[str, float] = None
    ) -> QualityPrediction:
        """Predict the quality of a rule using ML models."""
        if features is None:
            features = self._extract_rule_features(rule)
        
        # Store features for future learning
        self.rule_features[rule.id] = features
        
        if self.quality_predictor is None:
            # No trained model yet, return heuristic prediction
            heuristic_quality = self._heuristic_quality_estimation(rule)
            return QualityPrediction(
                rule_id=rule.id,
                predicted_quality=heuristic_quality,
                confidence_interval=(heuristic_quality - 0.1, heuristic_quality + 0.1),
                feature_importance={},
                prediction_timestamp=time.time()
            )
        
        try:
            # Prepare feature vector
            feature_vector = self._features_to_vector(features)
            
            # Make prediction
            predicted_quality = self.quality_predictor.predict([feature_vector])[0]
            
            # Calculate confidence interval (simplified)
            prediction_std = 0.1  # Could be improved with actual model uncertainty
            confidence_interval = (
                max(0.0, predicted_quality - prediction_std),
                min(1.0, predicted_quality + prediction_std)
            )
            
            # Get feature importance
            feature_importance = {}
            if hasattr(self.quality_predictor, 'feature_importances_'):
                for i, importance in enumerate(self.quality_predictor.feature_importances_):
                    if i < len(self.feature_names):
                        feature_importance[self.feature_names[i]] = importance
            
            return QualityPrediction(
                rule_id=rule.id,
                predicted_quality=predicted_quality,
                confidence_interval=confidence_interval,
                feature_importance=feature_importance,
                prediction_timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Quality prediction failed for rule {rule.id}: {e}")
            # Fallback to heuristic
            heuristic_quality = self._heuristic_quality_estimation(rule)
            return QualityPrediction(
                rule_id=rule.id,
                predicted_quality=heuristic_quality,
                confidence_interval=(heuristic_quality - 0.2, heuristic_quality + 0.2),
                feature_importance={},
                prediction_timestamp=time.time()
            )
    
    async def generate_improvement_recommendations(
        self,
        rule: Rule,
        effectiveness: RuleEffectiveness,
        quality_prediction: QualityPrediction
    ) -> List[ImprovementRecommendation]:
        """Generate recommendations for improving a rule."""
        recommendations = []
        
        # Low quality recommendation
        if quality_prediction.predicted_quality < self.adaptive_thresholds['quality_threshold']:
            recommendations.append(
                ImprovementRecommendation(
                    rule_id=rule.id,
                    recommendation_type="quality_enhancement",
                    description=f"Rule quality ({quality_prediction.predicted_quality:.3f}) is below threshold. Consider adding more examples or clarifying instructions.",
                    expected_improvement=0.2,
                    confidence=0.8,
                    implementation_priority=1
                )
            )
        
        # Low success rate recommendation
        if effectiveness.success_rate < self.adaptive_thresholds['success_threshold']:
            recommendations.append(
                ImprovementRecommendation(
                    rule_id=rule.id,
                    recommendation_type="success_rate_improvement",
                    description=f"Success rate ({effectiveness.success_rate:.3f}) is low. Consider simplifying the rule or providing better context.",
                    expected_improvement=0.15,
                    confidence=0.7,
                    implementation_priority=2
                )
            )
        
        # Feature importance recommendations
        if quality_prediction.feature_importance:
            low_importance_features = [
                feature for feature, importance in quality_prediction.feature_importance.items()
                if importance < 0.1 and feature in ['has_examples', 'content_length', 'clarity_score']
            ]
            
            if low_importance_features:
                recommendations.append(
                    ImprovementRecommendation(
                        rule_id=rule.id,
                        recommendation_type="feature_enhancement",
                        description=f"Enhance features: {', '.join(low_importance_features)}",
                        expected_improvement=0.1,
                        confidence=0.6,
                        implementation_priority=3
                    )
                )
        
        # Sort by priority and expected improvement
        recommendations.sort(
            key=lambda r: (r.implementation_priority, -r.expected_improvement)
        )
        
        return recommendations
    
    async def self_award_quality_improvements(
        self,
        rules: List[GeneratedRule],
        batch_performance: Dict[str, Any]
    ) -> Dict[str, float]:
        """Self-award system that boosts quality scores for improving rules."""
        awards = {}
        
        # Calculate improvement metrics
        overall_improvement = batch_performance.get('improvement_score', 0.0)
        
        # Award bonus for rules that exceed expectations
        for rule in rules:
            rule_id = rule.rule.id
            current_quality = batch_performance.get('quality_scores', {}).get(rule_id, 0.5)
            
            # Check if rule has shown improvement over time
            quality_history = self.quality_history.get(rule_id, [])
            if len(quality_history) >= 2:
                recent_trend = np.polyfit(
                    range(len(quality_history)), 
                    quality_history, 
                    1
                )[0]  # Slope of trend line
                
                if recent_trend > 0.01:  # Positive trend
                    improvement_award = min(0.1, recent_trend * 2)
                    awards[rule_id] = improvement_award
                    logger.info(f"📈 Self-awarded rule {rule_id}: +{improvement_award:.3f} for improvement trend")
            
            # Award for exceeding predicted quality
            predicted_quality = batch_performance.get('predicted_qualities', {}).get(rule_id)
            if predicted_quality and current_quality > predicted_quality + 0.1:
                prediction_award = min(0.15, current_quality - predicted_quality)
                awards[rule_id] = awards.get(rule_id, 0) + prediction_award
                logger.info(f"🎯 Self-awarded rule {rule_id}: +{prediction_award:.3f} for exceeding prediction")
        
        # Global performance award
        if overall_improvement > 0.7:
            global_award = 0.05
            for rule in rules:
                awards[rule.rule.id] = awards.get(rule.rule.id, 0) + global_award
            logger.info(f"🌟 Global performance award: +{global_award:.3f} for all rules")
        
        return awards
    
    async def update_adaptive_thresholds(
        self,
        performance_metrics: Dict[str, float]
    ):
        """Update adaptive thresholds based on system performance."""
        # Adjust quality threshold based on overall performance
        overall_quality = performance_metrics.get('overall_coherence', 0.5)
        
        if overall_quality > 0.8:
            # Raise the bar when performance is high
            self.adaptive_thresholds['quality_threshold'] = min(
                0.9, self.adaptive_thresholds['quality_threshold'] + 0.05
            )
        elif overall_quality < 0.4:
            # Lower the bar when performance is low
            self.adaptive_thresholds['quality_threshold'] = max(
                0.3, self.adaptive_thresholds['quality_threshold'] - 0.05
            )
        
        # Adjust success threshold based on historical success rates
        avg_success_rate = performance_metrics.get('average_success_rate', 0.6)
        target_success_rate = avg_success_rate * 0.9  # Aim for 90% of average
        
        self.adaptive_thresholds['success_threshold'] = max(
            0.3, min(0.9, target_success_rate)
        )
        
        logger.info(f"📊 Updated adaptive thresholds: {self.adaptive_thresholds}")
    
    async def _update_ml_models(self):
        """Update ML models with new feedback data."""
        if len(self.feedback_signals) < self.min_feedback_signals:
            logger.debug("Insufficient feedback signals for model update")
            return
        
        try:
            # Prepare training data
            X, y_quality, y_success = await self._prepare_training_data()
            
            if len(X) < self.min_feedback_signals:
                return
            
            # Update quality predictor
            if len(y_quality) > 0:
                X_qual, X_qual_test, y_qual, y_qual_test = train_test_split(
                    X, y_quality, test_size=0.2, random_state=42
                )
                
                self.quality_predictor = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=self.learning_rate,
                    random_state=42
                )
                self.quality_predictor.fit(X_qual, y_qual)
                
                # Evaluate model
                if len(X_qual_test) > 0:
                    y_pred = self.quality_predictor.predict(X_qual_test)
                    mse = mean_squared_error(y_qual_test, y_pred)
                    self.model_performance['quality_predictor_mse'] = mse
                    logger.info(f"🤖 Updated quality predictor (MSE: {mse:.4f})")
            
            # Update success classifier
            if len(y_success) > 0:
                X_succ, X_succ_test, y_succ, y_succ_test = train_test_split(
                    X, y_success, test_size=0.2, random_state=42
                )
                
                self.success_classifier = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                self.success_classifier.fit(X_succ, y_succ)
                
                # Evaluate model
                if len(X_succ_test) > 0:
                    y_pred = self.success_classifier.predict(X_succ_test)
                    accuracy = accuracy_score(y_succ_test, y_pred)
                    self.model_performance['success_classifier_accuracy'] = accuracy
                    logger.info(f"🎯 Updated success classifier (Accuracy: {accuracy:.4f})")
            
            self.last_model_update = time.time()
            
        except Exception as e:
            logger.error(f"Failed to update ML models: {e}")
    
    async def _prepare_training_data(self) -> Tuple[np.ndarray, List[float], List[int]]:
        """Prepare training data from feedback signals."""
        # Group signals by rule ID
        signals_by_rule = defaultdict(list)
        for signal in self.feedback_signals:
            signals_by_rule[signal.rule_id].append(signal)
        
        X = []
        y_quality = []
        y_success = []
        feature_names = set()
        
        for rule_id, signals in signals_by_rule.items():
            if rule_id not in self.rule_features:
                continue
                
            features = self.rule_features[rule_id]
            feature_names.update(features.keys())
            
            # Extract quality and success labels from signals
            quality_signals = [
                s for s in signals 
                if s.signal_type in ['quality_score', 'user_rating']
            ]
            success_signals = [
                s for s in signals 
                if s.signal_type == 'usage_success'
            ]
            
            if quality_signals:
                avg_quality = np.mean([s.value for s in quality_signals])
                X.append(features)
                y_quality.append(avg_quality)
            
            if success_signals:
                success_rate = np.mean([s.value for s in success_signals])
                if len(X) == len(y_quality):  # Ensure X is aligned
                    y_success.append(1 if success_rate > 0.5 else 0)
                else:
                    X.append(features)
                    y_success.append(1 if success_rate > 0.5 else 0)
        
        # Ensure all feature dictionaries have same keys
        self.feature_names = sorted(feature_names)
        X_array = np.array([
            [features.get(fname, 0.0) for fname in self.feature_names]
            for features in X
        ])
        
        return X_array, y_quality, y_success
    
    def _extract_rule_features(self, rule: Rule) -> Dict[str, float]:
        """Extract numerical features from a rule for ML models."""
        features = {
            'content_length': len(rule.content) / 1000.0,  # Normalize
            'title_length': len(rule.title) / 100.0,
            'description_length': len(rule.description) / 500.0,
            'has_examples': float(len(rule.examples) > 0),
            'example_count': min(len(rule.examples) / 5.0, 1.0),
            'priority': rule.priority / 5.0,
            'confidence_score': rule.confidence_score,
            'tag_count': min(len(rule.tags) / 10.0, 1.0),
            'has_antipatterns': float(len(rule.anti_patterns) > 0),
            'rule_type_code': float(rule.type.value == "code_pattern"),
            'rule_type_best': float(rule.type.value == "best_practice"),
            'rule_type_error': float(rule.type.value == "error_handling"),
        }
        
        # Text complexity features
        if rule.content:
            word_count = len(rule.content.split())
            sentence_count = rule.content.count('.') + rule.content.count('!') + rule.content.count('?')
            features['avg_sentence_length'] = word_count / max(sentence_count, 1) / 20.0
            features['has_code_blocks'] = float('```' in rule.content)
            features['has_bullet_points'] = float('•' in rule.content or '-' in rule.content)
        
        return features
    
    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array."""
        return np.array([features.get(fname, 0.0) for fname in self.feature_names])
    
    def _heuristic_quality_estimation(self, rule: Rule) -> float:
        """Fallback heuristic quality estimation."""
        score = 0.5  # Base score
        
        # Content quality indicators
        if len(rule.content) > 100:
            score += 0.1
        if len(rule.examples) > 0:
            score += 0.15
        if rule.confidence_score > 0.7:
            score += 0.1
        if len(rule.tags) > 2:
            score += 0.05
        
        # Penalty for very short content
        if len(rule.content) < 50:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    async def _calculate_rule_effectiveness(
        self,
        rule: GeneratedRule,
        signals: List[FeedbackSignal]
    ) -> RuleEffectiveness:
        """Calculate rule effectiveness from feedback signals."""
        if not signals:
            return RuleEffectiveness(
                rule_id=rule.rule.id,
                title=rule.rule.title,
                usage_count=0,
                success_count=0,
                success_rate=0.0,
                avg_feedback=0.0,
                last_used_at=None,
                sections_effectiveness={}
            )
        
        usage_count = len([s for s in signals if s.signal_type == 'usage_success'])
        success_count = len([s for s in signals if s.signal_type == 'usage_success' and s.value > 0.5])
        success_rate = success_count / usage_count if usage_count > 0 else 0.0
        
        feedback_signals = [s for s in signals if s.signal_type in ['user_rating', 'quality_score']]
        avg_feedback = np.mean([s.value for s in feedback_signals]) if feedback_signals else 0.0
        
        last_used = max([s.timestamp for s in signals]) if signals else None
        last_used_dt = datetime.fromtimestamp(last_used) if last_used else None
        
        return RuleEffectiveness(
            rule_id=rule.rule.id,
            title=rule.rule.title,
            usage_count=usage_count,
            success_count=success_count,
            success_rate=success_rate,
            avg_feedback=avg_feedback,
            last_used_at=last_used_dt,
            sections_effectiveness={}
        )
    
    def _should_update_models(self) -> bool:
        """Determine if ML models should be updated."""
        time_since_update = time.time() - self.last_model_update
        hours_since_update = time_since_update / 3600
        
        return (
            hours_since_update >= self.model_update_interval_hours and
            len(self.feedback_signals) >= self.min_feedback_signals
        )
    
    async def save_state(self, filepath: str):
        """Save the engine state to disk."""
        state = {
            'feedback_signals': [
                {
                    'rule_id': s.rule_id,
                    'signal_type': s.signal_type,
                    'value': s.value,
                    'timestamp': s.timestamp,
                    'context': s.context,
                    'source': s.source,
                    'confidence': s.confidence
                }
                for s in self.feedback_signals
            ],
            'quality_history': dict(self.quality_history),
            'adaptive_thresholds': self.adaptive_thresholds,
            'model_performance': self.model_performance,
            'last_model_update': self.last_model_update
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"💾 Saved engine state to {filepath}")
    
    async def load_state(self, filepath: str):
        """Load engine state from disk."""
        if not Path(filepath).exists():
            logger.warning(f"State file {filepath} does not exist")
            return
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore feedback signals
            self.feedback_signals = [
                FeedbackSignal(
                    rule_id=s['rule_id'],
                    signal_type=s['signal_type'],
                    value=s['value'],
                    timestamp=s['timestamp'],
                    context=s.get('context', {}),
                    source=s.get('source', 'system'),
                    confidence=s.get('confidence', 1.0)
                )
                for s in state.get('feedback_signals', [])
            ]
            
            # Restore other state
            self.quality_history = defaultdict(list, state.get('quality_history', {}))
            self.adaptive_thresholds.update(state.get('adaptive_thresholds', {}))
            self.model_performance = state.get('model_performance', {})
            self.last_model_update = state.get('last_model_update', 0)
            
            logger.info(f"📥 Loaded engine state from {filepath}")
            
            # Retrain models with loaded data
            if self._should_update_models():
                await self._update_ml_models()
                
        except Exception as e:
            logger.error(f"Failed to load engine state: {e}")