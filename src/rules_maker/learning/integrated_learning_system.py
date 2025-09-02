"""
Integrated learning system that combines existing and ML capabilities.

Extends existing learning with ML capabilities as outlined in the integration guide,
building on existing LearningEngine and SelfImprovingEngine components.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .engine import LearningEngine
from .pattern_analyzer import SemanticAnalyzer
from .models import GeneratedRule, UsageEvent, UsageInsights, OptimizedRules
from ..models import Rule
from ..strategies.ml_quality_strategy import MLQualityStrategy

logger = logging.getLogger(__name__)


class IntegratedLearningSystem:
    """Integrated learning system combining existing and ML capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize integrated learning system.
        
        Args:
            config: System configuration
        """
        self.config = config or {}
        
        # Initialize existing components
        self.base_engine = LearningEngine(self.config.get('base_engine_config', {}))
        
        # Initialize ML components (optional, avoid hard dep on numpy/sklearn)
        try:
            from .self_improving_engine import SelfImprovingEngine  # type: ignore
            self.ml_engine = SelfImprovingEngine(
                quality_threshold=self.config.get('quality_threshold', 0.7)
            )
        except Exception as e:
            logger.warning(f"SelfImprovingEngine unavailable ({e}); ML engine disabled")
            self.ml_engine = None
        
        # Initialize strategy and analyzer
        self.ml_strategy = MLQualityStrategy(self.config.get('ml_strategy_config', {}))
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Configuration
        self.enable_ml = self.config.get('enable_ml', True)
        self.ml_weight = self.config.get('ml_weight', 0.6)  # Weight for ML vs base predictions
        self.feedback_integration = self.config.get('feedback_integration', True)
        
        logger.info("Initialized integrated learning system")
    
    async def learn_and_improve(self, rules: List[GeneratedRule], 
                               usage_data: List[UsageEvent]) -> OptimizedRules:
        """Combined learning and improvement using both existing and ML capabilities.
        
        Args:
            rules: Generated rules with usage history
            usage_data: Usage event data
            
        Returns:
            Optimized rules with combined insights
        """
        logger.info(f"Starting integrated learning for {len(rules)} rules with {len(usage_data)} usage events")
        
        # Phase 1: Use existing pattern analysis
        base_insights = self.base_engine.analyze_usage_patterns(rules)
        logger.debug(f"Base analysis: {base_insights.total_events} events, {base_insights.global_success_rate:.3f} success rate")
        
        # Phase 2: Add ML-powered quality prediction
        ml_predictions = {}
        if self.enable_ml:
            ml_predictions = await self._generate_ml_predictions(rules)
            logger.debug(f"Generated ML predictions for {len(ml_predictions)} rules")
        
        # Phase 3: Combine insights for optimization
        combined_optimization = await self._combine_insights_for_optimization(
            base_insights, ml_predictions, usage_data
        )
        
        # Phase 4: Apply integrated learning strategies
        final_optimization = await self._apply_integrated_strategies(
            combined_optimization, rules, usage_data
        )
        
        logger.info(f"Completed integrated learning with quality score: {final_optimization.quality_score:.3f}")
        
        return final_optimization
    
    async def _generate_ml_predictions(self, rules: List[GeneratedRule]) -> Dict[str, Dict[str, Any]]:
        """Generate ML predictions for rule quality.
        
        Args:
            rules: Rules to analyze
            
        Returns:
            Dictionary mapping rule_id to ML predictions
        """
        predictions = {}
        
        for rule in rules:
            try:
                # Prepare content for analysis
                rule_content = f"{rule.rule.title} {rule.rule.description}"
                if rule.rule.examples:
                    rule_content += " " + " ".join(rule.rule.examples)
                
                # Get ML prediction
                prediction = await self.ml_strategy.predict(
                    rule_content, 
                    getattr(rule.rule, 'source_url', '')
                )
                
                predictions[rule.rule.id] = prediction
                
            except Exception as e:
                logger.warning(f"ML prediction failed for rule {rule.rule.id}: {e}")
                # Provide fallback prediction
                predictions[rule.rule.id] = {
                    'quality_score': 0.5,
                    'confidence': 0.3,
                    'is_high_quality': False,
                    'method': 'fallback'
                }
        
        return predictions
    
    async def _combine_insights_for_optimization(
        self, 
        base_insights: UsageInsights,
        ml_predictions: Dict[str, Dict[str, Any]],
        usage_data: List[UsageEvent]
    ) -> OptimizedRules:
        """Combine base and ML insights for optimization.
        
        Args:
            base_insights: Base learning engine insights
            ml_predictions: ML quality predictions
            usage_data: Usage event data
            
        Returns:
            Combined optimization results
        """
        # Start with base optimization
        rule_map = {}
        for rule_id, effectiveness in base_insights.by_rule.items():
            # Create a mock rule for optimization (in real implementation, 
            # you'd pass the actual rule objects)
            mock_rule = Rule(
                id=rule_id,
                title=effectiveness.title,
                description="",
                examples=[],
                priority=3,
                tags=[],
                rule_type="general",
                confidence_score=0.5,
                metadata={}
            )
            rule_map[rule_id] = mock_rule
        
        # Apply base optimization
        self.base_engine.config['rule_map'] = rule_map
        base_optimization = self.base_engine.optimize_rules(base_insights)
        
        # Enhance with ML predictions
        enhanced_rules = []
        enhanced_changes = base_optimization.changes.copy()
        
        for rule in base_optimization.rules:
            ml_pred = ml_predictions.get(rule.id, {})
            
            if ml_pred and self.enable_ml:
                # Combine base and ML quality scores
                base_quality = rule.confidence_score
                ml_quality = ml_pred.get('quality_score', 0.5)
                combined_quality = (
                    (1 - self.ml_weight) * base_quality + 
                    self.ml_weight * ml_quality
                )
                
                # Adjust priority based on ML predictions
                ml_priority_adjustment = 0
                if ml_pred.get('is_high_quality', False):
                    ml_priority_adjustment = 1
                elif ml_pred.get('quality_score', 0.5) < 0.4:
                    ml_priority_adjustment = -1
                
                adjusted_priority = max(1, min(5, rule.priority + ml_priority_adjustment))
                
                # Update rule metadata with ML insights
                enhanced_metadata = rule.metadata.copy()
                enhanced_metadata.update({
                    'ml_quality_score': ml_quality,
                    'ml_confidence': ml_pred.get('confidence', 0.0),
                    'combined_quality': combined_quality,
                    'ml_enhanced': True,
                    'integration_timestamp': datetime.now().isoformat()
                })
                
                # Create enhanced rule
                enhanced_rule = rule.model_copy(update={
                    'priority': adjusted_priority,
                    'confidence_score': combined_quality,
                    'metadata': enhanced_metadata
                })
                
                enhanced_rules.append(enhanced_rule)
                
                # Record the ML enhancement
                enhanced_changes.append({
                    'rule_id': rule.id,
                    'change_type': 'ml_integration_enhancement',
                    'description': f'ML enhancement: quality={ml_quality:.3f}, combined={combined_quality:.3f}',
                    'before': {'confidence': base_quality, 'priority': rule.priority - ml_priority_adjustment},
                    'after': {'confidence': combined_quality, 'priority': adjusted_priority},
                    'ml_prediction': ml_pred
                })
            else:
                enhanced_rules.append(rule)
        
        # Calculate combined quality score
        if enhanced_rules:
            combined_quality_score = sum(r.confidence_score for r in enhanced_rules) / len(enhanced_rules)
        else:
            combined_quality_score = base_optimization.quality_score
        
        return OptimizedRules(
            rules=enhanced_rules,
            changes=enhanced_changes,
            quality_score=combined_quality_score
        )
    
    async def _apply_integrated_strategies(
        self,
        combined_optimization: OptimizedRules,
        original_rules: List[GeneratedRule],
        usage_data: List[UsageEvent]
    ) -> OptimizedRules:
        """Apply integrated learning strategies for final optimization.
        
        Args:
            combined_optimization: Combined optimization results
            original_rules: Original generated rules
            usage_data: Usage event data
            
        Returns:
            Final optimized rules
        """
        # Apply self-improving engine if enabled
        if self.enable_ml and self.feedback_integration:
            try:
                # Collect feedback signals from usage data
                await self._collect_integrated_feedback(usage_data)
                
                # Apply self-improvement mechanisms
                improved_rules = []
                for rule in combined_optimization.rules:
                    # Use ML engine for self-improvement
                    rule_content = f"{rule.title} {rule.description}"
                    
                    # Check if rule should be improved based on usage patterns
                    usage_events_for_rule = [
                        event for event in usage_data 
                        if event.context.get('rule_id') == rule.id
                    ]
                    
                    if len(usage_events_for_rule) >= 3:  # Enough data for improvement
                        try:
                            improvement = await self.ml_engine.predict_rule_quality(rule_content)
                            if improvement and improvement.get('suggested_improvements'):
                                # Apply improvements to rule metadata
                                improved_metadata = rule.metadata.copy()
                                improved_metadata.update({
                                    'self_improvement': improvement,
                                    'improvement_applied': True
                                })
                                
                                improved_rule = rule.model_copy(update={'metadata': improved_metadata})
                                improved_rules.append(improved_rule)
                                
                                # Record improvement
                                combined_optimization.changes.append({
                                    'rule_id': rule.id,
                                    'change_type': 'self_improvement',
                                    'description': f'Self-improvement applied based on {len(usage_events_for_rule)} usage events',
                                    'improvement': improvement
                                })
                            else:
                                improved_rules.append(rule)
                        except Exception as e:
                            logger.warning(f"Self-improvement failed for rule {rule.id}: {e}")
                            improved_rules.append(rule)
                    else:
                        improved_rules.append(rule)
                
                combined_optimization.rules = improved_rules
                
            except Exception as e:
                logger.warning(f"Integrated strategy application failed: {e}")
        
        # Final quality score calculation
        if combined_optimization.rules:
            final_quality = sum(r.confidence_score for r in combined_optimization.rules) / len(combined_optimization.rules)
            combined_optimization.quality_score = final_quality
        
        # Add system metadata
        system_metadata = {
            'integrated_learning': True,
            'ml_enabled': self.enable_ml,
            'ml_weight': self.ml_weight,
            'feedback_integration': self.feedback_integration,
            'optimization_timestamp': datetime.now().isoformat(),
            'rules_processed': len(combined_optimization.rules),
            'changes_applied': len(combined_optimization.changes)
        }
        
        # Add metadata to first rule or create summary
        if combined_optimization.rules:
            first_rule_metadata = combined_optimization.rules[0].metadata.copy()
            first_rule_metadata['system_integration'] = system_metadata
            combined_optimization.rules[0] = combined_optimization.rules[0].model_copy(
                update={'metadata': first_rule_metadata}
            )
        
        return combined_optimization
    
    async def _collect_integrated_feedback(self, usage_data: List[UsageEvent]) -> None:
        """Collect and process feedback from usage data for ML improvement.
        
        Args:
            usage_data: Usage event data
        """
        for event in usage_data:
            try:
                # Convert usage event to feedback signal
                rule_id = event.context.get('rule_id', f"unknown_{event.timestamp}")
                signal_type = "usage_success" if event.success else "usage_failure"
                value = event.feedback_score if event.feedback_score is not None else (0.8 if event.success else 0.2)
                
                # Collect feedback signal for ML engine
                await self.ml_engine.collect_feedback_signal(
                    rule_id=rule_id,
                    signal_type=signal_type,
                    value=value,
                    source="integrated_learning",
                    metadata={
                        'event_timestamp': event.timestamp.isoformat() if event.timestamp else None,
                        'context': event.context
                    }
                )
                
            except Exception as e:
                logger.warning(f"Failed to collect feedback from usage event: {e}")
    
    async def get_system_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics for the integrated system.
        
        Returns:
            System performance statistics
        """
        stats = {
            'system_type': 'integrated_learning',
            'components': {
                'base_engine': 'active',
                'ml_engine': 'active' if self.enable_ml else 'disabled',
                'ml_strategy': 'trained' if self.ml_strategy.is_trained else 'untrained'
            },
            'configuration': {
                'ml_enabled': self.enable_ml,
                'ml_weight': self.ml_weight,
                'feedback_integration': self.feedback_integration,
                'quality_threshold': self.config.get('quality_threshold', 0.7)
            }
        }
        
        # Add ML engine stats if available
        if self.enable_ml:
            try:
                ml_stats = await self.ml_engine.get_performance_stats()
                stats['ml_performance'] = ml_stats
            except Exception as e:
                logger.warning(f"Failed to get ML performance stats: {e}")
                stats['ml_performance'] = {'error': str(e)}
        
        # Add base engine performance if available
        if hasattr(self.base_engine, 'performance_metrics') and self.base_engine.performance_metrics:
            stats['base_performance'] = {
                'success_threshold': self.base_engine.success_threshold,
                'min_events': self.base_engine.min_events,
                'priority_boost': self.base_engine.priority_boost
            }
        
        return stats
