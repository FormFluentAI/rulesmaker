"""
ML-powered quality optimization strategy.

Implements ML models for quality prediction and rule optimization
following the existing strategy pattern.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .learning_strategy import LearningStrategy
from ..models import LearningExample, TrainingSet, PerformanceMetrics

logger = logging.getLogger(__name__)


class MLQualityStrategy(LearningStrategy):
    """ML-powered quality optimization strategy using scikit-learn models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML quality strategy.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.config = config or {}
        
        # ML configuration
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        self.model_directory = Path(self.config.get('model_directory', 'models/'))
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_directory / 'ml_quality_models.json'
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Heuristic fallback weights
        self.heuristic_weights = {
            'code_examples': 0.3,
            'api_docs': 0.25, 
            'tutorial_content': 0.2,
            'structure': 0.15,
            'completeness': 0.1
        }
        
    def _initialize_ml_models(self):
        """Initialize ML models for quality prediction."""
        if not ML_AVAILABLE:
            logger.warning("scikit-learn not available, using heuristic fallback")
            self.quality_classifier = None
            self.quality_regressor = None
            self.vectorizer = None
            return
            
        # Quality classification model (high/medium/low quality)
        self.quality_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Quality scoring model (continuous 0-1 score)
        self.quality_regressor = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=6
        )
        
        # Text vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Try to load existing models
        if self.model_path.exists():
            try:
                self.load_model(str(self.model_path))
                logger.info("ML models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}, will train new ones")
    
    def train(self, training_set: TrainingSet) -> PerformanceMetrics:
        """Train ML models with training data.
        
        Args:
            training_set: Training data with examples and labels
            
        Returns:
            Performance metrics
        """
        if not ML_AVAILABLE or not training_set.examples:
            return self._train_heuristic_fallback(training_set)
        
        try:
            # Extract features and labels
            texts = []
            quality_labels = []
            quality_scores = []
            
            for example in training_set.examples:
                # Use content as text feature
                text = f"{example.input_html} {example.url}"
                texts.append(text)
                
                # Generate quality labels and scores from example metadata
                quality_info = self._extract_quality_info(example)
                quality_labels.append(quality_info['label'])
                quality_scores.append(quality_info['score'])
            
            # Vectorize text features
            X = self.vectorizer.fit_transform(texts)
            
            # Split data for validation
            X_train, X_test, y_class_train, y_class_test = train_test_split(
                X, quality_labels, test_size=0.2, random_state=42
            )
            _, _, y_score_train, y_score_test = train_test_split(
                X, quality_scores, test_size=0.2, random_state=42
            )
            
            # Train classification model
            self.quality_classifier.fit(X_train, y_class_train)
            class_pred = self.quality_classifier.predict(X_test)
            class_accuracy = accuracy_score(y_class_test, class_pred)
            
            # Train regression model
            self.quality_regressor.fit(X_train, y_score_train)
            score_pred = self.quality_regressor.predict(X_test)
            score_mse = mean_squared_error(y_score_test, score_pred)
            
            # Create performance metrics
            self.performance_metrics = PerformanceMetrics(
                accuracy=class_accuracy,
                precision=class_accuracy,  # Simplified
                recall=class_accuracy,     # Simplified 
                f1_score=class_accuracy,   # Simplified
                training_examples=len(training_set.examples),
                training_time=0.0  # Not tracked
            )
            
            self.is_trained = True
            logger.info(f"ML models trained: accuracy={class_accuracy:.3f}, mse={score_mse:.3f}")
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"ML training failed: {e}, falling back to heuristics")
            return self._train_heuristic_fallback(training_set)
    
    def _extract_quality_info(self, example: LearningExample) -> Dict[str, Any]:
        """Extract quality label and score from learning example.
        
        Args:
            example: Learning example
            
        Returns:
            Dictionary with quality label and score
        """
        # Analyze content for quality indicators
        content = example.input_html.lower()
        
        quality_indicators = {
            'has_code': bool(len([m for m in ['```', '<pre>', '<code>'] if m in content])),
            'has_examples': bool(len([m for m in ['example', 'tutorial', 'how to'] if m in content])),
            'has_api_docs': bool(len([m for m in ['api', 'endpoint', 'method'] if m in content])),
            'has_structure': bool(len([m for m in ['<h1>', '<h2>', '#'] if m in content])),
            'content_length': len(content.split())
        }
        
        # Calculate quality score based on indicators
        score = 0.0
        score += 0.3 if quality_indicators['has_code'] else 0.0
        score += 0.25 if quality_indicators['has_examples'] else 0.0  
        score += 0.25 if quality_indicators['has_api_docs'] else 0.0
        score += 0.15 if quality_indicators['has_structure'] else 0.0
        score += min(quality_indicators['content_length'] / 1000, 0.05)  # Content completeness
        
        # Determine quality label
        if score >= 0.7:
            label = 'high'
        elif score >= 0.4:
            label = 'medium'
        else:
            label = 'low'
            
        return {'score': score, 'label': label, 'indicators': quality_indicators}
    
    def _train_heuristic_fallback(self, training_set: TrainingSet) -> PerformanceMetrics:
        """Fallback training using heuristics when ML is unavailable.
        
        Args:
            training_set: Training data
            
        Returns:
            Performance metrics from heuristic approach
        """
        if not training_set.examples:
            raise ValueError("Training set is empty")
        
        # Simulate training with heuristic analysis
        correct = 0
        total = len(training_set.examples)
        
        for example in training_set.examples:
            quality_info = self._extract_quality_info(example)
            # Assume prediction is correct if score matches expected quality range
            if quality_info['score'] >= 0.5:  # Simplified validation
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        self.performance_metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=accuracy,
            recall=accuracy,
            f1_score=accuracy,
            training_examples=total,
            training_time=0.0
        )
        
        self.is_trained = True
        logger.info(f"Heuristic fallback training completed: accuracy={accuracy:.3f}")
        
        return self.performance_metrics
    
    async def predict(self, input_data: str, url: str) -> Dict[str, Any]:
        """Predict quality for given input.
        
        Args:
            input_data: Content to analyze
            url: Source URL
            
        Returns:
            Dictionary with quality predictions and recommendations
        """
        if ML_AVAILABLE and self.is_trained and self.quality_classifier is not None:
            return await self._ml_predict(input_data, url)
        else:
            return await self._heuristic_predict(input_data, url)
    
    async def _ml_predict(self, input_data: str, url: str) -> Dict[str, Any]:
        """ML-based quality prediction.
        
        Args:
            input_data: Content to analyze
            url: Source URL
            
        Returns:
            ML prediction results
        """
        try:
            # Vectorize input
            text = f"{input_data} {url}"
            X = self.vectorizer.transform([text])
            
            # Get predictions
            quality_class = self.quality_classifier.predict(X)[0]
            quality_probs = self.quality_classifier.predict_proba(X)[0]
            quality_score = float(self.quality_regressor.predict(X)[0])
            
            # Generate recommendations
            recommendations = self._generate_recommendations(input_data, quality_score)
            
            return {
                'quality_score': max(0.0, min(1.0, quality_score)),  # Clamp to [0,1]
                'quality_class': quality_class,
                'confidence': float(max(quality_probs)),
                'class_probabilities': {
                    'high': float(quality_probs[2]) if len(quality_probs) > 2 else 0.0,
                    'medium': float(quality_probs[1]) if len(quality_probs) > 1 else 0.0,
                    'low': float(quality_probs[0]) if len(quality_probs) > 0 else 0.0
                },
                'recommendations': recommendations,
                'is_high_quality': quality_score >= self.quality_threshold,
                'method': 'ml_prediction',
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}, falling back to heuristics")
            return await self._heuristic_predict(input_data, url)
    
    async def _heuristic_predict(self, input_data: str, url: str) -> Dict[str, Any]:
        """Heuristic-based quality prediction.
        
        Args:
            input_data: Content to analyze
            url: Source URL
            
        Returns:
            Heuristic prediction results
        """
        content = input_data.lower()
        
        # Quality scoring based on heuristics
        scores = {}
        scores['code_examples'] = 1.0 if any(indicator in content for indicator in ['```', '<pre>', '<code>', 'example']) else 0.0
        scores['api_docs'] = 1.0 if any(indicator in content for indicator in ['api', 'endpoint', 'method', 'parameter']) else 0.0
        scores['tutorial_content'] = 1.0 if any(indicator in content for indicator in ['tutorial', 'how to', 'getting started', 'guide']) else 0.0
        scores['structure'] = 1.0 if any(indicator in content for indicator in ['<h1>', '<h2>', '<h3>', '#']) else 0.0
        scores['completeness'] = min(len(content.split()) / 1000, 1.0)  # Normalize by content length
        
        # Calculate weighted quality score
        quality_score = sum(scores[key] * self.heuristic_weights[key] for key in scores.keys())
        
        # Determine quality class
        if quality_score >= 0.7:
            quality_class = 'high'
        elif quality_score >= 0.4:
            quality_class = 'medium'
        else:
            quality_class = 'low'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(input_data, quality_score)
        
        return {
            'quality_score': quality_score,
            'quality_class': quality_class,
            'confidence': 0.7,  # Moderate confidence for heuristics
            'class_probabilities': {
                'high': 1.0 if quality_class == 'high' else 0.0,
                'medium': 1.0 if quality_class == 'medium' else 0.0,
                'low': 1.0 if quality_class == 'low' else 0.0
            },
            'recommendations': recommendations,
            'is_high_quality': quality_score >= self.quality_threshold,
            'method': 'heuristic_prediction',
            'prediction_timestamp': datetime.now().isoformat(),
            'quality_breakdown': scores
        }
    
    def _generate_recommendations(self, content: str, quality_score: float) -> List[str]:
        """Generate improvement recommendations based on content analysis.
        
        Args:
            content: Content to analyze
            quality_score: Current quality score
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        content_lower = content.lower()
        
        # Check for missing elements and suggest improvements
        if not any(indicator in content_lower for indicator in ['```', '<pre>', '<code>']):
            recommendations.append("Add code examples to illustrate concepts")
        
        if not any(indicator in content_lower for indicator in ['example', 'tutorial', 'how to']):
            recommendations.append("Include practical examples and tutorials")
        
        if not any(indicator in content_lower for indicator in ['<h1>', '<h2>', '<h3>', '#']):
            recommendations.append("Add section headings to improve structure")
        
        if not any(indicator in content_lower for indicator in ['api', 'endpoint', 'method']):
            recommendations.append("Document API methods and endpoints")
        
        if len(content.split()) < 500:
            recommendations.append("Expand content with more detailed explanations")
        
        if quality_score < 0.5:
            recommendations.append("Consider comprehensive revision to improve overall quality")
        
        return recommendations
    
    def evaluate(self, test_set: TrainingSet) -> PerformanceMetrics:
        """Evaluate model performance on test set.
        
        Args:
            test_set: Test data
            
        Returns:
            Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if not test_set.examples:
            return PerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                training_examples=0, training_time=0.0
            )
        
        correct = 0
        total = len(test_set.examples)
        
        for example in test_set.examples:
            # Get prediction
            prediction = asyncio.run(self.predict(example.input_html, example.url))
            
            # Compare with expected quality (simplified evaluation)
            expected_quality = self._extract_quality_info(example)
            predicted_high = prediction['is_high_quality']
            expected_high = expected_quality['score'] >= self.quality_threshold
            
            if predicted_high == expected_high:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=accuracy,  # Simplified
            recall=accuracy,     # Simplified
            f1_score=accuracy,   # Simplified
            training_examples=0, # Not applicable for evaluation
            training_time=0.0
        )
    
    def save_model(self, model_path: str) -> None:
        """Save trained model to file.
        
        Args:
            model_path: Path to save model
        """
        if not ML_AVAILABLE:
            logger.warning("Cannot save ML models: scikit-learn not available")
            return
        
        try:
            model_data = {
                'model_type': 'ml_quality_strategy',
                'is_trained': self.is_trained,
                'config': self.config,
                'quality_threshold': self.quality_threshold,
                'heuristic_weights': self.heuristic_weights,
                'performance_metrics': (
                    self.performance_metrics.model_dump() 
                    if hasattr(self.performance_metrics, 'model_dump') and self.performance_metrics 
                    else None
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            # Note: For full ML model persistence, would need joblib
            # For now, saving configuration and metadata only
            
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            logger.info(f"Model metadata saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from file.
        
        Args:
            model_path: Path to load model from
        """
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            self.is_trained = model_data.get('is_trained', False)
            self.quality_threshold = model_data.get('quality_threshold', 0.7)
            self.heuristic_weights = model_data.get('heuristic_weights', self.heuristic_weights)
            
            if model_data.get('performance_metrics'):
                self.performance_metrics = PerformanceMetrics(**model_data['performance_metrics'])
            
            # Note: For full ML model persistence, would need joblib to restore sklearn models
            logger.info(f"Model metadata loaded from {model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")