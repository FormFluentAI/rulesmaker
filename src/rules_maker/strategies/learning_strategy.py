"""
Learning strategy for Rules Maker.
"""

from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import json
from pathlib import Path
from bs4 import BeautifulSoup

from ..models import LearningExample, TrainingSet, PerformanceMetrics, DocumentationType


class LearningStrategy(ABC):
    """Base class for learning strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the learning strategy.
        
        Args:
            config: Learning configuration
        """
        self.config = config or {}
        self.is_trained = False
        self.performance_metrics = None
    
    @abstractmethod
    def train(self, training_set: TrainingSet) -> PerformanceMetrics:
        """Train the learning model.
        
        Args:
            training_set: Training data
            
        Returns:
            Performance metrics
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: str, url: str) -> Dict[str, Any]:
        """Make predictions on new data.
        
        Args:
            input_data: Input HTML or text
            url: Source URL
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_set: TrainingSet) -> PerformanceMetrics:
        """Evaluate model performance.
        
        Args:
            test_set: Test data
            
        Returns:
            Performance metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """Save trained model.
        
        Args:
            model_path: Path to save model
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load trained model.
        
        Args:
            model_path: Path to load model from
        """
        pass


class BasicLearningStrategy(LearningStrategy):
    """Basic learning strategy using simple rules and patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize basic learning strategy."""
        super().__init__(config)
        self.patterns = {}
        self.vocabulary = set()
        self.doc_type_keywords = {
            DocumentationType.API: ['api', 'endpoint', 'rest', 'graphql', 'request', 'response'],
            DocumentationType.TUTORIAL: ['tutorial', 'guide', 'step', 'learn', 'how to'],
            DocumentationType.REFERENCE: ['reference', 'docs', 'documentation', 'spec'],
            DocumentationType.FRAMEWORK: ['framework', 'library', 'package', 'module'],
            DocumentationType.README: ['readme', 'getting started', 'installation'],
        }
    
    def train(self, training_set: TrainingSet) -> PerformanceMetrics:
        """Train using pattern recognition."""
        if not training_set.examples:
            raise ValueError("Training set is empty")
        
        # Extract patterns from training examples
        for example in training_set.examples:
            self._extract_patterns(example)
        
        # Calculate performance metrics
        correct_predictions = 0
        total_predictions = len(training_set.examples)
        
        for example in training_set.examples:
            prediction = self.predict(example.input_html, example.url)
            if prediction.get('documentation_type') == example.documentation_type:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        self.performance_metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=accuracy,  # Simplified for basic strategy
            recall=accuracy,
            f1_score=accuracy,
            training_examples=total_predictions,
            training_time=0.0  # Not tracked in basic strategy
        )
        
        self.is_trained = True
        return self.performance_metrics
    
    def predict(self, input_data: str, url: str) -> Dict[str, Any]:
        """Make predictions using pattern matching."""
        from ..utils import detect_documentation_type
        from bs4 import BeautifulSoup
        
        # Basic prediction using existing logic
        soup = BeautifulSoup(input_data, 'html.parser')
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else ""
        
        # Use utility function as baseline
        doc_type = detect_documentation_type(url, title_text, input_data)
        
        # Calculate confidence based on keyword matches
        confidence = self._calculate_confidence(input_data, url, doc_type)
        
        return {
            'documentation_type': doc_type,
            'confidence_score': confidence,
            'predicted_sections': self._predict_sections(soup),
            'metadata': {
                'prediction_method': 'pattern_matching',
                'title': title_text,
                'url': url
            }
        }
    
    def evaluate(self, test_set: TrainingSet) -> PerformanceMetrics:
        """Evaluate model on test set."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        correct_predictions = 0
        total_predictions = len(test_set.examples)
        
        for example in test_set.examples:
            prediction = self.predict(example.input_html, example.url)
            if prediction.get('documentation_type') == example.documentation_type:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=accuracy,
            recall=accuracy,
            f1_score=accuracy,
            training_examples=0,  # Not applicable for evaluation
            training_time=0.0
        )
    
    def save_model(self, model_path: str) -> None:
        """Save model to file."""
        model_data = {
            'patterns': self.patterns,
            'vocabulary': list(self.vocabulary),
            'doc_type_keywords': {k.value: v for k, v in self.doc_type_keywords.items()},
            'is_trained': self.is_trained,
            'performance_metrics': (self.performance_metrics.model_dump() if hasattr(self.performance_metrics, 'model_dump') else self.performance_metrics.dict()) if self.performance_metrics else None
        }
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, model_path: str) -> None:
        """Load model from file."""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        self.patterns = model_data.get('patterns', {})
        self.vocabulary = set(model_data.get('vocabulary', []))
        
        # Convert doc type keywords back to enum keys
        doc_type_keywords = model_data.get('doc_type_keywords', {})
        self.doc_type_keywords = {
            DocumentationType(k): v for k, v in doc_type_keywords.items()
        }
        
        self.is_trained = model_data.get('is_trained', False)
        
        if model_data.get('performance_metrics'):
            self.performance_metrics = PerformanceMetrics(**model_data['performance_metrics'])
    
    def _extract_patterns(self, example: LearningExample) -> None:
        """Extract patterns from a training example."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(example.input_html, 'html.parser')
        text = soup.get_text().lower()
        
        # Extract words for vocabulary
        words = text.split()
        self.vocabulary.update(words)
        
        # Extract patterns for document type
        doc_type = example.documentation_type
        if doc_type not in self.patterns:
            self.patterns[doc_type] = {
                'common_words': {},
                'url_patterns': [],
                'title_patterns': []
            }
        
        # Count word frequencies
        for word in words:
            if word not in self.patterns[doc_type]['common_words']:
                self.patterns[doc_type]['common_words'][word] = 0
            self.patterns[doc_type]['common_words'][word] += 1
        
        # Extract URL patterns
        url_parts = example.url.lower().split('/')
        self.patterns[doc_type]['url_patterns'].extend(url_parts)
        
        # Extract title patterns
        title = soup.find('title')
        if title:
            title_words = title.get_text().lower().split()
            self.patterns[doc_type]['title_patterns'].extend(title_words)
    
    def _calculate_confidence(self, content: str, url: str, doc_type: DocumentationType) -> float:
        """Calculate confidence score for prediction."""
        confidence = 0.5  # Base confidence
        
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Check for keyword matches
        if doc_type in self.doc_type_keywords:
            keywords = self.doc_type_keywords[doc_type]
            matches = sum(1 for keyword in keywords if keyword in content_lower or keyword in url_lower)
            keyword_confidence = min(matches * 0.1, 0.3)  # Max 0.3 from keywords
            confidence += keyword_confidence
        
        # Check for pattern matches if trained
        if self.is_trained and doc_type in self.patterns:
            pattern_confidence = self._check_pattern_match(content_lower, url_lower, doc_type)
            confidence += pattern_confidence
        
        return min(confidence, 1.0)
    
    def _check_pattern_match(self, content: str, url: str, doc_type: DocumentationType) -> float:
        """Check how well content matches learned patterns."""
        if doc_type not in self.patterns:
            return 0.0
        
        patterns = self.patterns[doc_type]
        match_score = 0.0
        
        # Check common words
        content_words = content.split()
        common_words = patterns.get('common_words', {})
        
        if common_words:
            word_matches = sum(1 for word in content_words if word in common_words)
            word_score = min(word_matches / len(content_words), 0.2)
            match_score += word_score
        
        # Check URL patterns
        url_patterns = patterns.get('url_patterns', [])
        if url_patterns:
            url_matches = sum(1 for pattern in url_patterns if pattern in url)
            url_score = min(url_matches * 0.05, 0.1)
            match_score += url_score
        
        return match_score
    
    def _predict_sections(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Predict likely sections in the document."""
        sections = []
        
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            title = heading.get_text(strip=True)
            if title:
                sections.append({
                    'title': title,
                    'level': int(heading.name[1]),
                    'type': self._classify_section_type(title)
                })
        
        return sections
    
    def _classify_section_type(self, title: str) -> str:
        """Classify section type based on title."""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['install', 'setup', 'getting started']):
            return 'installation'
        elif any(word in title_lower for word in ['example', 'usage', 'tutorial']):
            return 'example'
        elif any(word in title_lower for word in ['api', 'reference', 'docs']):
            return 'reference'
        elif any(word in title_lower for word in ['config', 'configuration', 'settings']):
            return 'configuration'
        else:
            return 'general'
