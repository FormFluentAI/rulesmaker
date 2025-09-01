"""
ML-based content extractor using trained models.

Uses machine learning models to automatically extract and classify
content from documentation pages.
"""

import re
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

from bs4 import BeautifulSoup, Tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from .base import ContentExtractor
from ..models import (
    ContentSection, ExtractionPattern, LearningExample, 
    TrainingSet, DocumentationType
)

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class MLContentExtractor(ContentExtractor):
    """ML-based content extraction using trained models."""
    
    def __init__(
        self, 
        patterns: Optional[List[ExtractionPattern]] = None,
        model_path: Optional[str] = None,
        use_transformers: bool = True
    ):
        """Initialize the ML extractor."""
        super().__init__(patterns)
        
        self.model_path = model_path
        self.use_transformers = use_transformers
        
        # Initialize models
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.section_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.content_clusterer = KMeans(n_clusters=8, random_state=42)
        
        # Initialize transformer model for semantic understanding
        if self.use_transformers:
            try:
                self.sentence_transformer = SentenceTransformer(
                    'all-MiniLM-L6-v2'
                )
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.use_transformers = False
                self.sentence_transformer = None
        else:
            self.sentence_transformer = None
        
        # Training data and model state
        self.is_trained = False
        self.section_types = [
            'introduction', 'installation', 'quickstart', 'tutorial',
            'api_reference', 'examples', 'configuration', 'troubleshooting'
        ]
        
        # Load pre-trained model if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def extract(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured content from HTML."""
        try:
            # Extract basic information
            title = self._extract_title(soup)
            main_content = self._extract_main_content(soup)
            sections = self.extract_sections(soup, url)
            
            # Classify document type
            doc_type = self._classify_document_type(title, main_content, url)
            
            # Extract code examples
            code_examples = self._extract_code_examples(soup)
            
            # Extract navigation structure
            navigation = self._extract_navigation(soup)
            
            # Extract metadata
            metadata = self._extract_advanced_metadata(soup, url)
            
            return {
                'title': title,
                'content': main_content,
                'sections': [section.dict() for section in sections],
                'document_type': doc_type,
                'code_examples': code_examples,
                'navigation': navigation,
                'metadata': metadata,
                'confidence_score': self._calculate_confidence_score(sections)
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                'title': '',
                'content': '',
                'sections': [],
                'document_type': 'unknown',
                'code_examples': [],
                'navigation': {},
                'metadata': {},
                'confidence_score': 0.0
            }
    
    def extract_sections(self, soup: BeautifulSoup, url: str) -> List[ContentSection]:
        """Extract content sections using ML classification."""
        sections = []
        
        # Find all potential section elements
        potential_sections = self._find_potential_sections(soup)
        
        for element, content_text in potential_sections:
            # Extract section information
            title = self._extract_section_title(element)
            content = self._clean_content(content_text)
            level = self._determine_section_level(element)
            
            if not title or not content:
                continue
            
            # Classify section type
            section_type = self._classify_section_type(title, content)
            
            # Calculate confidence score
            confidence = self._calculate_section_confidence(element, title, content)
            
            # Create section
            section = ContentSection(
                title=title,
                content=content,
                level=level,
                url=url,
                metadata={
                    'section_type': section_type,
                    'confidence': confidence,
                    'html_tag': element.name if hasattr(element, 'name') else 'unknown',
                    'word_count': len(content.split()),
                    'has_code': self._has_code_content(content)
                }
            )
            
            sections.append(section)
        
        # Sort sections by confidence and position
        sections.sort(key=lambda s: (s.metadata.get('confidence', 0), s.level), reverse=True)
        
        return sections
    
    def train(self, training_set: TrainingSet) -> Dict[str, float]:
        """Train the ML models on provided examples."""
        if not training_set.examples:
            raise ValueError("Training set is empty")
        
        logger.info(f"Training ML extractor with {len(training_set.examples)} examples")
        
        # Prepare training data
        X_text = []
        X_features = []
        y_section_types = []
        
        for example in training_set.examples:
            soup = BeautifulSoup(example.input_html, 'html.parser')
            
            # Extract features for each section in the example
            sections = self._find_potential_sections(soup)
            
            for element, content_text in sections:
                # Text features
                X_text.append(content_text)
                
                # Structural features
                features = self._extract_structural_features(element, content_text)
                X_features.append(features)
                
                # Labels (if available in expected output)
                section_type = self._get_section_type_from_expected(
                    example.expected_output, content_text
                )
                y_section_types.append(section_type)
        
        if not X_text:
            raise ValueError("No training data extracted from examples")
        
        # Train TF-IDF vectorizer
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_text)
        
        # Combine TF-IDF with structural features
        X_combined = np.hstack([X_tfidf.toarray(), np.array(X_features)])
        
        # Train section classifier
        if len(set(y_section_types)) > 1:
            self.section_classifier.fit(X_combined, y_section_types)
        
        # Train content clusterer
        self.content_clusterer.fit(X_tfidf)
        
        # Train transformer embeddings if available
        if self.use_transformers and self.sentence_transformer:
            embeddings = self.sentence_transformer.encode(X_text)
            # Store embeddings for similarity comparison
            self.reference_embeddings = embeddings
            self.reference_texts = X_text
        
        self.is_trained = True
        
        # Evaluate performance
        performance = self._evaluate_model(X_combined, y_section_types)
        
        logger.info(f"Training completed. Performance: {performance}")
        return performance
    
    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'section_classifier': self.section_classifier,
            'content_clusterer': self.content_clusterer,
            'section_types': self.section_types,
            'is_trained': self.is_trained,
            'patterns': self.patterns
        }
        
        if self.use_transformers and hasattr(self, 'reference_embeddings'):
            model_data['reference_embeddings'] = self.reference_embeddings
            model_data['reference_texts'] = self.reference_texts
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.section_classifier = model_data['section_classifier']
            self.content_clusterer = model_data['content_clusterer']
            self.section_types = model_data['section_types']
            self.is_trained = model_data['is_trained']
            self.patterns = model_data.get('patterns', [])
            
            if 'reference_embeddings' in model_data:
                self.reference_embeddings = model_data['reference_embeddings']
                self.reference_texts = model_data['reference_texts']
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {str(e)}")
            raise
    
    def _find_potential_sections(self, soup: BeautifulSoup) -> List[Tuple[Tag, str]]:
        """Find potential content sections in HTML."""
        sections = []
        
        # Look for heading-based sections
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            content_parts = []
            current = heading.next_sibling
            
            # Collect content until next heading of same or higher level
            heading_level = int(heading.name[1])
            
            while current:
                if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    next_level = int(current.name[1])
                    if next_level <= heading_level:
                        break
                
                if hasattr(current, 'get_text'):
                    text = current.get_text().strip()
                    if text:
                        content_parts.append(text)
                
                current = current.next_sibling
            
            content_text = '\n'.join(content_parts)
            if content_text.strip():
                sections.append((heading, content_text))
        
        # Look for section-like containers
        containers = soup.find_all(['section', 'article', 'div'], class_=re.compile(r'(section|content|doc|tutorial|guide)'))
        
        for container in containers:
            content_text = container.get_text().strip()
            if len(content_text) > 50:  # Minimum content length
                sections.append((container, content_text))
        
        return sections
    
    def _extract_structural_features(self, element: Tag, content_text: str) -> List[float]:
        """Extract structural features from HTML element."""
        features = []
        
        # Text-based features
        features.append(len(content_text))  # Content length
        features.append(len(content_text.split()))  # Word count
        features.append(len(sent_tokenize(content_text)))  # Sentence count
        
        # HTML structure features
        features.append(1.0 if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] else 0.0)
        features.append(1.0 if element.find('code') else 0.0)  # Has code
        features.append(1.0 if element.find('pre') else 0.0)  # Has preformatted text
        features.append(len(element.find_all('a')))  # Number of links
        features.append(len(element.find_all(['ul', 'ol'])))  # Number of lists
        
        # CSS class features
        class_names = element.get('class', [])
        features.append(1.0 if any('example' in cls.lower() for cls in class_names) else 0.0)
        features.append(1.0 if any('api' in cls.lower() for cls in class_names) else 0.0)
        features.append(1.0 if any('tutorial' in cls.lower() for cls in class_names) else 0.0)
        
        # Position features
        parent = element.parent
        siblings = list(parent.children) if parent else []
        position = siblings.index(element) if element in siblings else 0
        features.append(position / max(len(siblings), 1))  # Relative position
        
        return features
    
    def _classify_section_type(self, title: str, content: str) -> str:
        """Classify the type of a content section."""
        if not self.is_trained:
            return self._rule_based_section_classification(title, content)
        
        try:
            # Prepare features
            text_features = self.tfidf_vectorizer.transform([content])
            
            # Create dummy structural features for classification
            dummy_features = [0.0] * 12  # Placeholder features
            combined_features = np.hstack([text_features.toarray(), [dummy_features]])
            
            # Predict section type
            predicted_type = self.section_classifier.predict(combined_features)[0]
            return predicted_type
            
        except Exception as e:
            logger.warning(f"ML classification failed, falling back to rules: {str(e)}")
            return self._rule_based_section_classification(title, content)
    
    def _rule_based_section_classification(self, title: str, content: str) -> str:
        """Fallback rule-based section classification."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Installation section
        if any(keyword in title_lower for keyword in ['install', 'setup', 'getting started']):
            return 'installation'
        
        # API reference
        if any(keyword in title_lower for keyword in ['api', 'reference', 'methods', 'functions']):
            return 'api_reference'
        
        # Examples
        if any(keyword in title_lower for keyword in ['example', 'sample', 'demo']):
            return 'examples'
        
        # Configuration
        if any(keyword in title_lower for keyword in ['config', 'setting', 'option']):
            return 'configuration'
        
        # Tutorial/Guide
        if any(keyword in title_lower for keyword in ['tutorial', 'guide', 'walkthrough', 'how to']):
            return 'tutorial'
        
        # Introduction
        if any(keyword in title_lower for keyword in ['intro', 'overview', 'about', 'what is']):
            return 'introduction'
        
        # Troubleshooting
        if any(keyword in title_lower for keyword in ['trouble', 'error', 'problem', 'issue', 'faq']):
            return 'troubleshooting'
        
        # Default based on content analysis
        if 'import' in content_lower or 'require' in content_lower:
            return 'installation'
        elif len(re.findall(r'```|<code>', content_lower)) > 2:
            return 'examples'
        else:
            return 'introduction'
    
    def _calculate_section_confidence(self, element: Tag, title: str, content: str) -> float:
        """Calculate confidence score for section extraction."""
        confidence = 0.0
        
        # Title quality
        if title and len(title) > 3:
            confidence += 0.2
        
        # Content quality
        word_count = len(content.split())
        if word_count > 10:
            confidence += 0.3
        if word_count > 50:
            confidence += 0.2
        
        # Structure quality
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            confidence += 0.2
        
        # Content indicators
        if element.find('code') or element.find('pre'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try different title sources
        if soup.title:
            return soup.title.get_text().strip()
        
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        return ""
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content text."""
        # Remove navigation, sidebar, footer elements
        for element in soup.find_all(['nav', 'sidebar', 'footer', 'header']):
            element.decompose()
        
        # Look for main content containers
        main_selectors = [
            'main', '[role="main"]', '.content', '.main-content',
            '.documentation', '.docs', 'article', '.article'
        ]
        
        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                return main_element.get_text().strip()
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text().strip()
        
        return soup.get_text().strip()
    
    def _extract_code_examples(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract code examples from the page."""
        examples = []
        
        # Find code blocks
        code_elements = soup.find_all(['pre', 'code'])
        
        for element in code_elements:
            code_text = element.get_text().strip()
            
            # Skip very short code snippets
            if len(code_text) < 10:
                continue
            
            # Detect language
            language = 'unknown'
            class_names = element.get('class', [])
            for cls in class_names:
                if cls.startswith('language-') or cls.startswith('lang-'):
                    language = cls.split('-', 1)[1]
                    break
            
            examples.append({
                'code': code_text,
                'language': language,
                'context': self._get_code_context(element)
            })
        
        return examples
    
    def _get_code_context(self, code_element: Tag) -> str:
        """Get context around a code element."""
        context_parts = []
        
        # Look for preceding text
        prev_sibling = code_element.previous_sibling
        while prev_sibling and len(context_parts) < 3:
            if hasattr(prev_sibling, 'get_text'):
                text = prev_sibling.get_text().strip()
                if text:
                    context_parts.insert(0, text)
            prev_sibling = prev_sibling.previous_sibling
        
        return ' '.join(context_parts)
    
    def _extract_navigation(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract navigation structure."""
        navigation = {}
        
        # Find navigation elements
        nav_elements = soup.find_all(['nav', '[role="navigation"]'])
        
        for nav in nav_elements:
            links = []
            for link in nav.find_all('a', href=True):
                links.append({
                    'text': link.get_text().strip(),
                    'href': link['href']
                })
            
            if links:
                nav_id = nav.get('id', nav.get('class', ['unknown'])[0])
                navigation[nav_id] = links
        
        return navigation
    
    def _extract_advanced_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract advanced metadata from the page."""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[f"meta_{name}"] = content
        
        # Structured data (JSON-LD)
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                import json
                structured_data = json.loads(script.string)
                metadata['structured_data'] = structured_data
            except:
                pass
        
        # Word count and reading time
        content_text = soup.get_text()
        word_count = len(content_text.split())
        metadata['word_count'] = word_count
        metadata['estimated_reading_time'] = max(1, word_count // 200)  # Assume 200 WPM
        
        return metadata
    
    def _classify_document_type(self, title: str, content: str, url: str) -> str:
        """Classify the type of documentation."""
        # Rule-based classification
        title_lower = title.lower()
        content_lower = content.lower()
        url_lower = url.lower()
        
        # API documentation
        if any(keyword in title_lower + url_lower for keyword in ['api', 'reference']):
            return DocumentationType.API.value
        
        # Tutorial
        if any(keyword in title_lower + url_lower for keyword in ['tutorial', 'guide', 'walkthrough']):
            return DocumentationType.TUTORIAL.value
        
        # Installation/Setup
        if any(keyword in title_lower + content_lower for keyword in ['install', 'setup', 'getting started']):
            return DocumentationType.GUIDE.value
        
        # README
        if 'readme' in url_lower or title_lower == 'readme':
            return DocumentationType.README.value
        
        return DocumentationType.UNKNOWN.value
    
    def _calculate_confidence_score(self, sections: List[ContentSection]) -> float:
        """Calculate overall confidence score for extraction."""
        if not sections:
            return 0.0
        
        section_confidences = [
            section.metadata.get('confidence', 0.0) 
            for section in sections
        ]
        
        return sum(section_confidences) / len(section_confidences)
    
    def _has_code_content(self, content: str) -> bool:
        """Check if content contains code snippets."""
        code_patterns = [
            r'```',  # Markdown code blocks
            r'`[^`]+`',  # Inline code
            r'import\s+\w+',  # Import statements
            r'function\s+\w+',  # Function definitions
            r'class\s+\w+',  # Class definitions
            r'<code>',  # HTML code tags
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content text."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters but keep punctuation
        content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'`]', '', content)
        
        return content.strip()
    
    def _extract_section_title(self, element: Tag) -> str:
        """Extract title from section element."""
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return element.get_text().strip()
        
        # Look for title in child elements
        title_element = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if title_element:
            return title_element.get_text().strip()
        
        # Use class or id as fallback
        title = element.get('title') or element.get('data-title')
        if title:
            return title
        
        return ""
    
    def _determine_section_level(self, element: Tag) -> int:
        """Determine hierarchical level of section."""
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return int(element.name[1])
        
        # Look for heading in children
        heading = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if heading:
            return int(heading.name[1])
        
        return 1  # Default level
    
    def _get_section_type_from_expected(
        self, 
        expected_output: Dict[str, Any], 
        content_text: str
    ) -> str:
        """Extract section type from expected output."""
        # This would need to be implemented based on your training data format
        # For now, return a default
        return 'introduction'
    
    def _evaluate_model(
        self, 
        X: np.ndarray, 
        y: List[str]
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if len(set(y)) < 2:
            return {'accuracy': 0.0}
        
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(self.section_classifier, X, y, cv=3)
            return {
                'accuracy': scores.mean(),
                'std': scores.std()
            }
        except Exception as e:
            logger.warning(f"Model evaluation failed: {str(e)}")
            return {'accuracy': 0.0}
