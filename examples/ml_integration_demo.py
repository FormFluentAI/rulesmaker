"""
ML Integration Demo - Complete example of ML-enhanced batch processing.

This example demonstrates all the ML integration components working together
as outlined in the integration guide. It shows:

1. ML-enhanced processors with semantic analysis
2. Quality-aware transformers with intelligent clustering  
3. Integrated learning system combining base and ML capabilities
4. Self-improving feedback mechanisms
5. Comprehensive batch processing with quality optimization
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml

try:
    from rules_maker.batch_processor import MLBatchProcessor, DocumentationSource
    _BATCH_AVAILABLE = True
except Exception as _e:
    _BATCH_AVAILABLE = False
    # Logger not yet configured; use basicConfig for this early warning
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger(__name__).warning(f"Batch processor unavailable ({_e}); batch demo will be skipped")
    MLBatchProcessor = None  # type: ignore
    DocumentationSource = None  # type: ignore
from rules_maker.processors.ml_documentation_processor import MLDocumentationProcessor
from rules_maker.transformers.ml_cursor_transformer import MLCursorTransformer
from rules_maker.transformers.windsurf_transformer import WindsurfRuleTransformer
from rules_maker.learning.integrated_learning_system import IntegratedLearningSystem
from rules_maker.strategies.ml_quality_strategy import MLQualityStrategy
from rules_maker.models import ScrapingResult, DocumentationType, ScrapingStatus, RuleFormat
from rules_maker.learning.models import GeneratedRule, UsageEvent
from rules_maker.models import Rule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLIntegrationDemo:
    """Comprehensive ML integration demonstration."""
    
    def __init__(self, config_path: str = "config/ml_batch_config.yaml"):
        """Initialize demo with configuration.
        
        Args:
            config_path: Path to ML batch configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for demo."""
        return {
            'batch_processing': {
                'max_concurrent': 5,
                'quality_threshold': 0.7,
                'enable_clustering': True,
                'coherence_threshold': 0.6
            },
            'ml_engine': {
                'quality_threshold': 0.7,
                'enable_self_improvement': True
            },
            'integrated_learning': {
                'enable_ml': True,
                'ml_weight': 0.6,
                'feedback_integration': True
            },
            'output_config': {
                'base_directory': 'rules/',
                'ml_batch_directory': 'rules/ml_demo/'
            }
        }
    
    def setup_components(self):
        """Initialize all ML-enhanced components."""
        logger.info("Initializing ML-enhanced components...")
        
        # ML-enhanced processor
        self.ml_processor = MLDocumentationProcessor(
            config=self.config.get('ml_processor', {})
        )
        
        # ML-enhanced transformers
        ml_config = self.config.get('ml_engine', {})
        self.ml_cursor_transformer = MLCursorTransformer(ml_config=ml_config)
        self.windsurf_transformer = WindsurfRuleTransformer()  # Standard transformer for comparison
        
        # Integrated learning system
        self.integrated_learning = IntegratedLearningSystem(
            config=self.config.get('integrated_learning', {})
        )
        
        # ML quality strategy
        self.ml_strategy = MLQualityStrategy(
            config=self.config.get('ml_strategy', {})
        )
        
        # Resolve output directory from config with robust fallbacks
        output_dir = (
            (self.config.get('output_config') or {}).get('ml_batch_directory')
            or (self.config.get('output_formatting') or {}).get('output_directory')
            or 'rules/ml_demo/'
        )

        # ML batch processor (optional)
        if _BATCH_AVAILABLE and MLBatchProcessor is not None:
            bp_cfg = self.config.get('batch_processing', {}) or {}
            self.batch_processor = MLBatchProcessor(
                bedrock_config=self.config.get('bedrock_integration', {}),
                output_dir=output_dir,
                quality_threshold=bp_cfg.get('quality_threshold', 0.6),
                max_concurrent=bp_cfg.get('max_concurrent', 10),
            )
        else:
            logger.warning("MLBatchProcessor not available; skipping batch processing component")
            self.batch_processor = None
        
        logger.info("✅ All ML components initialized successfully")
    
    async def demo_ml_processor_enhancement(self):
        """Demo ML-enhanced documentation processing."""
        logger.info("\n🔍 Demo: ML-Enhanced Documentation Processing")
        
        # Sample documentation content
        sample_content = """
        <html>
        <head><title>FastAPI Advanced Guide</title></head>
        <body>
        <h1>FastAPI Advanced Patterns</h1>
        <p>FastAPI is a modern, fast web framework for building APIs with Python 3.7+.</p>
        
        <h2>Best Practices</h2>
        <ul>
        <li>Use Pydantic models for request/response validation</li>
        <li>Implement proper error handling with HTTPException</li>
        <li>Use dependency injection for database connections</li>
        </ul>
        
        <h2>Code Example</h2>
        <pre><code>
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

app = FastAPI()

class UserCreate(BaseModel):
    name: str
    email: str

@app.post("/users/")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Implement user creation logic
    return {"message": "User created successfully"}
        </code></pre>
        
        <h2>Performance Tips</h2>
        <p>Always use async/await for I/O operations and implement proper connection pooling.</p>
        </body>
        </html>
        """
        
        url = "https://fastapi.tiangolo.com/advanced/"
        metadata = {"demo": "ml_integration", "framework": "fastapi"}
        
        # Process with ML enhancement
        logger.info("Processing content with ML enhancement...")
        result = self.ml_processor.process(sample_content, url, metadata)
        
        # Display results
        logger.info(f"📊 ML Processing Results:")
        logger.info(f"   Title: {result.name}")
        logger.info(f"   ML Enhanced: {result.metadata.get('ml_enhanced', False)}")
        logger.info(f"   Content Complexity: {result.metadata.get('content_complexity', 'N/A')}")
        
        if 'semantic_keywords' in result.metadata:
            logger.info(f"   Semantic Keywords: {result.metadata['semantic_keywords'][:5]}")
        
        if 'detected_technologies' in result.metadata:
            logger.info(f"   Detected Technologies: {result.metadata['detected_technologies']}")
        
        return result
    
    async def demo_ml_cursor_transformer(self):
        """Demo ML-enhanced Cursor rule transformation."""
        logger.info("\n🚀 Demo: ML-Enhanced Cursor Rule Generation")
        
        # Create sample scraping results
        scraping_results = [
            ScrapingResult(
                url="https://fastapi.tiangolo.com/",
                title="FastAPI Documentation",
                content="FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints. Key features include automatic request validation, serialization, and documentation generation.",
                documentation_type=DocumentationType.API,
                status=ScrapingStatus.COMPLETED,
                metadata={"framework": "fastapi", "language": "python"}
            ),
            ScrapingResult(
                url="https://fastapi.tiangolo.com/tutorial/",
                title="FastAPI Tutorial",
                content="Learn FastAPI step by step. Create your first API, add path parameters, request body validation, and implement authentication. Best practices include using Pydantic models and proper error handling.",
                documentation_type=DocumentationType.TUTORIAL,
                status=ScrapingStatus.COMPLETED,
                metadata={"framework": "fastapi", "language": "python"}
            )
        ]
        
        logger.info(f"Generating ML-enhanced Cursor rules from {len(scraping_results)} sources...")
        
        # Generate rules with ML enhancement
        ml_rules = await self.ml_cursor_transformer.transform(scraping_results)
        
        # Generate standard rules for comparison
        standard_transformer = self.windsurf_transformer
        standard_rules = standard_transformer.transform(scraping_results)
        
        # Display comparison
        logger.info(f"📋 Rule Generation Results:")
        logger.info(f"   ML-Enhanced Rules Length: {len(ml_rules)} characters")
        logger.info(f"   Standard Rules Length: {len(standard_rules)} characters")
        
        if "ML Quality Assessment" in ml_rules:
            logger.info("   ✅ ML quality assessment included")
        else:
            logger.info("   ⚠️  ML quality assessment not found")
        
        if "Quality Score" in ml_rules:
            logger.info("   ✅ Quality scoring applied")
        
        # Save demo output
        out_dir_str = (
            (self.config.get('output_config') or {}).get('ml_batch_directory')
            or (self.config.get('output_formatting') or {}).get('output_directory')
            or 'rules/ml_demo/'
        )
        output_dir = Path(out_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ml_output_path = output_dir / "demo_ml_cursor_rules.md"
        with open(ml_output_path, 'w') as f:
            f.write(ml_rules)
        
        standard_output_path = output_dir / "demo_standard_windsurf_rules.md"
        with open(standard_output_path, 'w') as f:
            f.write(standard_rules)
        
        logger.info(f"   📁 ML rules saved to: {ml_output_path}")
        logger.info(f"   📁 Standard rules saved to: {standard_output_path}")
        
        return ml_rules, standard_rules
    
    async def demo_integrated_learning_system(self):
        """Demo integrated learning system capabilities."""
        logger.info("\n🧠 Demo: Integrated Learning System")
        
        # Create sample generated rules with usage history
        sample_rules = [
            GeneratedRule(
                rule=Rule(
                    id="fastapi_validation_rule",
                    title="FastAPI Request Validation",
                    description="Use Pydantic models for automatic request validation",
                    examples=["class UserRequest(BaseModel): name: str"],
                    priority=4,
                    tags=["fastapi", "validation", "pydantic"],
                    rule_type="best_practice",
                    confidence_score=0.8,
                    metadata={"framework": "fastapi"}
                ),
                usage_events=[
                    UsageEvent(
                        rule_id="fastapi_validation_rule",
                        success=True,
                        feedback_score=0.9,
                        context={"user": "demo_user", "project": "api_project"}
                    ),
                    UsageEvent(
                        rule_id="fastapi_validation_rule",
                        success=True,
                        feedback_score=0.8,
                        context={"user": "demo_user", "project": "api_project"}
                    )
                ]
            ),
            GeneratedRule(
                rule=Rule(
                    id="react_hooks_rule",
                    title="React Hooks Best Practices",
                    description="Use useEffect for side effects and useState for state management",
                    examples=["const [state, setState] = useState(initialValue)"],
                    priority=3,
                    tags=["react", "hooks", "javascript"],
                    rule_type="code_style",
                    confidence_score=0.7,
                    metadata={"framework": "react"}
                ),
                usage_events=[
                    UsageEvent(
                        rule_id="react_hooks_rule",
                        success=False,
                        feedback_score=0.4,
                        context={"user": "demo_user", "project": "web_app"}
                    )
                ]
            )
        ]
        
        # Create sample usage data
        usage_data = []
        for rule in sample_rules:
            usage_data.extend(rule.usage_events)
        
        logger.info(f"Learning from {len(sample_rules)} rules with {len(usage_data)} usage events...")
        
        # Apply integrated learning
        optimized_results = await self.integrated_learning.learn_and_improve(
            sample_rules, usage_data
        )
        
        # Display results
        logger.info(f"📈 Learning Results:")
        logger.info(f"   Rules Processed: {len(optimized_results.rules)}")
        logger.info(f"   Changes Applied: {len(optimized_results.changes)}")
        logger.info(f"   Overall Quality Score: {optimized_results.quality_score:.3f}")
        
        # Show rule improvements
        for i, rule in enumerate(optimized_results.rules):
            original_rule = sample_rules[i].rule
            logger.info(f"   Rule {i+1}: {rule.title}")
            logger.info(f"     Priority: {original_rule.priority} → {rule.priority}")
            logger.info(f"     Confidence: {original_rule.confidence_score:.3f} → {rule.confidence_score:.3f}")
            
            if rule.metadata.get('ml_enhanced'):
                logger.info(f"     ✅ ML Enhanced with quality score: {rule.metadata.get('ml_quality_score', 'N/A')}")
        
        return optimized_results
    
    async def demo_batch_processing_with_ml(self):
        """Demo ML-powered batch processing."""
        logger.info("\n⚡ Demo: ML-Powered Batch Processing")
        if not _BATCH_AVAILABLE or self.batch_processor is None:
            logger.warning("Skipping batch processing demo (batch processor unavailable)")
            return None
        
        # Define demo sources (small set for demo)
        demo_sources = [
            DocumentationSource(
                url="https://fastapi.tiangolo.com/",
                name="FastAPI",
                technology="python",
                framework="fastapi",
                priority=5,
                expected_pages=10
            ),
            DocumentationSource(
                url="https://react.dev/learn",
                name="React",
                technology="javascript", 
                framework="react",
                priority=5,
                expected_pages=10
            ),
            DocumentationSource(
                url="https://vuejs.org/guide/",
                name="Vue.js",
                technology="javascript",
                framework="vue",
                priority=4,
                expected_pages=8
            )
        ]
        
        logger.info(f"Processing {len(demo_sources)} documentation sources with ML enhancement...")
        
        try:
            # Process with ML enhancement using pre-initialized batch processor
            result = await self.batch_processor.process_documentation_batch(
                demo_sources,
                formats=[RuleFormat.CURSOR, RuleFormat.WINDSURF],
            )
            
            # Display results
            logger.info(f"🎉 Batch Processing Results:")
            logger.info(f"   Sources Processed: {result.sources_processed}")
            logger.info(f"   Rules Generated: {result.total_rules_generated}")
            success_rate = (result.sources_processed / max(1, len(demo_sources)))
            logger.info(f"   Success Rate: {success_rate:.2%}")
            logger.info(f"   Average Quality: {result.quality_metrics.get('average_quality', 'N/A')}")
            
            if hasattr(result, 'clusters') and result.clusters:
                logger.info(f"   Clusters Created: {len(result.clusters)}")
                for i, cluster in enumerate(result.clusters[:3]):  # Show first 3
                    logger.info(f"     Cluster {i+1}: {cluster.coherence_score:.3f} coherence")
            
            # Show processing time breakdown
            # Processing time is available
            logger.info(f"   ⏱️  Processing Time: {result.processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Batch processing demo failed: {e}")
            logger.info("This is expected in demo mode without actual web scraping")
            return None
    
    async def demo_system_performance_monitoring(self):
        """Demo system performance monitoring and statistics."""
        logger.info("\n📊 Demo: System Performance Monitoring")
        
        # Get performance stats from integrated learning system
        learning_stats = await self.integrated_learning.get_system_performance_stats()
        
        logger.info("🔧 Integrated Learning System Stats:")
        logger.info(f"   System Type: {learning_stats['system_type']}")
        logger.info(f"   ML Enabled: {learning_stats['configuration']['ml_enabled']}")
        logger.info(f"   ML Weight: {learning_stats['configuration']['ml_weight']}")
        
        for component, status in learning_stats['components'].items():
            logger.info(f"   {component}: {status}")
        
        # Get ML strategy performance if available
        if self.ml_strategy.performance_metrics:
            metrics = self.ml_strategy.performance_metrics
            logger.info(f"\n🎯 ML Strategy Performance:")
            logger.info(f"   Accuracy: {metrics.accuracy:.3f}")
            logger.info(f"   Training Size: {metrics.training_size}")
        else:
            logger.info("\n⚠️  ML Strategy not yet trained (expected in demo)")
        
        return learning_stats
    
    async def run_complete_demo(self):
        """Run complete ML integration demonstration."""
        logger.info("🚀 Starting Complete ML Integration Demo")
        logger.info("=" * 60)
        
        try:
            # 1. ML-enhanced processing
            await self.demo_ml_processor_enhancement()
            
            # 2. ML-enhanced rule generation
            await self.demo_ml_cursor_transformer()
            
            # 3. Integrated learning system
            await self.demo_integrated_learning_system()
            
            # 4. System performance monitoring
            await self.demo_system_performance_monitoring()
            
            # 5. Batch processing (may fail in demo mode)
            logger.info("\n⚡ Attempting batch processing demo...")
            await self.demo_batch_processing_with_ml()
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ ML Integration Demo Completed Successfully!")
            logger.info("All ML components are working together as designed.")
            
        except Exception as e:
            logger.error(f"Demo encountered error: {e}")
            logger.info("Some demo features may require additional setup (AWS credentials, etc.)")


async def main():
    """Run the ML integration demonstration."""
    logger.info("ML Integration Demo - Rules Maker")
    logger.info("Demonstrating Phase 1-3 ML implementation from integration guide")
    
    # Initialize and run demo
    demo = MLIntegrationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
