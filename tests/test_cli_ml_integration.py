"""
Integration tests for ML-enhanced CLI commands.
Tests all the newly implemented ML batch processing, quality assessment, 
learning, and analytics commands.
"""

import pytest
import json
import tempfile
from pathlib import Path
from click.testing import CliRunner

from rules_maker.cli import main


class TestMLBatchCommands:
    """Test ML batch processing commands."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_ml_batch_frameworks_dry_run(self):
        """Test ML batch frameworks command in dry run mode."""
        result = self.runner.invoke(main, [
            'ml-batch', 'frameworks',
            '--dry-run',
            '--output-dir', 'test_frameworks_output',
            '--quality-threshold', '0.8'
        ])
        
        # Should succeed in dry run even without ML dependencies
        assert result.exit_code == 0 or "ML batch features not available" in result.output
        
        if result.exit_code == 0:
            assert "Dry run mode" in result.output
            assert "frameworks" in result.output.lower()
    
    def test_ml_batch_cloud_dry_run(self):
        """Test ML batch cloud command in dry run mode."""
        result = self.runner.invoke(main, [
            'ml-batch', 'cloud',
            '--dry-run',
            '--output-dir', 'test_cloud_output',
            '--bedrock'
        ])
        
        assert result.exit_code == 0 or "ML batch features not available" in result.output
        
        if result.exit_code == 0:
            assert "Dry run mode" in result.output
            assert "cloud platforms" in result.output.lower()
    
    def test_ml_batch_custom_dry_run(self):
        """Test ML batch custom command with dry run."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            sources_data = [
                {
                    "url": "https://example.com/docs",
                    "name": "Example Framework",
                    "technology": "javascript",
                    "framework": "example",
                    "priority": 3
                }
            ]
            json.dump(sources_data, f)
            sources_file = f.name
        
        try:
            result = self.runner.invoke(main, [
                'ml-batch', 'custom',
                sources_file,
                '--output-dir', 'test_custom_output',
                '--dry-run'
            ])
            
            assert result.exit_code == 0 or "ML batch features not available" in result.output
            
            if result.exit_code == 0:
                assert "Dry run mode" in result.output
                assert "1 custom sources" in result.output
        finally:
            Path(sources_file).unlink()


class TestConfigCommands:
    """Test configuration management commands."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_config_init_minimal(self):
        """Test config initialization with minimal template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            result = self.runner.invoke(main, [
                'config', 'init',
                '--output', str(config_path),
                '--template', 'minimal'
            ])
            
            assert result.exit_code == 0
            assert "Configuration initialized" in result.output
            assert config_path.exists()
            
            # Verify config content
            config_content = config_path.read_text()
            assert "batch_processing" in config_content
            assert "ml_engine" in config_content
    
    def test_config_init_standard(self):
        """Test config initialization with standard template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "standard_config.yaml"
            
            result = self.runner.invoke(main, [
                'config', 'init',
                '--output', str(config_path),
                '--template', 'standard'
            ])
            
            assert result.exit_code == 0
            assert "Configuration initialized" in result.output
            assert config_path.exists()
            
            # Verify config has more sections than minimal
            config_content = config_path.read_text()
            assert "batch_processing" in config_content
            assert "integrated_learning" in config_content
            assert "bedrock_integration" in config_content
    
    def test_config_validate_valid(self):
        """Test config validation with valid configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "valid_config.yaml"
            
            # First create a config
            self.runner.invoke(main, [
                'config', 'init',
                '--output', str(config_path),
                '--template', 'minimal'
            ])
            
            # Then validate it
            result = self.runner.invoke(main, [
                'config', 'validate',
                str(config_path)
            ])
            
            assert result.exit_code == 0
            assert "Configuration validation passed" in result.output
    
    def test_config_validate_invalid(self):
        """Test config validation with invalid configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write invalid config
            f.write("invalid_yaml_content: [\n")  # Unclosed bracket
            invalid_config = f.name
        
        try:
            result = self.runner.invoke(main, [
                'config', 'validate',
                invalid_config
            ])
            
            assert result.exit_code != 0
            assert "Invalid YAML syntax" in result.output or "Validation failed" in result.output
        finally:
            Path(invalid_config).unlink()


class TestQualityCommands:
    """Test quality assessment commands."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_quality_assess_empty_directory(self):
        """Test quality assessment on empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(main, [
                'quality', 'assess',
                temp_dir,
                '--format', 'all'
            ])
            
            # Should handle empty directory gracefully
            assert result.exit_code == 0 or "Quality assessment features not available" in result.output
            
            if result.exit_code == 0:
                assert "No rule files found" in result.output
    
    def test_quality_assess_with_mock_rules(self):
        """Test quality assessment with mock rule files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock rule files
            cursor_rule = temp_path / "test_cursor.mdc"
            cursor_rule.write_text("""
            # Test Cursor Rules
            
            ## Code Style
            - Always use TypeScript
            - Never use `any` type
            
            ## Examples
            ```typescript
            function example(): string {
                return "hello";
            }
            ```
            
            **NEVER** ignore type errors.
            """)
            
            windsurf_rule = temp_path / "test_windsurf.md" 
            windsurf_rule.write_text("""
            # Test Windsurf Rules
            
            ## Best Practices
            1. Follow SOLID principles
            2. Write unit tests
            
            Example implementation:
            ```javascript
            const pattern = "observer";
            ```
            """)
            
            result = self.runner.invoke(main, [
                'quality', 'assess',
                str(temp_path),
                '--format', 'all',
                '--threshold', '0.5'
            ])
            
            # Should work if ML features available
            if "Quality assessment features not available" in result.output:
                # Expected when ML dependencies not installed
                assert result.exit_code != 0
            else:
                assert result.exit_code == 0
                assert "Quality Assessment Summary" in result.output
    
    def test_quality_cluster_no_insights(self):
        """Test cluster analysis with no insights files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(main, [
                'quality', 'cluster',
                temp_dir,
                '--min-coherence', '0.6'
            ])
            
            # Should handle missing insights gracefully
            assert result.exit_code == 0 or "Quality assessment features not available" in result.output
            
            if result.exit_code == 0:
                assert "No insights files found" in result.output


class TestLearningCommands:
    """Test learning system commands."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_learning_feedback_invalid_value(self):
        """Test learning feedback with invalid value."""
        result = self.runner.invoke(main, [
            'learning', 'feedback',
            '--rule-id', 'test_rule_123',
            '--signal-type', 'user_rating',
            '--value', '1.5'  # Invalid - should be 0.0-1.0
        ])
        
        # Should fail with validation error
        if "Learning features not available" not in result.output:
            assert result.exit_code != 0
            assert "must be between 0.0 and 1.0" in result.output
    
    def test_learning_feedback_invalid_context(self):
        """Test learning feedback with invalid JSON context."""
        result = self.runner.invoke(main, [
            'learning', 'feedback',
            '--rule-id', 'test_rule_123',
            '--signal-type', 'user_rating',
            '--value', '0.8',
            '--context', 'invalid_json{'
        ])
        
        # Should fail with JSON validation error
        if "Learning features not available" not in result.output:
            assert result.exit_code != 0
            assert "must be valid JSON" in result.output
    
    def test_learning_analyze_empty_directory(self):
        """Test learning analysis on empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(main, [
                'learning', 'analyze',
                temp_dir,
                '--format', 'json'
            ])
            
            # Should handle empty directory
            if "Learning features not available" not in result.output:
                assert result.exit_code == 0
                assert "Analyzing 0 rule files" in result.output


class TestAnalyticsCommands:
    """Test analytics commands."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_analytics_insights_empty_directory(self):
        """Test analytics insights on empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(main, [
                'analytics', 'insights',
                temp_dir,
                '--format', 'md'
            ])
            
            assert result.exit_code == 0
            assert "No processing results found" in result.output
    
    def test_analytics_insights_with_mock_data(self):
        """Test analytics insights with mock insights data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock insights file
            insights_file = temp_path / "processing_insights.json"
            mock_insights = {
                "result_summary": {
                    "sources_processed": 5,
                    "total_rules_generated": 15,
                    "processing_time": 120.5,
                    "quality_metrics": {
                        "overall_coherence": 0.85
                    }
                },
                "clusters": [
                    {
                        "id": "javascript_cursor_cluster_0",
                        "name": "JavaScript Cursor - Cluster 1",
                        "technology": "javascript",
                        "coherence_score": 0.9,
                        "rules_count": 8
                    },
                    {
                        "id": "python_cursor_cluster_0", 
                        "name": "Python Cursor - Cluster 1",
                        "technology": "python",
                        "coherence_score": 0.8,
                        "rules_count": 7
                    }
                ],
                "insights": {
                    "technology_distribution": {
                        "javascript": 8,
                        "python": 7
                    }
                }
            }
            
            with open(insights_file, 'w') as f:
                json.dump(mock_insights, f)
            
            result = self.runner.invoke(main, [
                'analytics', 'insights',
                str(temp_path),
                '--format', 'json'
            ])
            
            assert result.exit_code == 0
            assert "Processing Insights Summary" in result.output
            assert "Sources processed: 5" in result.output


class TestBedrockCommands:
    """Test enhanced Bedrock commands."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_bedrock_batch_dry_run(self):
        """Test Bedrock batch processing in dry run mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            sources_data = [
                {
                    "url": "https://fastapi.tiangolo.com/",
                    "name": "FastAPI",
                    "technology": "python",
                    "framework": "fastapi",
                    "priority": 5
                }
            ]
            json.dump(sources_data, f)
            sources_file = f.name
        
        try:
            result = self.runner.invoke(main, [
                'bedrock', 'batch',
                sources_file,
                '--output-dir', 'test_bedrock_output',
                '--dry-run',
                '--cost-limit', '5.0'
            ])
            
            # Should work in dry run mode
            if "Bedrock batch features require ML components" in result.output:
                assert "Install dependencies" in result.output
            else:
                assert result.exit_code == 0
                assert "Dry run mode" in result.output
                assert "Cost limit: $5.0" in result.output
        finally:
            Path(sources_file).unlink()


class TestEnhancedScrapeCommand:
    """Test enhanced scrape command with ML options."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_scrape_help_shows_ml_options(self):
        """Test that scrape command help shows new ML options."""
        result = self.runner.invoke(main, ['scrape', '--help'])
        
        assert result.exit_code == 0
        assert "--ml-enhanced" in result.output
        assert "--quality-assessment" in result.output  
        assert "--learning-feedback" in result.output
    
    def test_scrape_with_ml_options_no_url(self):
        """Test scrape command with ML options but no URL."""
        result = self.runner.invoke(main, [
            'scrape',
            '--ml-enhanced',
            '--quality-assessment',
            '--output', 'test_output.md'
        ])
        
        # Should prompt for URL or fail gracefully
        # The exact behavior depends on implementation details
        assert result.exit_code != 0 or "Enter documentation URL" in result.output


class TestCLIIntegration:
    """Test overall CLI integration and consistency."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_main_help_shows_all_command_groups(self):
        """Test that main CLI help shows all new command groups."""
        result = self.runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "ml-batch" in result.output
        assert "config" in result.output
        assert "quality" in result.output
        assert "learning" in result.output
        assert "analytics" in result.output
        assert "bedrock" in result.output
    
    def test_command_groups_have_help(self):
        """Test that all command groups have proper help."""
        command_groups = [
            'ml-batch', 'config', 'quality', 
            'learning', 'analytics', 'bedrock'
        ]
        
        for cmd_group in command_groups:
            result = self.runner.invoke(main, [cmd_group, '--help'])
            # Should either show help or indicate missing dependencies
            assert result.exit_code == 0 or "not available" in result.output
    
    def test_dependency_checks(self):
        """Test that ML commands properly check for dependencies."""
        ml_commands = [
            ['ml-batch', 'frameworks', '--dry-run'],
            ['quality', 'assess', '/nonexistent'],
            ['learning', 'feedback', '--rule-id', 'test', '--signal-type', 'user_rating', '--value', '0.5']
        ]
        
        for cmd in ml_commands:
            result = self.runner.invoke(main, cmd)
            # Should either work or show dependency error
            if result.exit_code != 0:
                assert ("not available" in result.output or 
                        "Install" in result.output or
                        "No such file" in result.output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])