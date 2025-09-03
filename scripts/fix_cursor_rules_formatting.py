#!/usr/bin/env python3
"""
Fix Cursor Rules Formatting

This script fixes the YAML frontmatter formatting in all .mdc files to use proper YAML list syntax
for the globs field instead of JSON array format.
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_globs_formatting(content: str) -> str:
    """Fix the globs formatting from JSON array to proper YAML list format."""
    
    # Pattern to match the incorrect format: globs: ["pattern1", "pattern2", ...]
    pattern = r'globs:\s*\[([^\]]+)\]'
    
    def replace_globs(match):
        # Extract the array content
        array_content = match.group(1)
        
        # Split by comma and clean up each item
        items = [item.strip().strip('"\'') for item in array_content.split(',')]
        
        # Convert to proper YAML list format
        yaml_list = "globs:\n" + "\n".join([f"  - '{item}'" for item in items if item])
        
        return yaml_list
    
    # Replace the pattern
    fixed_content = re.sub(pattern, replace_globs, content)
    
    return fixed_content


def fix_cursor_rule_file(file_path: Path) -> bool:
    """Fix the formatting of a single cursor rule file."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file has the incorrect format
        if 'globs: [' in content:
            logger.info(f"Fixing formatting in: {file_path.name}")
            
            # Fix the formatting
            fixed_content = fix_globs_formatting(content)
            
            # Write the fixed content back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"✅ Fixed formatting in: {file_path.name}")
            return True
        else:
            logger.info(f"✅ No formatting issues in: {file_path.name}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error fixing {file_path.name}: {e}")
        return False


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent
    cursor_rules_dir = project_root / ".cursor" / "rules"
    
    if not cursor_rules_dir.exists():
        logger.error(f"Cursor rules directory not found: {cursor_rules_dir}")
        sys.exit(1)
    
    logger.info("🔧 Fixing Cursor rules formatting...")
    logger.info(f"Processing directory: {cursor_rules_dir}")
    
    fixed_files = []
    total_files = 0
    
    # Process all .mdc files
    for mdc_file in cursor_rules_dir.glob("*.mdc"):
        total_files += 1
        if fix_cursor_rule_file(mdc_file):
            fixed_files.append(mdc_file.name)
    
    logger.info(f"🎉 Formatting fix completed!")
    logger.info(f"📊 Processed {total_files} files")
    logger.info(f"🔧 Fixed {len(fixed_files)} files")
    
    if fixed_files:
        logger.info("Fixed files:")
        for file_name in fixed_files:
            logger.info(f"  - {file_name}")
    
    # Show example of the fix
    if fixed_files:
        logger.info("\n📝 Example of the fix:")
        logger.info("Before: globs: [\"**/*.tsx\", \"**/*.ts\", \"**/*.jsx\"]")
        logger.info("After:")
        logger.info("globs:")
        logger.info("  - '**/*.tsx'")
        logger.info("  - '**/*.ts'")
        logger.info("  - '**/*.jsx'")


if __name__ == "__main__":
    main()
