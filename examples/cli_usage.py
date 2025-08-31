"""
CLI usage example for Rules Maker.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    
    return result.returncode == 0


def main():
    """Demonstrate CLI usage."""
    
    print("Rules Maker CLI Examples")
    print("=" * 40)
    
    # Example 1: Simple scraping
    print("\n1. Simple URL scraping:")
    run_command("rules-maker scrape https://docs.python.org/3/library/json.html")
    
    # Example 2: Deep scraping with output
    print("\n2. Deep scraping with file output:")
    run_command("rules-maker scrape https://flask.palletsprojects.com/en/2.3.x/ --deep --output flask_rules.cursorrules --max-pages 10")
    
    # Example 3: Batch processing
    print("\n3. Creating batch URLs file:")
    urls = [
        "https://docs.python.org/3/library/json.html",
        "https://requests.readthedocs.io/en/latest/",
        "https://flask.palletsprojects.com/en/2.3.x/"
    ]
    
    with open("example_urls.txt", "w") as f:
        f.write("\n".join(urls))
    
    print("Created example_urls.txt with sample URLs")
    
    print("\n4. Batch processing:")
    run_command("rules-maker batch example_urls.txt --output-dir ./rules --format cursor")
    
    # Example 5: List templates
    print("\n5. List available templates:")
    run_command("rules-maker templates")
    
    # Example 6: Initialize config
    print("\n6. Initialize configuration:")
    run_command("rules-maker init-config config.yaml")
    
    print("\nCLI examples completed!")


if __name__ == "__main__":
    main()
