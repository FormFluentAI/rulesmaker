#!/usr/bin/env python3
"""
Smart Rules Organizer - Intelligent categorization and sorting of windsurf rule files
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

class RulesOrganizer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.sorted_dir = self.base_dir / "rules" / "sorted"
        self.manifest_file = self.sorted_dir / "MANIFEST.json"

        # Define comprehensive categorization rules
        self.categories = {
            # Programming Languages
            "python": {
                "keywords": ["python", "pydata", "readthedocs", "scipy", "numpy", "pandas", "matplotlib", "scikit", "pytorch", "tensorflow", "keras", "jupyter", "ipython", "pip", "conda", "venv"],
                "frameworks": ["django", "flask", "fastapi", "tornado", "bottle", "cherrypy", "sanic", "aiohttp", "requests", "httpx", "click", "rich", "black", "flake8", "mypy", "pytest", "tox", "coverage"],
                "ai_ml": ["transformers", "tokenizers", "sentence-transformers", "spacy", "nltk", "gensim", "xgboost", "lightgbm", "scikit-learn", "statsmodels"]
            },
            "javascript": {
                "keywords": ["javascript", "js", "node", "npm", "yarn", "webpack", "babel", "eslint", "prettier", "jest", "mocha", "chai"],
                "frameworks": ["react", "vue", "angular", "svelte", "nextjs", "nuxt", "express", "koa", "socket.io", "threejs", "phaser", "docusaurus", "storybook"],
                "libraries": ["jquery", "lodash", "moment", "axios", "redux", "mobx", "recoil", "zustand", "tanstack", "trpc", "graphql", "apollo"]
            },
            "typescript": {
                "keywords": ["typescript", "ts", "tsc"],
                "frameworks": ["nestjs", "strapi", "prisma", "trpc", "tanstack"],
                "ui_libraries": ["chakra-ui", "mui", "mantine", "shadcn", "radix-ui", "headlessui", "ant.design"]
            },
            "css": {
                "keywords": ["css", "scss", "sass", "less", "stylus"],
                "frameworks": ["tailwind", "bootstrap", "bulma", "foundation", "materialize"],
                "tools": ["postcss", "autoprefixer", "css-modules"]
            },

            # Backend & Infrastructure
            "databases": {
                "relational": ["postgresql", "mysql", "sqlite", "sqlalchemy", "alembic", "prisma"],
                "nosql": ["mongodb", "redis", "neo4j", "cassandra", "couchdb"],
                "orm": ["sqlalchemy", "prisma", "mongoose", "sequelize"],
                "time_series": ["influxdb", "timescale", "prometheus"]
            },
            "devops": {
                "containers": ["docker", "kubernetes", "helm", "istio", "argoproj"],
                "ci_cd": ["jenkins", "concourse", "spinnaker", "github-actions"],
                "cloud": ["aws", "azure", "gcp", "terraform", "ansible"],
                "monitoring": ["grafana", "prometheus", "elastic"]
            },

            # AI & Machine Learning
            "ai_ml": {
                "frameworks": ["tensorflow", "pytorch", "keras", "jax", "transformers"],
                "platforms": ["huggingface", "openai", "anthropic", "ollama"],
                "tools": ["mlflow", "dvc", "wandb", "comet"]
            },

            # Mobile & Desktop
            "mobile": {
                "cross_platform": ["react-native", "flutter", "ionic", "capacitor"],
                "native": ["android", "ios", "swift", "kotlin"]
            },

            # Game Development
            "game_dev": {
                "engines": ["unity", "godot", "unreal"],
                "web": ["threejs", "phaser", "babylon"]
            },

            # Other Languages
            "java": {
                "frameworks": ["java", "spring", "hibernate", "maven", "gradle"],
                "tools": ["java", "maven", "gradle"]
            },
            "csharp": {
                "frameworks": [".net", "csharp", "asp.net", "entity-framework"],
                "tools": [".net", "csharp"]
            },
            "go": {
                "frameworks": ["golang", "go"],
                "tools": ["golang", "go"]
            },
            "rust": {
                "frameworks": ["rust"],
                "tools": ["rust"]
            },
            "php": {
                "frameworks": ["php", "laravel", "symfony"],
                "tools": ["php"]
            },
            "ruby": {
                "frameworks": ["ruby", "rails"],
                "tools": ["ruby"]
            },
            "scala": {
                "frameworks": ["scala", "akka"],
                "tools": ["scala"]
            },
            "clojure": {
                "frameworks": ["clojure"],
                "tools": ["clojure"]
            },
            "elixir": {
                "frameworks": ["elixir"],
                "tools": ["elixir"]
            },
            "r": {
                "frameworks": ["r-project", "rstudio"],
                "tools": ["r-project", "rstudio"]
            },
            "lua": {
                "frameworks": ["lua"],
                "tools": ["lua"]
            },
            "solidity": {
                "frameworks": ["solidity", "ethereum", "web3"],
                "tools": ["solidity"]
            }
        }

        # Special domain mappings
        self.domain_mappings = {
            "docs.python.org": "python",
            "readthedocs.io": "python",
            "palletsprojects.com": "python",
            "tiangolo.com": "python",
            "react.dev": "javascript",
            "vuejs.org": "javascript",
            "angular.io": "typescript",
            "svelte.dev": "javascript",
            "nextjs.org": "javascript",
            "nuxtjs.org": "javascript",
            "nestjs.com": "typescript",
            "tailwindcss.com": "css",
            "getbootstrap.com": "css",
            "bulma.io": "css",
            "mui.com": "typescript",
            "chakra-ui.com": "typescript",
            "ant.design": "typescript",
            "mantine.dev": "typescript",
            "ui.shadcn.com": "typescript",
            "radix-ui.com": "typescript",
            "docs.docker.com": "devops",
            "kubernetes.io": "devops",
            "helm.sh": "devops",
            "docs.aws.amazon.com": "devops",
            "docs.microsoft.com": "devops",
            "cloud.google.com": "devops",
            "www.terraform.io": "devops",
            "docs.ansible.com": "devops",
            "www.jenkins.io": "devops",
            "grafana.com": "devops",
            "prometheus.io": "devops",
            "www.elastic.co": "devops",
            "huggingface.co": "ai_ml",
            "www.openai.com": "ai_ml",
            "www.anthropic.com": "ai_ml",
            "ollama.com": "ai_ml",
            "developer.android.com": "mobile",
            "developer.apple.com": "mobile",
            "flutter.dev": "mobile",
            "ionicframework.com": "mobile",
            "docs.godotengine.org": "game_dev",
            "unity.com": "game_dev",
            "threejs.org": "game_dev",
            "phaser.io": "game_dev"
        }

    def analyze_filename(self, filename: str) -> Tuple[str, str, str]:
        """Analyze filename to determine category, subcategory, and technology"""
        filename_lower = filename.lower().replace('.md', '').replace('.mdc', '').replace('_', '.')

        # Check domain mappings first
        for domain, category in self.domain_mappings.items():
            if domain in filename_lower:
                return category, "frameworks", self._extract_technology(filename_lower)

        # Check keywords in each category
        for category, subcats in self.categories.items():
            for subcat_name, keywords in subcats.items():
                if isinstance(keywords, list):
                    for keyword in keywords:
                        if keyword in filename_lower:
                            return category, subcat_name, self._extract_technology(filename_lower)

        # Fallback categorization
        if any(lang in filename_lower for lang in ['python', 'django', 'flask', 'fastapi']):
            return "python", "frameworks", "python"
        elif any(lang in filename_lower for lang in ['javascript', 'js', 'react', 'vue', 'angular', 'node']):
            return "javascript", "frameworks", "javascript"
        elif any(lang in filename_lower for lang in ['typescript', 'ts', 'nestjs']):
            return "typescript", "frameworks", "typescript"
        elif any(css in filename_lower for css in ['css', 'tailwind', 'bootstrap', 'bulma']):
            return "css", "frameworks", "css"
        else:
            return "unknown", "misc", "unknown"

    def _extract_technology(self, filename: str) -> str:
        """Extract primary technology from filename"""
        filename_lower = filename.lower()

        # Technology detection patterns
        tech_patterns = {
            "python": ["python", "django", "flask", "fastapi", "pytorch", "tensorflow"],
            "javascript": ["javascript", "react", "vue", "angular", "svelte", "node"],
            "typescript": ["typescript", "nestjs", "prisma"],
            "css": ["css", "tailwind", "bootstrap", "bulma"],
            "java": ["java", "spring"],
            "csharp": ["csharp", ".net"],
            "go": ["golang", "go"],
            "rust": ["rust"],
            "php": ["php", "laravel"],
            "ruby": ["ruby", "rails"],
            "scala": ["scala"],
            "clojure": ["clojure"],
            "elixir": ["elixir"],
            "r": ["r-project"],
            "lua": ["lua"],
            "solidity": ["solidity"]
        }

        for tech, patterns in tech_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                return tech

        return "unknown"

    def create_directory_structure(self):
        """Create organized directory structure"""
        # Ensure base sorted directory exists
        self.sorted_dir.mkdir(exist_ok=True)

        structure = {
            "python": ["frameworks", "ai_ml", "data_science", "testing", "tools"],
            "javascript": ["frameworks", "libraries", "build_tools", "testing"],
            "typescript": ["frameworks", "ui_libraries", "tools"],
            "css": ["frameworks", "tools", "utilities"],
            "databases": ["relational", "nosql", "orm", "time_series"],
            "devops": ["containers", "ci_cd", "cloud", "monitoring"],
            "ai_ml": ["frameworks", "platforms", "tools"],
            "mobile": ["cross_platform", "native"],
            "game_dev": ["engines", "web"],
            "java": ["frameworks", "tools"],
            "csharp": ["frameworks", "tools"],
            "go": ["frameworks", "tools"],
            "rust": ["frameworks", "tools"],
            "php": ["frameworks", "tools"],
            "ruby": ["frameworks", "tools"],
            "scala": ["frameworks", "tools"],
            "clojure": ["frameworks", "tools"],
            "elixir": ["frameworks", "tools"],
            "r": ["frameworks", "tools"],
            "lua": ["frameworks", "tools"],
            "solidity": ["frameworks", "tools"],
            "unknown": ["misc"]
        }

        for category, subcategories in structure.items():
            category_dir = self.sorted_dir / category
            try:
                category_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Created directory: {category_dir}")
            except Exception as e:
                print(f"  Warning: Could not create {category_dir}: {e}")

            for subcategory in subcategories:
                sub_dir = category_dir / subcategory
                try:
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    print(f"    Created subdirectory: {sub_dir}")
                except Exception as e:
                    print(f"    Warning: Could not create {sub_dir}: {e}")

    def organize_files(self):
        """Organize all windsurf files into categorized directories"""
        # Find all windsurf files in the base directory and its subdirectories
        # This handles files that are already organized in subdirectories
        windsurf_files = list(self.base_dir.rglob("*.md")) + list(self.base_dir.rglob("*.mdc"))
        
        print(f"Found {len(windsurf_files)} rule files to organize")
        
        if not windsurf_files:
            print("No rule files found to organize")
            return {}, []

        organized_files = defaultdict(lambda: defaultdict(list))
        moved_files = []

        for file_path in windsurf_files:
            # Skip manifest and insights files
            if any(skip_name in file_path.name for skip_name in ["insights", "MANIFEST", "README"]):
                continue

            print(f"Processing: {file_path.name}")

            category, subcategory, technology = self.analyze_filename(file_path.name)

            # Create target directory
            target_dir = self.sorted_dir / category / subcategory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy file (don't move to preserve original organization)
            target_path = target_dir / file_path.name

            try:
                shutil.copy2(str(file_path), str(target_path))
                print(f"  Copied to: {target_path.relative_to(self.sorted_dir)}")

                organized_files[category][subcategory].append({
                    "filename": file_path.name,
                    "original_path": str(file_path),
                    "new_path": str(target_path),
                    "technology": technology
                })

                moved_files.append({
                    "filename": file_path.name,
                    "from": str(file_path.relative_to(self.base_dir)),
                    "to": str(target_path.relative_to(self.sorted_dir)),
                    "category": category,
                    "subcategory": subcategory,
                    "technology": technology
                })
            except Exception as e:
                print(f"  Error copying {file_path.name}: {e}")

        print(f"Successfully organized {len(moved_files)} files")
        return organized_files, moved_files

    def create_manifest(self, organized_files: Dict, moved_files: List):
        """Create a comprehensive manifest file"""
        if not organized_files and not moved_files:
            print("No files to organize - creating basic manifest")
            manifest = {
                "organization_info": {
                    "total_files": 0,
                    "categories": [],
                    "created_at": "2025-09-02",
                    "organizer_version": "1.0",
                    "note": "No files were moved - they may already be organized"
                },
                "structure": {},
                "file_mappings": [],
                "search_index": {}
            }
        else:
            manifest = {
                "organization_info": {
                    "total_files": sum(len(files) for cat in organized_files.values() for files in cat.values()),
                    "categories": list(organized_files.keys()),
                    "created_at": "2025-09-02",
                    "organizer_version": "1.0"
                },
                "structure": {},
                "file_mappings": moved_files,
                "search_index": {}
            }

            # Build structure and search index
            for category, subcategories in organized_files.items():
                manifest["structure"][category] = {}
                manifest["search_index"][category] = []

                for subcategory, files in subcategories.items():
                    manifest["structure"][category][subcategory] = [
                        {
                            "filename": f["filename"],
                            "technology": f["technology"],
                            "path": f["new_path"]
                        } for f in files
                    ]

                    # Add to search index
                    for file_info in files:
                        manifest["search_index"][category].append({
                            "filename": file_info["filename"],
                            "technology": file_info["technology"],
                            "category": category,
                            "subcategory": subcategory,
                            "path": file_info["new_path"]
                        })

        # Write manifest
        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest

    def create_readme(self):
        """Create a README file explaining the organization"""
        readme_content = f"""# Windsurf Rules Organization

This directory contains intelligently organized windsurf workflow rules generated from various documentation sources.

## Organization Structure

The rules are organized into the following hierarchical structure:

```
sorted/
├── python/
│   ├── frameworks/     # Django, Flask, FastAPI, etc.
│   ├── ai_ml/         # TensorFlow, PyTorch, Transformers, etc.
│   ├── data_science/  # NumPy, Pandas, SciPy, etc.
│   ├── testing/       # pytest, tox, coverage, etc.
│   └── tools/         # Black, flake8, mypy, etc.
├── javascript/
│   ├── frameworks/    # React, Vue, Angular, Svelte, etc.
│   ├── libraries/     # Redux, MobX, Three.js, etc.
│   ├── build_tools/   # Webpack, Babel, Rollup, etc.
│   └── testing/       # Jest, Mocha, Testing Library, etc.
├── typescript/
│   ├── frameworks/    # NestJS, Strapi, etc.
│   ├── ui_libraries/  # Chakra UI, MUI, Mantine, etc.
│   └── tools/         # TypeScript tools and utilities
├── css/
│   ├── frameworks/    # Tailwind, Bootstrap, Bulma, etc.
│   ├── tools/         # PostCSS, Autoprefixer, etc.
│   └── utilities/     # CSS utilities and helpers
├── databases/
│   ├── relational/    # PostgreSQL, MySQL, SQLite
│   ├── nosql/         # MongoDB, Redis, Neo4j
│   ├── orm/           # SQLAlchemy, Prisma, Mongoose
│   └── time_series/   # InfluxDB, TimescaleDB
├── devops/
│   ├── containers/    # Docker, Kubernetes, Helm
│   ├── ci_cd/         # Jenkins, Concourse, GitHub Actions
│   ├── cloud/         # AWS, Azure, GCP, Terraform
│   └── monitoring/    # Grafana, Prometheus, ELK
├── ai_ml/
│   ├── frameworks/    # TensorFlow, PyTorch, JAX
│   ├── platforms/     # Hugging Face, OpenAI, Anthropic
│   └── tools/         # MLflow, DVC, Weights & Biases
├── mobile/
│   ├── cross_platform/ # React Native, Flutter, Ionic
│   └── native/        # Android, iOS development
├── game_dev/
│   ├── engines/       # Unity, Godot, Unreal
│   └── web/           # Three.js, Phaser, Babylon.js
└── [other languages]/ # Java, C#, Go, Rust, PHP, Ruby, etc.
```

## File Naming Convention

Files are named based on their source URL with the following pattern:
- `domain.path.filename.windsurf`
- Example: `fastapi.tiangolo.com_.windsurf`

## Search and Discovery

- Use `MANIFEST.json` for programmatic access to all files
- Each category has a search index for quick lookup
- Files are grouped by technology and purpose

## Statistics

- **Total Rules**: {sum(len(files) for cat in json.load(open(self.manifest_file)).get('structure', {}).values() for files in cat.values())}
- **Categories**: {len(json.load(open(self.manifest_file)).get('structure', {}))}
- **Last Updated**: 2025-09-02

## Usage

To find rules for a specific technology:

1. Navigate to the appropriate category directory
2. Look in the relevant subcategory
3. Open the `.windsurf` file for the specific tool/framework

Example: Python web frameworks → `python/frameworks/`
Example: React development → `javascript/frameworks/`
"""

        readme_path = self.sorted_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def run_organization(self):
        """Run the complete organization process"""
        print("🚀 Starting smart organization of windsurf rules...")

        # Create directory structure
        print("📁 Creating directory structure...")
        self.create_directory_structure()

        # Organize files
        print("📋 Analyzing and organizing files...")
        organized_files, moved_files = self.organize_files()

        # Create manifest
        print("📄 Creating manifest file...")
        manifest = self.create_manifest(organized_files, moved_files)

        # Create README
        print("📖 Creating README...")
        self.create_readme()

        print("✅ Organization complete!")
        print(f"📊 Organized {len(moved_files)} files into {len(organized_files)} categories")
        print(f"📋 Manifest saved to: {self.manifest_file}")
        print(f"📖 README saved to: {self.sorted_dir / 'README.md'}")

        return manifest

def main():
    base_dir = "/home/ollie/dev/rules-maker"
    organizer = RulesOrganizer(base_dir)
    manifest = organizer.run_organization()

    print("\n📈 Organization Summary:")
    for category, subcategories in manifest['structure'].items():
        total_files = sum(len(files) for files in subcategories.values())
        print(f"  {category}: {total_files} files")
        for subcategory, files in subcategories.items():
            print(f"    └─ {subcategory}: {len(files)} files")

if __name__ == "__main__":
    main()
