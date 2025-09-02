# Windsurf Rules Organization

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

- **Total Rules**: 0
- **Categories**: 0
- **Last Updated**: 2025-09-02

## Usage

To find rules for a specific technology:

1. Navigate to the appropriate category directory
2. Look in the relevant subcategory
3. Open the `.windsurf` file for the specific tool/framework

Example: Python web frameworks → `python/frameworks/`
Example: React development → `javascript/frameworks/`
