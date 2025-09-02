"""
Updated documentation sources with current valid URLs.

This module provides corrected and validated documentation source lists
to replace the outdated URLs that were causing scraping failures.
"""

from typing import List
from ..batch_processor import DocumentationSource


def get_updated_web_frameworks() -> List[DocumentationSource]:
    """Updated web framework documentation sources with current URLs."""
    return [
        # React moved to react.dev
        DocumentationSource("https://react.dev/learn", "React", "javascript", "react", priority=5),
        DocumentationSource("https://vuejs.org/guide/", "Vue.js", "javascript", "vue", priority=5),
        DocumentationSource("https://angular.io/docs", "Angular", "javascript", "angular", priority=4),
        DocumentationSource("https://nextjs.org/docs", "Next.js", "javascript", "nextjs", priority=4),
        DocumentationSource("https://nuxt.com/docs", "Nuxt.js", "javascript", "nuxtjs", priority=3),
        DocumentationSource("https://svelte.dev/docs/introduction", "Svelte", "javascript", "svelte", priority=3),
        DocumentationSource("https://kit.svelte.dev/docs/introduction", "SvelteKit", "javascript", "sveltekit", priority=3),
        DocumentationSource("https://remix.run/docs/en/main", "Remix", "javascript", "remix", priority=3),
        DocumentationSource("https://expressjs.com/en/4x/api.html", "Express.js", "javascript", "express", priority=3),
        # Fastify moved to fastify.dev
        DocumentationSource("https://www.fastify.io/docs/latest/", "Fastify", "javascript", "fastify", priority=2),
    ]


def get_updated_python_frameworks() -> List[DocumentationSource]:
    """Updated Python framework documentation sources."""
    return [
        DocumentationSource("https://fastapi.tiangolo.com/", "FastAPI", "python", "fastapi", priority=5),
        DocumentationSource("https://flask.palletsprojects.com/en/3.0.x/", "Flask", "python", "flask", priority=4),
        DocumentationSource("https://docs.djangoproject.com/en/stable/", "Django", "python", "django", priority=4),
        DocumentationSource("https://docs.pydantic.dev/latest/", "Pydantic", "python", "pydantic", priority=4),
        DocumentationSource("https://docs.sqlalchemy.org/en/20/", "SQLAlchemy", "python", "sqlalchemy", priority=3),
        DocumentationSource("https://docs.celeryq.dev/en/stable/", "Celery", "python", "celery", priority=3),
        DocumentationSource("https://docs.pytest.org/en/stable/", "Pytest", "python", "pytest", priority=4),
        DocumentationSource("https://click.palletsprojects.com/en/8.1.x/", "Click", "python", "click", priority=2),
        DocumentationSource("https://requests.readthedocs.io/en/latest/", "Requests", "python", "requests", priority=3),
        DocumentationSource("https://www.python-httpx.org/", "HTTPX", "python", "httpx", priority=2),
    ]


def get_updated_backend_frameworks() -> List[DocumentationSource]:
    """Updated backend framework documentation sources."""
    return [
        DocumentationSource("https://spring.io/guides", "Spring Boot", "java", "spring", priority=4),
        DocumentationSource("https://docs.spring.io/spring-framework/reference/", "Spring Framework", "java", "spring", priority=3),
        DocumentationSource("https://quarkus.io/guides/", "Quarkus", "java", "quarkus", priority=3),
        DocumentationSource("https://micronaut.io/documentation.html", "Micronaut", "java", "micronaut", priority=2),
        # Rails moved to guides.rubyonrails.org
        DocumentationSource("https://guides.rubyonrails.org/", "Ruby on Rails", "ruby", "rails", priority=3),
        DocumentationSource("https://laravel.com/docs/10.x", "Laravel", "php", "laravel", priority=3),
        DocumentationSource("https://symfony.com/doc/current/index.html", "Symfony", "php", "symfony", priority=2),
        DocumentationSource("https://learn.microsoft.com/en-us/aspnet/core/", "ASP.NET Core", "csharp", "dotnet", priority=3),
        DocumentationSource("https://go.dev/doc/", "Go Documentation", "go", framework=None, priority=4),
        DocumentationSource("https://doc.rust-lang.org/book/", "Rust Book", "rust", framework=None, priority=3),
    ]


def get_updated_cloud_platforms() -> List[DocumentationSource]:
    """Updated cloud platform documentation sources."""
    return [
        DocumentationSource("https://docs.aws.amazon.com/", "AWS", "cloud", "aws", priority=5),
        DocumentationSource("https://learn.microsoft.com/en-us/azure/", "Azure", "cloud", "azure", priority=4),
        DocumentationSource("https://cloud.google.com/docs", "Google Cloud", "cloud", "gcp", priority=4),
        DocumentationSource("https://kubernetes.io/docs/home/", "Kubernetes", "cloud", "kubernetes", priority=5),
        DocumentationSource("https://docs.docker.com/", "Docker", "cloud", "docker", priority=4),
        DocumentationSource("https://developer.hashicorp.com/terraform/docs", "Terraform", "cloud", "terraform", priority=4),
        DocumentationSource("https://docs.ansible.com/ansible/latest/", "Ansible", "cloud", "ansible", priority=3),
        DocumentationSource("https://helm.sh/docs/", "Helm", "cloud", "helm", priority=3),
        DocumentationSource("https://istio.io/latest/docs/", "Istio", "cloud", "istio", priority=2),
        DocumentationSource("https://prometheus.io/docs/introduction/overview/", "Prometheus", "cloud", "prometheus", priority=3),
    ]


def get_updated_databases() -> List[DocumentationSource]:
    """Updated database documentation sources."""
    return [
        DocumentationSource("https://www.mongodb.com/docs/", "MongoDB", "database", "mongodb", priority=4),
        DocumentationSource("https://www.postgresql.org/docs/current/", "PostgreSQL", "database", "postgresql", priority=4),
        DocumentationSource("https://dev.mysql.com/doc/", "MySQL", "database", "mysql", priority=3),
        DocumentationSource("https://redis.io/docs/", "Redis", "database", "redis", priority=4),
        DocumentationSource("https://www.elastic.co/guide/index.html", "Elasticsearch", "database", "elasticsearch", priority=3),
        DocumentationSource("https://cassandra.apache.org/doc/stable/", "Apache Cassandra", "database", "cassandra", priority=2),
        DocumentationSource("https://neo4j.com/docs/", "Neo4j", "database", "neo4j", priority=2),
        DocumentationSource("https://docs.influxdata.com/influxdb/v2.7/", "InfluxDB", "database", "influxdb", priority=2),
    ]


def get_updated_ml_ai_tools() -> List[DocumentationSource]:
    """Updated ML/AI tool documentation sources."""
    return [
        DocumentationSource("https://pytorch.org/docs/stable/index.html", "PyTorch", "python", "pytorch", priority=4),
        DocumentationSource("https://www.tensorflow.org/guide", "TensorFlow", "python", "tensorflow", priority=4),
        DocumentationSource("https://scikit-learn.org/stable/user_guide.html", "Scikit-learn", "python", "sklearn", priority=3),
        # Hugging Face Transformers - use alternative URL
        DocumentationSource("https://huggingface.co/docs/transformers/index", "Hugging Face Transformers", "python", "transformers", priority=3),
        DocumentationSource("https://pandas.pydata.org/docs/", "Pandas", "python", "pandas", priority=4),
        DocumentationSource("https://numpy.org/doc/stable/", "NumPy", "python", "numpy", priority=3),
        DocumentationSource("https://matplotlib.org/stable/index.html", "Matplotlib", "python", "matplotlib", priority=2),
        DocumentationSource("https://docs.opencv.org/4.x/", "OpenCV", "python", "opencv", priority=2),
    ]


def get_updated_devtools() -> List[DocumentationSource]:
    """Updated development tools documentation sources."""
    return [
        DocumentationSource("https://docs.github.com/en", "GitHub", "devtools", "github", priority=4),
        DocumentationSource("https://docs.gitlab.com/", "GitLab", "devtools", "gitlab", priority=3),
        DocumentationSource("https://docs.docker.com/compose/", "Docker Compose", "devtools", "docker-compose", priority=3),
        DocumentationSource("https://nginx.org/en/docs/", "Nginx", "devtools", "nginx", priority=3),
        DocumentationSource("https://httpd.apache.org/docs/2.4/", "Apache HTTP Server", "devtools", "apache", priority=2),
        DocumentationSource("https://www.jenkins.io/doc/", "Jenkins", "devtools", "jenkins", priority=3),
        DocumentationSource("https://docs.sonarsource.com/sonarqube/latest/", "SonarQube", "devtools", "sonarqube", priority=2),
        # Jest moved URL structure
        DocumentationSource("https://jestjs.io/", "Jest", "javascript", "jest", priority=3),
        DocumentationSource("https://mochajs.org/", "Mocha", "javascript", "mocha", priority=2),
        # Cypress docs moved
        DocumentationSource("https://docs.cypress.io/guides/overview/why-cypress", "Cypress", "javascript", "cypress", priority=3),
    ]


def get_updated_mobile_frameworks() -> List[DocumentationSource]:
    """Updated mobile framework documentation sources.""" 
    return [
        # React Native docs moved
        DocumentationSource("https://reactnative.dev/", "React Native", "javascript", "react-native", priority=4),
        DocumentationSource("https://docs.flutter.dev/", "Flutter", "dart", "flutter", priority=4),
        DocumentationSource("https://ionicframework.com/docs", "Ionic", "javascript", "ionic", priority=3),
        DocumentationSource("https://docs.expo.dev/", "Expo", "javascript", "expo", priority=3),
        DocumentationSource("https://developer.android.com/guide", "Android", "kotlin", "android", priority=4),
        DocumentationSource("https://developer.apple.com/documentation/", "iOS", "swift", "ios", priority=3),
    ]


def get_updated_additional_sources() -> List[DocumentationSource]:
    """Additional updated documentation sources."""
    return [
        DocumentationSource("https://www.electronjs.org/docs/latest/", "Electron", "javascript", "electron", priority=2),
        DocumentationSource("https://tauri.app/v1/guides/", "Tauri", "rust", "tauri", priority=2),
        DocumentationSource("https://www.solidjs.com/docs/latest", "Solid.js", "javascript", "solid", priority=2),
        DocumentationSource("https://qwik.builder.io/docs/", "Qwik", "javascript", "qwik", priority=2),
        DocumentationSource("https://lit.dev/docs/", "Lit", "javascript", "lit", priority=2),
        DocumentationSource("https://stenciljs.com/docs/introduction", "Stencil", "javascript", "stencil", priority=1),
        DocumentationSource("https://docs.astro.build/en/getting-started/", "Astro", "javascript", "astro", priority=3),
        DocumentationSource("https://vitejs.dev/guide/", "Vite", "javascript", "vite", priority=3),
        DocumentationSource("https://webpack.js.org/guides/", "Webpack", "javascript", "webpack", priority=2),
        DocumentationSource("https://parceljs.org/docs/", "Parcel", "javascript", "parcel", priority=1),
        # Rollup guide moved
        DocumentationSource("https://rollupjs.org/introduction/", "Rollup", "javascript", "rollup", priority=2),
        DocumentationSource("https://esbuild.github.io/", "esbuild", "javascript", "esbuild", priority=2),
    ]


def get_comprehensive_updated_sources() -> List[DocumentationSource]:
    """Get all updated documentation sources combined."""
    all_sources = (
        get_updated_web_frameworks() + 
        get_updated_python_frameworks() + 
        get_updated_backend_frameworks() +
        get_updated_cloud_platforms() + 
        get_updated_databases() + 
        get_updated_ml_ai_tools() + 
        get_updated_devtools() + 
        get_updated_mobile_frameworks() + 
        get_updated_additional_sources()
    )
    
    return all_sources


# Convenience functions for quick access
def process_updated_popular_frameworks(
    output_dir: str = "rules/frameworks_updated",
    bedrock_config=None
):
    """Process popular frameworks with updated URLs."""
    from ..batch_processor import MLBatchProcessor
    
    sources = get_updated_web_frameworks() + get_updated_python_frameworks()
    
    processor = MLBatchProcessor(
        bedrock_config=bedrock_config,
        output_dir=output_dir,
        max_concurrent=8
    )
    
    return processor.process_documentation_batch(sources)


def process_updated_comprehensive_batch(
    output_dir: str = "rules/comprehensive_updated",
    bedrock_config=None
):
    """Process comprehensive batch with updated URLs."""
    from ..batch_processor import MLBatchProcessor
    
    sources = get_comprehensive_updated_sources()
    
    processor = MLBatchProcessor(
        bedrock_config=bedrock_config,
        output_dir=output_dir,
        max_concurrent=12
    )
    
    return processor.process_documentation_batch(sources)