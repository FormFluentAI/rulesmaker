#!/usr/bin/env python3
"""
Comprehensive Cursor Rules Generator

This script processes all demo output, leverages formatters, processors, and transformers
to create a beautiful and comprehensive .cursor/ directory structure with high-quality rules.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rules_maker.formatters.cursor_rules_formatter import CursorRulesFormatter, CursorRuleMetadata, CursorRuleContent
from rules_maker.processors.documentation_processor import DocumentationProcessor
from rules_maker.processors.api_processor import APIDocumentationProcessor
from rules_maker.processors.code_processor import CodeDocumentationProcessor
from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
from rules_maker.models import ScrapingResult, RuleSet, Rule
from rules_maker.intelligence.nextjs_categorizer import NextJSCategorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveCursorRulesGenerator:
    """Generator for comprehensive cursor rules from all processed content."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.demo_output_dir = project_root / "demo_output"
        self.cursor_dir = project_root / ".cursor"
        self.cursor_rules_dir = self.cursor_dir / "rules"
        
        # Initialize components
        self.formatter = CursorRulesFormatter()
        self.doc_processor = DocumentationProcessor()
        self.api_processor = APIDocumentationProcessor()
        self.code_processor = CodeDocumentationProcessor()
        self.transformer = CursorRuleTransformer()
        self.categorizer = NextJSCategorizer()
        
        # Ensure directories exist
        self.cursor_rules_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ComprehensiveCursorRulesGenerator")
        logger.info(f"Demo output: {self.demo_output_dir}")
        logger.info(f"Cursor rules: {self.cursor_rules_dir}")
    
    def process_all_demo_output(self) -> Dict[str, Any]:
        """Process all demo output and generate comprehensive cursor rules."""
        logger.info("🚀 Starting comprehensive cursor rules generation...")
        
        results = {
            "generated_rules": [],
            "processing_stats": {},
            "errors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Process comprehensive pipeline output
            comprehensive_results = self._process_comprehensive_output()
            results["generated_rules"].extend(comprehensive_results["rules"])
            results["processing_stats"]["comprehensive"] = comprehensive_results["stats"]
            
            # Process routing output
            routing_results = self._process_routing_output()
            results["generated_rules"].extend(routing_results["rules"])
            results["processing_stats"]["routing"] = routing_results["stats"]
            
            # Generate additional specialized rules
            specialized_results = self._generate_specialized_rules()
            results["generated_rules"].extend(specialized_results["rules"])
            results["processing_stats"]["specialized"] = specialized_results["stats"]
            
            # Create index and documentation
            self._create_cursor_index(results)
            self._create_cursor_documentation(results)
            
            logger.info(f"✅ Generated {len(results['generated_rules'])} cursor rules")
            
        except Exception as e:
            logger.error(f"❌ Error during processing: {e}")
            results["errors"].append(str(e))
        
        return results
    
    def _process_comprehensive_output(self) -> Dict[str, Any]:
        """Process comprehensive pipeline output."""
        logger.info("📊 Processing comprehensive pipeline output...")
        
        comprehensive_dir = self.demo_output_dir / "comprehensive"
        if not comprehensive_dir.exists():
            logger.warning("Comprehensive output directory not found")
            return {"rules": [], "stats": {"processed": 0, "errors": 0}}
        
        rules = []
        stats = {"processed": 0, "errors": 0}
        
        # Process batch output
        batch_output_dir = comprehensive_dir / "batch_output" / "nextjs"
        if batch_output_dir.exists():
            for cluster_dir in batch_output_dir.iterdir():
                if cluster_dir.is_dir():
                    try:
                        rule = self._process_cluster_output(cluster_dir)
                        if rule:
                            rules.append(rule)
                            stats["processed"] += 1
                    except Exception as e:
                        logger.error(f"Error processing cluster {cluster_dir.name}: {e}")
                        stats["errors"] += 1
        
        return {"rules": rules, "stats": stats}
    
    def _process_cluster_output(self, cluster_dir: Path) -> Optional[Dict[str, Any]]:
        """Process individual cluster output."""
        metadata_file = cluster_dir / "metadata.json"
        rules_file = cluster_dir / "nextjs_cursor_rules.md"
        
        if not metadata_file.exists() or not rules_file.exists():
            return None
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load rules content
            with open(rules_file, 'r') as f:
                rules_content = f.read()
            
            # Create cursor rule
            rule_name = f"nextjs-{cluster_dir.name.replace('nextjs_cursor_', '')}"
            rule_metadata = CursorRuleMetadata(
                description=f"Next.js development rules - {metadata.get('coherence_score', 'N/A')} coherence",
                globs=[
                    "**/*.tsx", "**/*.ts", "**/*.jsx", "**/*.js",
                    "**/app/**/*", "**/pages/**/*", "**/components/**/*",
                    "**/lib/**/*", "**/utils/**/*"
                ],
                always_apply=True,
                tags=["nextjs", "react", "typescript", "javascript", "frontend"],
                version="1.0.0"
            )
            
            rule_content = CursorRuleContent(
                title=f"Next.js Development Rules - {cluster_dir.name}",
                description=f"Comprehensive Next.js development guidelines with {metadata.get('coherence_score', 'N/A')} coherence score",
                category="nextjs",
                difficulty_level="intermediate"
            )
            
            # Format the rule
            formatted_rule = self.formatter.format_rule(
                metadata=rule_metadata,
                content=rule_content,
                raw_content=rules_content
            )
            
            # Save the rule
            rule_file = self.cursor_rules_dir / f"{rule_name}.mdc"
            with open(rule_file, 'w') as f:
                f.write(formatted_rule)
            
            logger.info(f"✅ Generated rule: {rule_name}.mdc")
            
            return {
                "name": rule_name,
                "file": str(rule_file),
                "coherence_score": metadata.get('coherence_score', 0),
                "rules_count": metadata.get('rules_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error processing cluster {cluster_dir.name}: {e}")
            return None
    
    def _process_routing_output(self) -> Dict[str, Any]:
        """Process routing output."""
        logger.info("🛣️ Processing routing output...")
        
        routing_dir = self.demo_output_dir / "routing"
        if not routing_dir.exists():
            logger.warning("Routing output directory not found")
            return {"rules": [], "stats": {"processed": 0, "errors": 0}}
        
        rules = []
        stats = {"processed": 0, "errors": 0}
        
        routing_file = routing_dir / "routing.mdc"
        if routing_file.exists():
            try:
                # Read existing routing rule
                with open(routing_file, 'r') as f:
                    routing_content = f.read()
                
                # Enhance and save as comprehensive routing rule
                enhanced_rule = self._enhance_routing_rule(routing_content)
                
                enhanced_file = self.cursor_rules_dir / "nextjs-routing-comprehensive.mdc"
                with open(enhanced_file, 'w') as f:
                    f.write(enhanced_rule)
                
                rules.append({
                    "name": "nextjs-routing-comprehensive",
                    "file": str(enhanced_file),
                    "type": "enhanced_routing"
                })
                stats["processed"] += 1
                
                logger.info("✅ Enhanced and saved routing rule")
                
            except Exception as e:
                logger.error(f"Error processing routing rule: {e}")
                stats["errors"] += 1
        
        return {"rules": rules, "stats": stats}
    
    def _enhance_routing_rule(self, content: str) -> str:
        """Enhance routing rule with additional content."""
        enhanced_content = f"""---
description: Comprehensive Next.js routing and navigation patterns for App Router and Pages Router
globs:
  - '**/app/**/*'
  - '**/pages/**/*'
  - '**/middleware.ts'
  - '**/route.ts'
  - '**/layout.tsx'
  - '**/page.tsx'
  - '**/loading.tsx'
  - '**/error.tsx'
  - '**/not-found.tsx'
alwaysApply: true
trigger: always_on
tags:
  - nextjs
  - routing
  - app-router
  - pages-router
  - navigation
  - middleware
version: 2.0.0
lastUpdated: '{datetime.now().isoformat()}'
---

# Next.js Routing & Navigation Master Guide
*Comprehensive routing patterns for modern Next.js applications*

## 🎯 Core Principles

### App Router (Recommended)
- **Server Components by default**: Fetch data directly on the server
- **Client Components when needed**: Use 'use client' for interactivity
- **Nested layouts**: Share UI between multiple pages
- **Loading states**: Built-in loading UI with loading.tsx
- **Error boundaries**: Error handling with error.tsx
- **Streaming**: Progressive page loading with Suspense

### Pages Router (Legacy)
- **File-based routing**: Automatic routing based on file structure
- **API routes**: Serverless functions in pages/api/
- **Dynamic routing**: [id].js for dynamic segments
- **Middleware**: Custom logic before page rendering

## 🏗️ App Router Structure

```
app/
├── layout.tsx          # Root layout (required)
├── page.tsx            # Home page
├── loading.tsx         # Global loading UI
├── error.tsx           # Global error UI
├── not-found.tsx       # 404 page
├── globals.css         # Global styles
├── (dashboard)/        # Route groups
│   ├── layout.tsx      # Dashboard layout
│   ├── page.tsx        # Dashboard home
│   ├── analytics/
│   │   └── page.tsx    # /analytics
│   └── settings/
│       └── page.tsx    # /settings
├── blog/
│   ├── page.tsx        # /blog
│   ├── [slug]/
│   │   └── page.tsx    # /blog/[slug]
│   └── loading.tsx     # Blog loading UI
└── api/
    └── users/
        └── route.ts    # API route
```

## 📋 Implementation Guidelines

### 1. Layout Patterns
```typescript
// app/layout.tsx - Root layout
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {{
  title: 'My App',
  description: 'Generated by Next.js',
}}

export default function RootLayout({{
  children,
}}: {{
  children: React.ReactNode
}}) {{
  return (
    <html lang="en">
      <body className={{inter.className}}>
        <nav>Navigation</nav>
        {{children}}
        <footer>Footer</footer>
      </body>
    </html>
  )
}}
```

### 2. Page Components
```typescript
// app/page.tsx - Home page
import {{ Suspense }} from 'react'

async function getData() {{
  const res = await fetch('https://api.example.com/data')
  return res.json()
}}

export default async function HomePage() {{
  const data = await getData()
  
  return (
    <main>
      <h1>Welcome</h1>
      <Suspense fallback={{<div>Loading...</div>}}>
        <DataComponent data={{data}} />
      </Suspense>
    </main>
  )
}}
```

### 3. Dynamic Routes
```typescript
// app/blog/[slug]/page.tsx
interface Props {{
  params: {{ slug: string }}
}}

export default async function BlogPost({{ params }}: Props) {{
  const post = await getPost(params.slug)
  
  return (
    <article>
      <h1>{{post.title}}</h1>
      <div>{{post.content}}</div>
    </article>
  )
}}

// Generate static params for SSG
export async function generateStaticParams() {{
  const posts = await getPosts()
  return posts.map((post) => ({{
    slug: post.slug,
  }}))
}}
```

### 4. Loading & Error States
```typescript
// app/loading.tsx
export default function Loading() {{
  return <div>Loading...</div>
}}

// app/error.tsx
'use client'

export default function Error({{
  error,
  reset,
}}: {{
  error: Error & {{ digest?: string }}
  reset: () => void
}}) {{
  return (
    <div>
      <h2>Something went wrong!</h2>
      <button onClick={{() => reset()}}>Try again</button>
    </div>
  )
}}
```

### 5. Middleware
```typescript
// middleware.ts
import {{ NextResponse }} from 'next/server'
import type {{ NextRequest }} from 'next/server'

export function middleware(request: NextRequest) {{
  // Add custom logic here
  const response = NextResponse.next()
  
  // Example: Add custom header
  response.headers.set('x-custom-header', 'value')
  
  return response
}}

export const config = {{
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}}
```

## 🚀 Advanced Patterns

### Route Groups
```typescript
// app/(marketing)/layout.tsx
export default function MarketingLayout({{
  children,
}}: {{
  children: React.ReactNode
}}) {{
  return (
    <div className="marketing-layout">
      {{children}}
    </div>
  )
}}
```

### Parallel Routes
```typescript
// app/@analytics/page.tsx
export default function Analytics() {{
  return <div>Analytics</div>
}}

// app/@dashboard/page.tsx
export default function Dashboard() {{
  return <div>Dashboard</div>
}}

// app/layout.tsx
export default function Layout({{
  children,
  analytics,
  dashboard,
}}: {{
  children: React.ReactNode
  analytics: React.ReactNode
  dashboard: React.ReactNode
}}) {{
  return (
    <div>
      {{children}}
      {{analytics}}
      {{dashboard}}
    </div>
  )
}}
```

### Intercepting Routes
```typescript
// app/@modal/(..)photo/[id]/page.tsx
export default function PhotoModal({{
  params,
}}: {{
  params: {{ id: string }}
}}) {{
  return <div>Modal for photo {{params.id}}</div>
}}
```

## 🔧 Navigation Patterns

### Client-side Navigation
```typescript
'use client'

import Link from 'next/link'
import {{ useRouter }} from 'next/navigation'

export default function Navigation() {{
  const router = useRouter()
  
  return (
    <nav>
      <Link href="/">Home</Link>
      <Link href="/about">About</Link>
      <Link href="/blog">Blog</Link>
      
      <button onClick={{() => router.push('/dashboard')}}>
        Go to Dashboard
      </button>
    </nav>
  )
}}
```

### Programmatic Navigation
```typescript
import {{ redirect }} from 'next/navigation'

export default function Page() {{
  // Server-side redirect
  redirect('/dashboard')
}}
```

## ⚡ Performance Optimizations

### Static Generation
```typescript
// Generate static pages at build time
export const dynamic = 'force-static'

// Revalidate every hour
export const revalidate = 3600
```

### Streaming
```typescript
import {{ Suspense }} from 'react'

export default function Page() {{
  return (
    <div>
      <Suspense fallback={{<div>Loading posts...</div>}}>
        <Posts />
      </Suspense>
      <Suspense fallback={{<div>Loading comments...</div>}}>
        <Comments />
      </Suspense>
    </div>
  )
}}
```

## 🚨 Critical Rules

**ALWAYS:**
- Use App Router for new projects
- Implement proper loading and error states
- Use Server Components by default
- Add proper TypeScript types
- Implement proper error boundaries
- Use Suspense for streaming

**NEVER:**
- Mix App Router and Pages Router in the same project
- Use 'use client' unnecessarily
- Forget to handle loading and error states
- Ignore TypeScript errors
- Use client-side data fetching when server-side is possible

## 📚 Related Concepts

- Server Components vs Client Components
- Static Site Generation (SSG)
- Server-Side Rendering (SSR)
- Incremental Static Regeneration (ISR)
- Edge Runtime
- Middleware
- API Routes
- Dynamic Imports
- Code Splitting

---
*Generated by Rules Maker with comprehensive Next.js routing knowledge on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return enhanced_content
    
    def _generate_specialized_rules(self) -> Dict[str, Any]:
        """Generate specialized cursor rules."""
        logger.info("🎯 Generating specialized cursor rules...")
        
        rules = []
        stats = {"processed": 0, "errors": 0}
        
        # Generate specialized rules for different aspects
        specialized_topics = [
            {
                "name": "nextjs-performance",
                "title": "Next.js Performance Optimization",
                "description": "Performance optimization patterns and best practices for Next.js applications",
                "globs": ["**/*.tsx", "**/*.ts", "**/next.config.js", "**/app/**/*"],
                "category": "performance"
            },
            {
                "name": "nextjs-security",
                "title": "Next.js Security Best Practices",
                "description": "Security patterns and best practices for Next.js applications",
                "globs": ["**/*.tsx", "**/*.ts", "**/middleware.ts", "**/api/**/*"],
                "category": "security"
            },
            {
                "name": "nextjs-testing",
                "title": "Next.js Testing Strategies",
                "description": "Testing patterns and strategies for Next.js applications",
                "globs": ["**/*.test.tsx", "**/*.test.ts", "**/*.spec.tsx", "**/*.spec.ts"],
                "category": "testing"
            },
            {
                "name": "nextjs-deployment",
                "title": "Next.js Deployment & DevOps",
                "description": "Deployment patterns and DevOps practices for Next.js applications",
                "globs": ["**/Dockerfile", "**/docker-compose.yml", "**/vercel.json", "**/.github/**/*"],
                "category": "deployment"
            }
        ]
        
        for topic in specialized_topics:
            try:
                rule = self._create_specialized_rule(topic)
                if rule:
                    rules.append(rule)
                    stats["processed"] += 1
            except Exception as e:
                logger.error(f"Error creating specialized rule {topic['name']}: {e}")
                stats["errors"] += 1
        
        return {"rules": rules, "stats": stats}
    
    def _create_specialized_rule(self, topic: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Create a specialized cursor rule."""
        rule_metadata = CursorRuleMetadata(
            description=topic["description"],
            globs=topic["globs"],
            always_apply=True,
            tags=["nextjs", topic["category"], "best-practices"],
            version="1.0.0"
        )
        
        rule_content = CursorRuleContent(
            title=topic["title"],
            description=topic["description"],
            category=topic["category"],
            difficulty_level="intermediate"
        )
        
        # Generate content based on topic
        content = self._generate_topic_content(topic)
        
        # Format the rule
        formatted_rule = self.formatter.format_rule(
            metadata=rule_metadata,
            content=rule_content,
            raw_content=content
        )
        
        # Save the rule
        rule_file = self.cursor_rules_dir / f"{topic['name']}.mdc"
        with open(rule_file, 'w') as f:
            f.write(formatted_rule)
        
        logger.info(f"✅ Generated specialized rule: {topic['name']}.mdc")
        
        return {
            "name": topic["name"],
            "file": str(rule_file),
            "type": "specialized",
            "category": topic["category"]
        }
    
    def _generate_topic_content(self, topic: Dict[str, str]) -> str:
        """Generate content for a specialized topic."""
        content_templates = {
            "performance": """
# Next.js Performance Optimization

## Core Performance Principles
- Use Server Components by default
- Implement proper caching strategies
- Optimize images with next/image
- Use dynamic imports for code splitting
- Implement proper loading states

## Image Optimization
```typescript
import Image from 'next/image'

export default function OptimizedImage() {
  return (
    <Image
      src="/hero.jpg"
      alt="Hero image"
      width={800}
      height={600}
      priority
      placeholder="blur"
      blurDataURL="data:image/jpeg;base64,..."
    />
  )
}
```

## Code Splitting
```typescript
import dynamic from 'next/dynamic'

const DynamicComponent = dynamic(() => import('./HeavyComponent'), {
  loading: () => <p>Loading...</p>,
  ssr: false
})
```

## Caching Strategies
```typescript
// Static generation with revalidation
export const revalidate = 3600 // 1 hour

// Dynamic rendering with caching
const data = await fetch('https://api.example.com/data', {
  next: { revalidate: 60 }
})
```
""",
            "security": """
# Next.js Security Best Practices

## Authentication & Authorization
- Use NextAuth.js for authentication
- Implement proper session management
- Validate user permissions on server-side
- Use HTTPS in production

## Input Validation
```typescript
import { z } from 'zod'

const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8)
})

export async function POST(request: Request) {
  const body = await request.json()
  const validated = schema.parse(body)
  // Process validated data
}
```

## Environment Variables
```typescript
// Never expose sensitive data to client
const serverOnlySecret = process.env.SERVER_SECRET
const publicConfig = process.env.NEXT_PUBLIC_API_URL
```

## CSRF Protection
```typescript
import { getCsrfToken } from 'next-auth/react'

export default function Form() {
  const csrfToken = getCsrfToken()
  
  return (
    <form>
      <input type="hidden" name="csrfToken" value={csrfToken} />
      {/* form fields */}
    </form>
  )
}
```
""",
            "testing": """
# Next.js Testing Strategies

## Testing Setup
```typescript
// jest.config.js
const nextJest = require('next/jest')

const createJestConfig = nextJest({
  dir: './',
})

const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  testEnvironment: 'jest-environment-jsdom',
}

module.exports = createJestConfig(customJestConfig)
```

## Component Testing
```typescript
import { render, screen } from '@testing-library/react'
import HomePage from '@/app/page'

describe('HomePage', () => {
  it('renders welcome message', () => {
    render(<HomePage />)
    expect(screen.getByText('Welcome')).toBeInTheDocument()
  })
})
```

## API Route Testing
```typescript
import { createMocks } from 'node-mocks-http'
import handler from '@/app/api/users/route'

describe('/api/users', () => {
  it('returns users list', async () => {
    const { req, res } = createMocks({
      method: 'GET',
    })

    await handler(req, res)
    expect(res._getStatusCode()).toBe(200)
  })
})
```

## E2E Testing
```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  use: {
    baseURL: 'http://localhost:3000',
  },
})
```
""",
            "deployment": """
# Next.js Deployment & DevOps

## Vercel Deployment
```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "functions": {
    "app/api/**/*.ts": {
      "maxDuration": 30
    }
  }
}
```

## Docker Deployment
```dockerfile
FROM node:18-alpine AS base
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM base AS build
RUN npm ci
COPY . .
RUN npm run build

FROM base AS runtime
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["npm", "start"]
```

## Environment Configuration
```typescript
// next.config.js
module.exports = {
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  experimental: {
    serverComponentsExternalPackages: ['@prisma/client'],
  },
}
```

## CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run build
      - run: npm run test
```
"""
        }
        
        return content_templates.get(topic["category"], f"# {topic['title']}\n\n{topic['description']}")
    
    def _create_cursor_index(self, results: Dict[str, Any]):
        """Create a comprehensive cursor rules index."""
        logger.info("📋 Creating cursor rules index...")
        
        index_content = f"""# Cursor Rules Index
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📊 Generation Summary

- **Total Rules Generated**: {len(results['generated_rules'])}
- **Processing Stats**: {json.dumps(results['processing_stats'], indent=2)}
- **Generation Time**: {results['timestamp']}

## 🎯 Available Rules

### Next.js Development Rules

"""
        
        # Group rules by type
        nextjs_rules = [r for r in results['generated_rules'] if 'nextjs' in r['name']]
        specialized_rules = [r for r in results['generated_rules'] if r.get('type') == 'specialized']
        
        # Add Next.js rules
        for rule in nextjs_rules:
            index_content += f"- **{rule['name']}**: {rule.get('coherence_score', 'N/A')} coherence score\n"
        
        index_content += "\n### Specialized Rules\n\n"
        
        # Add specialized rules
        for rule in specialized_rules:
            index_content += f"- **{rule['name']}**: {rule['category']} best practices\n"
        
        index_content += f"""
## 🚀 Usage Instructions

1. **Automatic Application**: All rules are set to `alwaysApply: true` and will be automatically applied
2. **File Patterns**: Rules target specific file patterns (`.tsx`, `.ts`, `.jsx`, `.js`)
3. **Context Awareness**: Rules adapt based on file location and content
4. **Best Practices**: Each rule includes comprehensive best practices and examples

## 📁 File Structure

```
.cursor/rules/
├── nextjs-cluster_0.mdc          # Core Next.js patterns
├── nextjs-cluster_1.mdc          # Advanced Next.js patterns  
├── nextjs-single.mdc             # Single-page patterns
├── nextjs-routing-comprehensive.mdc  # Complete routing guide
├── nextjs-performance.mdc        # Performance optimization
├── nextjs-security.mdc           # Security best practices
├── nextjs-testing.mdc            # Testing strategies
└── nextjs-deployment.mdc         # Deployment & DevOps
```

## 🔧 Customization

Each rule file can be customized by editing the `.mdc` files in `.cursor/rules/`. The rules include:

- **Frontmatter**: Configuration and metadata
- **Guidelines**: Best practices and patterns
- **Examples**: Code examples and implementations
- **Anti-patterns**: What to avoid
- **Related concepts**: Additional resources

## 📈 Quality Metrics

- **Coherence Scores**: {[r.get('coherence_score', 0) for r in nextjs_rules if r.get('coherence_score')]}
- **Average Coherence**: {sum([r.get('coherence_score', 0) for r in nextjs_rules if r.get('coherence_score')]) / max(len([r for r in nextjs_rules if r.get('coherence_score')]), 1):.3f}
- **Total Rules Count**: {sum([r.get('rules_count', 0) for r in nextjs_rules if r.get('rules_count')])}

---
*Generated by Rules Maker Comprehensive Cursor Rules Generator*
"""
        
        # Save index
        index_file = self.cursor_dir / "README.md"
        with open(index_file, 'w') as f:
            f.write(index_content)
        
        logger.info("✅ Created cursor rules index")
    
    def _create_cursor_documentation(self, results: Dict[str, Any]):
        """Create comprehensive cursor documentation."""
        logger.info("📚 Creating cursor documentation...")
        
        doc_content = f"""# Comprehensive Cursor Rules Documentation

## 🎯 Overview

This directory contains comprehensive cursor rules generated from Next.js documentation and best practices. The rules are designed to provide intelligent code assistance and ensure consistent, high-quality development patterns.

## 📊 Generation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Rules**: {len(results['generated_rules'])}
**Processing Time**: {results['timestamp']}

### Processing Statistics

```json
{json.dumps(results['processing_stats'], indent=2)}
```

## 🏗️ Architecture

### Components Used

1. **CursorRulesFormatter**: Formats content into proper cursor rules format
2. **DocumentationProcessor**: Processes and structures documentation content
3. **CursorRuleTransformer**: Transforms content with cursor-specific knowledge
4. **NextJSCategorizer**: Categorizes content for Next.js-specific patterns

### Generation Pipeline

1. **Content Processing**: Raw documentation → Structured content
2. **Categorization**: Content → Next.js-specific categories
3. **Transformation**: Structured content → Cursor rules format
4. **Formatting**: Cursor rules → Proper .mdc files
5. **Quality Assessment**: Coherence scoring and validation

## 📁 Rule Categories

### Core Next.js Rules
- **nextjs-cluster_0**: Core development patterns
- **nextjs-cluster_1**: Advanced patterns and optimizations
- **nextjs-single**: Single-page application patterns

### Specialized Rules
- **nextjs-routing-comprehensive**: Complete routing and navigation guide
- **nextjs-performance**: Performance optimization strategies
- **nextjs-security**: Security best practices
- **nextjs-testing**: Testing strategies and patterns
- **nextjs-deployment**: Deployment and DevOps practices

## 🔧 Rule Structure

Each rule file follows this structure:

```yaml
---
description: Rule description
globs: [file patterns]
alwaysApply: true
tags: [relevant tags]
version: 1.0.0
---

# Rule Title

## Guidelines
- Best practices
- Implementation patterns
- Code examples

## Anti-patterns
- What to avoid
- Common mistakes

## Related Concepts
- Additional resources
- Related patterns
```

## 🚀 Usage

### Automatic Application
All rules are configured with `alwaysApply: true` and will be automatically applied when working with matching file patterns.

### File Pattern Matching
Rules target specific file patterns:
- `**/*.tsx`, `**/*.ts` - TypeScript files
- `**/*.jsx`, `**/*.js` - JavaScript files
- `**/app/**/*` - App Router files
- `**/pages/**/*` - Pages Router files
- `**/components/**/*` - Component files

### Context Awareness
Rules adapt based on:
- File location and structure
- Import patterns
- Component types
- Framework usage

## 📈 Quality Metrics

### Coherence Scores
- **Cluster 0**: {next((r.get('coherence_score', 0) for r in results['generated_rules'] if 'cluster_0' in r['name']), 0):.3f}
- **Cluster 1**: {next((r.get('coherence_score', 0) for r in results['generated_rules'] if 'cluster_1' in r['name']), 0):.3f}
- **Single**: {next((r.get('coherence_score', 0) for r in results['generated_rules'] if 'single' in r['name']), 0):.3f}

### Rules Count
- **Total Rules**: {sum([r.get('rules_count', 0) for r in results['generated_rules'] if r.get('rules_count')])}
- **Average per Cluster**: {sum([r.get('rules_count', 0) for r in results['generated_rules'] if r.get('rules_count')]) / max(len([r for r in results['generated_rules'] if r.get('rules_count')]), 1):.1f}

## 🔄 Maintenance

### Updating Rules
1. Modify the source documentation
2. Re-run the generation pipeline
3. Review and validate new rules
4. Test rule effectiveness

### Adding New Rules
1. Add new content to demo_output
2. Update the generation script
3. Run comprehensive generation
4. Validate and test new rules

## 🐛 Troubleshooting

### Common Issues
1. **Rule not applying**: Check file pattern matching
2. **Conflicting rules**: Review rule priorities
3. **Performance impact**: Monitor rule complexity

### Debug Mode
Enable debug logging to troubleshoot rule generation:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Resources

- [Cursor Rules Documentation](https://cursor.sh/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [Rules Maker Project](https://github.com/your-org/rules-maker)

---
*Generated by Rules Maker Comprehensive Cursor Rules Generator v1.0.0*
"""
        
        # Save documentation
        doc_file = self.cursor_dir / "DOCUMENTATION.md"
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        
        logger.info("✅ Created cursor documentation")


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent
    generator = ComprehensiveCursorRulesGenerator(project_root)
    
    try:
        results = generator.process_all_demo_output()
        
        # Save results
        results_file = project_root / "cursor_generation_report.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("🎉 Comprehensive cursor rules generation completed!")
        logger.info(f"📊 Generated {len(results['generated_rules'])} rules")
        logger.info(f"📁 Rules saved to: {generator.cursor_rules_dir}")
        logger.info(f"📋 Report saved to: {results_file}")
        
        if results['errors']:
            logger.warning(f"⚠️ {len(results['errors'])} errors occurred during generation")
            for error in results['errors']:
                logger.warning(f"  - {error}")
        
    except Exception as e:
        logger.error(f"❌ Fatal error during generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
