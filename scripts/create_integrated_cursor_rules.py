#!/usr/bin/env python3
"""
Integrated Cursor Rules Generator

Uses the cursor transformer with integrated formatter to create properly formatted cursor rules.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rules_maker.transformers.cursor_transformer import CursorRuleTransformer
from rules_maker.models import ScrapingResult, RuleSet, Rule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedCursorRulesGenerator:
    """Generator that uses the integrated cursor transformer with formatter."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cursor_dir = project_root / ".cursor"
        self.cursor_rules_dir = self.cursor_dir / "rules"
        
        # Initialize the cursor transformer with integrated formatter
        self.transformer = CursorRuleTransformer()
        
        # Ensure directories exist
        self.cursor_rules_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized IntegratedCursorRulesGenerator with formatter")
        logger.info(f"Cursor rules: {self.cursor_rules_dir}")
    
    def create_comprehensive_rules(self) -> Dict[str, Any]:
        """Create comprehensive cursor rules using the integrated transformer and formatter."""
        logger.info("🚀 Creating comprehensive cursor rules with integrated formatter...")
        
        results = {
            "generated_rules": [],
            "processing_stats": {},
            "errors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Create comprehensive Next.js routing rule
            routing_rule = self._create_routing_rule()
            results["generated_rules"].append(routing_rule)
            
            # Create performance optimization rule
            performance_rule = self._create_performance_rule()
            results["generated_rules"].append(performance_rule)
            
            # Create security rule
            security_rule = self._create_security_rule()
            results["generated_rules"].append(security_rule)
            
            # Create testing rule
            testing_rule = self._create_testing_rule()
            results["generated_rules"].append(testing_rule)
            
            # Create deployment rule
            deployment_rule = self._create_deployment_rule()
            results["generated_rules"].append(deployment_rule)
            
            # Create index and documentation
            self._create_cursor_index(results)
            
            logger.info(f"✅ Generated {len(results['generated_rules'])} cursor rules with integrated formatter")
            
        except Exception as e:
            logger.error(f"❌ Error during processing: {e}")
            results["errors"].append(str(e))
        
        return results
    
    def _create_routing_rule(self) -> Dict[str, Any]:
        """Create comprehensive routing rule using the integrated formatter."""
        content = """# Next.js Routing & Navigation Master Guide

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

export const metadata = {
  title: 'My App',
  description: 'Generated by Next.js',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav>Navigation</nav>
        {children}
        <footer>Footer</footer>
      </body>
    </html>
  )
}
```

### 2. Page Components
```typescript
// app/page.tsx - Home page
import { Suspense } from 'react'

async function getData() {
  const res = await fetch('https://api.example.com/data')
  return res.json()
}

export default async function HomePage() {
  const data = await getData()
  
  return (
    <main>
      <h1>Welcome</h1>
      <Suspense fallback={<div>Loading...</div>}>
        <DataComponent data={data} />
      </Suspense>
    </main>
  )
}
```

### 3. Dynamic Routes
```typescript
// app/blog/[slug]/page.tsx
interface Props {
  params: { slug: string }
}

export default async function BlogPost({ params }: Props) {
  const post = await getPost(params.slug)
  
  return (
    <article>
      <h1>{post.title}</h1>
      <div>{post.content}</div>
    </article>
  )
}

// Generate static params for SSG
export async function generateStaticParams() {
  const posts = await getPosts()
  return posts.map((post) => ({
    slug: post.slug,
  }))
}
```

### 4. Loading & Error States
```typescript
// app/loading.tsx
export default function Loading() {
  return <div>Loading...</div>
}

// app/error.tsx
'use client'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div>
      <h2>Something went wrong!</h2>
      <button onClick={() => reset()}>Try again</button>
    </div>
  )
}
```

### 5. Middleware
```typescript
// middleware.ts
import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  // Add custom logic here
  const response = NextResponse.next()
  
  // Example: Add custom header
  response.headers.set('x-custom-header', 'value')
  
  return response
}

export const config = {
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}
```

## 🚀 Advanced Patterns

### Route Groups
```typescript
// app/(marketing)/layout.tsx
export default function MarketingLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="marketing-layout">
      {children}
    </div>
  )
}
```

### Parallel Routes
```typescript
// app/@analytics/page.tsx
export default function Analytics() {
  return <div>Analytics</div>
}

// app/@dashboard/page.tsx
export default function Dashboard() {
  return <div>Dashboard</div>
}

// app/layout.tsx
export default function Layout({
  children,
  analytics,
  dashboard,
}: {
  children: React.ReactNode
  analytics: React.ReactNode
  dashboard: React.ReactNode
}) {
  return (
    <div>
      {children}
      {analytics}
      {dashboard}
    </div>
  )
}
```

### Intercepting Routes
```typescript
// app/@modal/(..)photo/[id]/page.tsx
export default function PhotoModal({
  params,
}: {
  params: { id: string }
}) {
  return <div>Modal for photo {params.id}</div>
}
```

## 🔧 Navigation Patterns

### Client-side Navigation
```typescript
'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'

export default function Navigation() {
  const router = useRouter()
  
  return (
    <nav>
      <Link href="/">Home</Link>
      <Link href="/about">About</Link>
      <Link href="/blog">Blog</Link>
      
      <button onClick={() => router.push('/dashboard')}>
        Go to Dashboard
      </button>
    </nav>
  )
}
```

### Programmatic Navigation
```typescript
import { redirect } from 'next/navigation'

export default function Page() {
  // Server-side redirect
  redirect('/dashboard')
}
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
import { Suspense } from 'react'

export default function Page() {
  return (
    <div>
      <Suspense fallback={<div>Loading posts...</div>}>
        <Posts />
      </Suspense>
      <Suspense fallback={<div>Loading comments...</div>}>
        <Comments />
      </Suspense>
    </div>
  )
}
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
        
        # Use the integrated formatter
        formatted_rule = self.transformer.create_cursor_rule_with_formatter(
            name="nextjs-routing-comprehensive",
            description="Comprehensive Next.js routing and navigation patterns for App Router and Pages Router",
            content=content,
            globs=[
                "**/app/**/*",
                "**/pages/**/*",
                "**/middleware.ts",
                "**/route.ts",
                "**/layout.tsx",
                "**/page.tsx",
                "**/loading.tsx",
                "**/error.tsx",
                "**/not-found.tsx"
            ],
            tags=["nextjs", "routing", "app-router", "pages-router", "navigation", "middleware"]
        )
        
        # Save the rule
        rule_file = self.cursor_rules_dir / "nextjs-routing-comprehensive.mdc"
        with open(rule_file, 'w') as f:
            f.write(formatted_rule)
        
        logger.info("✅ Created comprehensive routing rule with integrated formatter")
        
        return {
            "name": "nextjs-routing-comprehensive",
            "file": str(rule_file),
            "type": "comprehensive_routing"
        }
    
    def _create_performance_rule(self) -> Dict[str, Any]:
        """Create performance optimization rule using the integrated formatter."""
        content = """# Next.js Performance Optimization

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

## Bundle Analysis
```bash
npm install --save-dev @next/bundle-analyzer
```

```typescript
// next.config.js
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
})

module.exports = withBundleAnalyzer({
  // your Next.js config
})
```

## Performance Monitoring
```typescript
// app/layout.tsx
import { SpeedInsights } from '@vercel/speed-insights/next'

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        {children}
        <SpeedInsights />
      </body>
    </html>
  )
}
```

## 🚨 Critical Performance Rules

**ALWAYS:**
- Use next/image for all images
- Implement proper loading states
- Use Server Components by default
- Enable compression and caching
- Monitor Core Web Vitals

**NEVER:**
- Load unnecessary JavaScript on the client
- Use large images without optimization
- Block rendering with synchronous operations
- Ignore bundle size analysis
- Skip performance monitoring

---
*Generated by Rules Maker with Next.js performance optimization knowledge*
"""
        
        # Use the integrated formatter
        formatted_rule = self.transformer.create_cursor_rule_with_formatter(
            name="nextjs-performance",
            description="Next.js performance optimization patterns and best practices",
            content=content,
            globs=["**/*.tsx", "**/*.ts", "**/next.config.js", "**/app/**/*"],
            tags=["nextjs", "performance", "optimization", "caching", "images"]
        )
        
        # Save the rule
        rule_file = self.cursor_rules_dir / "nextjs-performance.mdc"
        with open(rule_file, 'w') as f:
            f.write(formatted_rule)
        
        logger.info("✅ Created performance optimization rule with integrated formatter")
        
        return {
            "name": "nextjs-performance",
            "file": str(rule_file),
            "type": "performance"
        }
    
    def _create_security_rule(self) -> Dict[str, Any]:
        """Create security rule using the integrated formatter."""
        content = """# Next.js Security Best Practices

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

## Content Security Policy
```typescript
// next.config.js
module.exports = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline';"
          }
        ]
      }
    ]
  }
}
```

## API Route Security
```typescript
// app/api/protected/route.ts
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'

export async function GET() {
  const session = await getServerSession(authOptions)
  
  if (!session) {
    return new Response('Unauthorized', { status: 401 })
  }
  
  // Protected logic here
}
```

## 🚨 Critical Security Rules

**ALWAYS:**
- Validate all user inputs
- Use HTTPS in production
- Implement proper authentication
- Sanitize data before database operations
- Use environment variables for secrets

**NEVER:**
- Expose sensitive data to the client
- Trust user input without validation
- Store secrets in client-side code
- Skip authentication checks
- Use weak session management

---
*Generated by Rules Maker with Next.js security best practices*
"""
        
        # Use the integrated formatter
        formatted_rule = self.transformer.create_cursor_rule_with_formatter(
            name="nextjs-security",
            description="Next.js security patterns and best practices",
            content=content,
            globs=["**/*.tsx", "**/*.ts", "**/middleware.ts", "**/api/**/*"],
            tags=["nextjs", "security", "authentication", "validation", "csp"]
        )
        
        # Save the rule
        rule_file = self.cursor_rules_dir / "nextjs-security.mdc"
        with open(rule_file, 'w') as f:
            f.write(formatted_rule)
        
        logger.info("✅ Created security best practices rule with integrated formatter")
        
        return {
            "name": "nextjs-security",
            "file": str(rule_file),
            "type": "security"
        }
    
    def _create_testing_rule(self) -> Dict[str, Any]:
        """Create testing rule using the integrated formatter."""
        content = """# Next.js Testing Strategies

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

## Server Component Testing
```typescript
import { render } from '@testing-library/react'
import ServerComponent from '@/app/server-component'

// Mock fetch for server components
global.fetch = jest.fn(() =>
  Promise.resolve({
    json: () => Promise.resolve({ data: 'test' }),
  })
)

describe('ServerComponent', () => {
  it('renders server component', async () => {
    const component = await ServerComponent()
    const { container } = render(component)
    expect(container).toBeInTheDocument()
  })
})
```

## 🚨 Critical Testing Rules

**ALWAYS:**
- Test user interactions and workflows
- Mock external dependencies
- Test error states and edge cases
- Use proper test data setup
- Test accessibility features

**NEVER:**
- Test implementation details
- Skip error handling tests
- Use real API calls in tests
- Ignore test coverage
- Test third-party libraries

---
*Generated by Rules Maker with Next.js testing strategies*
"""
        
        # Use the integrated formatter
        formatted_rule = self.transformer.create_cursor_rule_with_formatter(
            name="nextjs-testing",
            description="Next.js testing patterns and strategies",
            content=content,
            globs=["**/*.test.tsx", "**/*.test.ts", "**/*.spec.tsx", "**/*.spec.ts"],
            tags=["nextjs", "testing", "jest", "playwright", "testing-library"]
        )
        
        # Save the rule
        rule_file = self.cursor_rules_dir / "nextjs-testing.mdc"
        with open(rule_file, 'w') as f:
            f.write(formatted_rule)
        
        logger.info("✅ Created testing strategies rule with integrated formatter")
        
        return {
            "name": "nextjs-testing",
            "file": str(rule_file),
            "type": "testing"
        }
    
    def _create_deployment_rule(self) -> Dict[str, Any]:
        """Create deployment rule using the integrated formatter."""
        content = """# Next.js Deployment & DevOps

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

## Static Export
```typescript
// next.config.js
module.exports = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  }
}
```

## 🚨 Critical Deployment Rules

**ALWAYS:**
- Use environment variables for configuration
- Test builds before deployment
- Monitor application performance
- Use proper caching strategies
- Implement proper error handling

**NEVER:**
- Deploy without testing
- Expose sensitive data in builds
- Skip performance monitoring
- Ignore build warnings
- Deploy to production without staging

---
*Generated by Rules Maker with Next.js deployment best practices*
"""
        
        # Use the integrated formatter
        formatted_rule = self.transformer.create_cursor_rule_with_formatter(
            name="nextjs-deployment",
            description="Next.js deployment patterns and DevOps practices",
            content=content,
            globs=["**/Dockerfile", "**/docker-compose.yml", "**/vercel.json", "**/.github/**/*"],
            tags=["nextjs", "deployment", "docker", "vercel", "ci-cd"]
        )
        
        # Save the rule
        rule_file = self.cursor_rules_dir / "nextjs-deployment.mdc"
        with open(rule_file, 'w') as f:
            f.write(formatted_rule)
        
        logger.info("✅ Created deployment & DevOps rule with integrated formatter")
        
        return {
            "name": "nextjs-deployment",
            "file": str(rule_file),
            "type": "deployment"
        }
    
    def _create_cursor_index(self, results: Dict[str, Any]):
        """Create a comprehensive cursor rules index."""
        logger.info("📋 Creating cursor rules index...")
        
        index_content = f"""# Cursor Rules Index
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 📊 Generation Summary

- **Total Rules Generated**: {len(results['generated_rules'])}
- **Generation Time**: {results['timestamp']}
- **Formatter Used**: Integrated CursorRulesFormatter

## 🎯 Available Rules

### Next.js Development Rules

"""
        
        # Add rules
        for rule in results['generated_rules']:
            index_content += f"- **{rule['name']}**: {rule['type']} best practices\n"
        
        index_content += f"""
## 🚀 Usage Instructions

1. **Automatic Application**: All rules are set to `alwaysApply: true` and will be automatically applied
2. **File Patterns**: Rules target specific file patterns (`.tsx`, `.ts`, `.jsx`, `.js`)
3. **Context Awareness**: Rules adapt based on file location and content
4. **Best Practices**: Each rule includes comprehensive best practices and examples

## 📁 File Structure

```
.cursor/rules/
├── nextjs-routing-comprehensive.mdc  # Complete routing guide
├── nextjs-performance.mdc            # Performance optimization
├── nextjs-security.mdc               # Security best practices
├── nextjs-testing.mdc                # Testing strategies
└── nextjs-deployment.mdc             # Deployment & DevOps
```

## 🔧 Customization

Each rule file can be customized by editing the `.mdc` files in `.cursor/rules/`. The rules include:

- **Frontmatter**: Configuration and metadata
- **Guidelines**: Best practices and patterns
- **Examples**: Code examples and implementations
- **Anti-patterns**: What to avoid
- **Related concepts**: Additional resources

## 🎯 Integration Features

- **Integrated Formatter**: Uses CursorRulesFormatter for proper YAML formatting
- **Proper Globs**: Correct YAML list syntax for file pattern matching
- **Comprehensive Content**: Rich content with examples and best practices
- **Metadata**: Proper versioning and timestamps

---
*Generated by Rules Maker Integrated Cursor Rules Generator with CursorRulesFormatter*
"""
        
        # Save index
        index_file = self.cursor_dir / "README.md"
        with open(index_file, 'w') as f:
            f.write(index_content)
        
        logger.info("✅ Created cursor rules index")


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent
    generator = IntegratedCursorRulesGenerator(project_root)
    
    try:
        results = generator.create_comprehensive_rules()
        
        # Save results
        results_file = project_root / "integrated_cursor_generation_report.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("🎉 Integrated cursor rules generation completed!")
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
