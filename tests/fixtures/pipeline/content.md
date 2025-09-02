# Project Docs: Async Processing and Logging

This project heavily uses asynchronous patterns in Python with asyncio.  
We rely on aiohttp for network IO and async file operations.

Key topics:
- Async Pattern: non-blocking IO with await
- Structured logging with context fields
- Testing with pytest and async fixtures

Anti-patterns to avoid:
- Blocking calls in event loop
- Excessive global state in utilities

Best practices:
- Use INFO for milestones, DEBUG for detailed traces
- Keep functions pure and side-effect free when possible

