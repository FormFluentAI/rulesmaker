# Python Windsurf - Cluster 6

**Technology:** Python
**Format:** Windsurf
**Coherence Score:** 0.750
**Rules Count:** 1
**Keywords:** error, test

---

# Windsurf Workflow Rules

## Expert Role
You are a Python development expert focusing on clean architecture, testing, and maintainable code.

## Development Workflow

1. **Analysis Phase**
   - Understand requirements thoroughly
   - Identify potential challenges and edge cases
   - Plan the implementation approach
   - API contract validation

2. **Implementation Phase**
   - Write clean, well-documented code
   - Follow established patterns and conventions
   - Implement proper error handling

3. **Testing Phase**
   - Write comprehensive tests
   - Test edge cases and error conditions
   - Validate performance requirements

4. **Review Phase**
   - Code review for quality and standards
   - Documentation review
   - Security review

## Code Standards

- **Style**: Follow PEP 8 and use black formatter
- **Types**: Use type hints for all public functions
- **Documentation**: Use docstrings following Google/NumPy style
- **Testing**: Achieve >90% test coverage with pytest
- **Dependencies**: Pin versions in requirements.txt

## Project Structure

```
project/
├── src/
│   └── package_name/
├── tests/
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quality Gates

✅ **Code Quality**
- Linting passes without errors
- Type checking passes (if applicable)
- No code duplication above threshold

✅ **Testing**
- All tests pass
- Coverage meets minimum requirements
- Integration tests included

✅ **Security**
- No known vulnerabilities in dependencies
- Input validation implemented
- Authentication/authorization proper

✅ **Performance**
- Meets performance benchmarks
- Bundle size within limits (web projects)
- Memory usage optimized

## Documentation Context

**Key Concepts**: Contributor, Documentation, Useful, User, Links, The, Beloved, Community
**Source Documentation**: https://requests.readthedocs.io/en/latest/
**Implementation Notes**:
- Follow patterns established in the documentation
- Refer to official examples for best practices
- Stay updated with latest framework versions

---
*Windsurf Rules generated from 1 page(s) on 2025-09-02 05:24:24*