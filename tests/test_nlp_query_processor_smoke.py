import pytest


def test_knowledge_base_title_name_fallback(monkeypatch):
    """Ensure sources without 'title' still work by falling back to 'name'."""
    from rules_maker.batch_processor import DocumentationSource
    from rules_maker.nlp import query_processor as qp

    # Provide a single source that has only 'name' (no 'title' attribute)
    src = DocumentationSource(
        url="https://react.dev/learn",
        name="React Docs",
        technology="javascript",
        framework="react",
        priority=5,
    )

    monkeypatch.setattr(
        qp,
        "get_comprehensive_updated_sources",
        lambda: [src],
    )

    proc = qp.NaturalLanguageQueryProcessor(bedrock_config=None)

    kb = proc.knowledge_base
    assert "react" in kb["technologies"], "Technology bucket missing"
    sources = kb["technologies"]["react"]["sources"]
    assert sources and isinstance(sources, list)
    assert sources[0]["title"] == "React Docs", "Should fall back to 'name' for title"


@pytest.mark.asyncio
async def test_query_processing_basic(monkeypatch):
    """Basic smoke test for NLP query processing using fallback sources."""
    from rules_maker.batch_processor import DocumentationSource
    from rules_maker.nlp import query_processor as qp

    sources = [
        DocumentationSource(
            url="https://react.dev/learn",
            name="React Docs",
            technology="javascript",
            framework="react",
            priority=5,
        ),
        DocumentationSource(
            url="https://reactnative.dev/",
            name="React Native",
            technology="javascript",
            framework="react-native",
            priority=3,
        ),
    ]

    monkeypatch.setattr(
        qp,
        "get_comprehensive_updated_sources",
        lambda: sources,
    )

    proc = qp.NaturalLanguageQueryProcessor(bedrock_config=None)
    ctx = qp.ProjectContext(technologies=["react"], project_type="web-app", experience_level="intermediate")
    resp = await proc.process_query("What are best practices for React hooks?", context=ctx)

    assert resp is not None
    assert isinstance(resp.answer, str) and len(resp.answer) > 0
    assert 0.0 <= resp.confidence <= 1.0
    # At least one relevant source should be returned
    assert len(resp.relevant_sources) >= 1

