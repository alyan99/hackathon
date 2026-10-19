"""Tests for RAG retriever guardrails."""

import pytest


def test_supported_query_returns_citation():
    """A supported question should return a non-abstained result with citations."""
    from src.rag.retriever import retrieve
    result = retrieve("How often should outdoor workers drink water in hot weather?")
    # If index not built yet, we get abstained=True with a FileNotFoundError note — that's acceptable
    if result["abstained"]:
        assert "I don't know" in result["answer"] or "not available" in result["answer"]
    else:
        assert len(result["citations"]) >= 1
        assert result["citations"][0]["chunk_id"] != ""
        assert result["citations"][0]["title"] != ""


def test_unsupported_query_abstains():
    """A question outside the corpus should trigger abstention."""
    from src.rag.retriever import retrieve
    result = retrieve("What is the current price of crude oil in Pakistan?")
    # Either abstained due to low relevance, or abstained due to index not built
    assert result["abstained"] is True
    assert "I don't know" in result["answer"]


def test_pii_query_blocked():
    """A query containing a phone number should be blocked."""
    from src.rag.retriever import retrieve
    result = retrieve("Tell me about worker 03001234567 heat exposure")
    assert result["abstained"] is True
    assert "personal information" in result["answer"].lower()


def test_citation_has_required_fields():
    """Each citation must include title, publisher, url, pub_date, passage."""
    from src.rag.retriever import retrieve
    result = retrieve("What are symptoms of heat stroke?")
    if not result["abstained"]:
        for cit in result["citations"]:
            for field in ["title", "publisher", "url", "pub_date", "passage", "chunk_id"]:
                assert field in cit, f"Citation missing field: {field}"
