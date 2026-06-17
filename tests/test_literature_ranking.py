"""Tests for relevance ranking + dedup in the literature screening stage.

These guard the fix for the Stage-5 failure mode where collection-order
(citation-sorted) truncation starved the screener of relevant papers.
"""

from researchclaw.pipeline.stage_impls._literature import (
    _dedupe_candidates,
    _rank_candidates_by_relevance,
)


def test_relevance_ranking_beats_citation_count():
    """A topical paper must rank above an off-topic mega-cited one."""
    topic = "parameter-efficient graph neural networks for brain connectivity"
    queries = [
        "BrainGNN interpretable graph network",
        "brain connectivity graph neural network",
    ]
    rows = [
        {
            "title": "Array programming with NumPy",
            "abstract": "fundamental numerical computing arrays in python",
            "citation_count": 20000,
            "year": 2020,
            "keyword_overlap": 0,
        },
        {
            "title": "BrainGNN: Interpretable Brain Graph Neural Network for fMRI",
            "abstract": (
                "a parameter efficient graph neural network for brain "
                "connectivity classification from fMRI"
            ),
            "citation_count": 100,
            "year": 2021,
            "keyword_overlap": 5,
        },
    ]

    ranked = _rank_candidates_by_relevance(rows, topic, queries)

    assert ranked[0]["title"].startswith("BrainGNN")
    assert ranked[-1]["title"] == "Array programming with NumPy"
    assert all("relevance_score" in r for r in ranked)


def test_relevance_dominates_even_at_zero_citations():
    """An on-topic paper with 0 citations must beat an off-topic blockbuster.

    Guards the core contract: citations are only a light tiebreak, so even a
    50k-citation paper from another field cannot outrank relevant work.
    """
    topic = "graph neural networks for brain connectivity classification"
    queries = ["brain connectivity graph neural network", "fMRI GNN classification"]
    rows = [
        {
            "title": "Deep Residual Learning for Image Recognition",
            "abstract": "Residual networks for image classification on ImageNet.",
            "citation_count": 50000,
            "year": 2016,
            "keyword_overlap": 1,
        },
        {
            "title": "A Graph Neural Network for Brain Connectivity Classification",
            "abstract": "We classify brain connectivity graphs from fMRI with a GNN.",
            "citation_count": 0,
            "year": 2024,
            "keyword_overlap": 4,
        },
    ]

    ranked = _rank_candidates_by_relevance(rows, topic, queries)

    assert ranked[0]["title"].startswith("A Graph Neural Network")


def test_dedupe_merges_cross_source_duplicates():
    """Same paper from two sources collapses to one richer record."""
    rows = [
        {
            "title": "BernNet: Learning Arbitrary Graph Spectral Filters",
            "abstract": "",
            "citation_count": 69,
            "source": "openalex",
            "doi": "10.1000/bernnet",
        },
        {
            "title": "BernNet: Learning Arbitrary Graph Spectral Filters",
            "abstract": "We propose BernNet, a spectral GNN using Bernstein polynomials.",
            "citation_count": 0,
            "source": "arxiv",
            "arxiv_id": "2106.10994",
        },
    ]

    deduped = _dedupe_candidates(rows)

    assert len(deduped) == 1
    merged = deduped[0]
    # keeps the abstract (from the arXiv record)...
    assert "Bernstein polynomials" in merged["abstract"]
    # ...and backfills the citation count (from the OpenAlex record)
    assert merged["citation_count"] == 69
