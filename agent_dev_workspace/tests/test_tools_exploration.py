import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def get_loaded_session_with_dtm():
    """Helper to load sample_fixture.csv and build the DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session

def test_cosine_similarity_known_answers():
    """Known Answer #16: Cosine similarity pairs on sample_fixture.csv."""
    session = get_loaded_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Pair 0 vs 1: catA vs catA (near-identical text) -> 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Pair 0 vs 3: catA vs catB (share only 'always') -> 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Pair 3 vs 4: catB vs catB (near-identical text) -> 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_idx1": 3, "doc_idx2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Pair 0 vs 7: catA vs empty text row -> 0.0 (edge case)
    res_0_7 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 7})
    assert res_0_7["similarity"] == 0.0

def test_cosine_similarity_raw_text():
    """Known Answer #16: Vectorizing raw text strings using session.artifacts['count_vectorizer']."""
    session = get_loaded_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    res = cosine_similarity_tool.invoke({
        "text1": "always alpha alpha alpha beta",
        "text2": "always alpha alpha beta"
    })
    assert res["similarity"] == pytest.approx(0.9847, abs=1e-3)

def test_feature_correlation_matrix_tool():
    """Known Answer #19: Feature correlation matrix across top-variance terms on sample_fixture.csv."""
    session = get_loaded_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({"top_k": 20})

    # Top 5 terms ranked by variance descending
    expected_top_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_terms"] == expected_top_terms

    # Term variances
    expected_variances = [1.1875, 1.1094, 0.25, 0.2344, 0.1094]
    assert np.allclose(res["variances"], expected_variances, atol=1e-3)

    # Key Pearson correlation coefficient assertions
    corr_matrix = np.array(res["correlation_matrix"])
    
    # Diagonal must be strictly 1.0 (term against itself)
    assert np.allclose(np.diag(corr_matrix), np.ones(5), atol=1e-3)

    # alpha (idx 0) vs beta (idx 3) -> 0.8885
    assert corr_matrix[0, 3] == pytest.approx(0.8885, abs=1e-3)
    assert corr_matrix[3, 0] == pytest.approx(0.8885, abs=1e-3)

    # gamma (idx 1) vs delta (idx 2) -> 0.8307
    assert corr_matrix[1, 2] == pytest.approx(0.8307, abs=1e-3)
    assert corr_matrix[2, 1] == pytest.approx(0.8307, abs=1e-3)

    # alpha (idx 0) vs delta (idx 2) -> -0.6882
    assert corr_matrix[0, 2] == pytest.approx(-0.6882, abs=1e-3)

    # Heatmap figure set on session.pending_figure
    assert session.pending_figure is not None
