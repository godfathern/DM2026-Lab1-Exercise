import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

def get_loaded_session_with_dtm():
    """Helper to load sample_fixture.csv and build the DTM so session.labels is populated."""
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

def test_reduce_dimensions_pca():
    """Known Answer #7: PCA 2D coordinates and explained_variance_ratio on sample_fixture.csv."""
    session = get_loaded_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    expected_var = [0.7532, 0.1965]

    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert np.allclose(res["explained_variance_ratio"], expected_var, atol=1e-3)
    assert session.pending_figure is not None

def test_reduce_dimensions_tsne():
    """Property test for t-SNE reduction (Known Answer #8): verifies 2D shape, duplicate coordinate invariance, seed determinism, intra-class distance, and plot figure."""
    session = get_loaded_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Run 1
    res1 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    coords1 = np.array(res1["coordinates"])

    # Property 1: Output shape must be 8 documents x 2 dimensions
    assert coords1.shape == (8, 2)

    # Property 2: Invariant property - Rows 5 and 6 are exact duplicate documents, so t-SNE must map them to identical coordinates
    assert np.allclose(coords1[5], coords1[6], atol=1e-3)

    # Property 3: Determinism - Running again with the same seed (random_state=42) produces identical coordinates
    res2 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    coords2 = np.array(res2["coordinates"])
    assert np.allclose(coords1, coords2, atol=1e-5)

    # Property 4: Cluster integrity - catA documents (Docs 0 & 1) are closer together than catA Doc 0 is to catB Doc 3
    dist_catA = np.linalg.norm(coords1[0] - coords1[1])
    dist_catA_to_catB = np.linalg.norm(coords1[0] - coords1[3])
    assert dist_catA < dist_catA_to_catB

    # Property 5: Figure generation set on session.pending_figure
    assert session.pending_figure is not None

def test_reduce_dimensions_umap():
    """Known Answer #9: UMAP 2D coordinates with n_neighbors=3, random_state=42."""
    session = get_loaded_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])

    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1e-1)
    assert session.pending_figure is not None

def test_binarize_labels_tool():
    """Custom test for binarize_labels_tool (Known Answer #15: No fixed contract).
    Builds DTM first so session.labels is populated, then verifies One-Hot Encoding shape (8x2) and class vectors.
    """
    session = get_loaded_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    assert res["n_documents"] == 8
    assert res["n_classes"] == 2
    assert res["classes"] == ["catA", "catB"]
    assert res["shape"] == [8, 2]

    matrix = np.array(res["binarized_matrix"])

    # catA rows (0, 1, 2, 7) must map to [1, 0]
    expected_catA = np.array([1, 0])
    for idx in [0, 1, 2, 7]:
        assert np.array_equal(matrix[idx], expected_catA)

    # catB rows (3, 4, 5, 6) must map to [0, 1]
    expected_catB = np.array([0, 1])
    for idx in [3, 4, 5, 6]:
        assert np.array_equal(matrix[idx], expected_catB)

    # Every row sum in One-Hot Encoding must equal 1
    assert np.array_equal(np.sum(matrix, axis=1), np.ones(8))

    # Verify artifacts storage
    assert session.artifacts["one_hot_labels"] == res["binarized_matrix"]
