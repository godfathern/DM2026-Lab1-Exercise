import pytest
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool():
    # Single session reused across setup and tool execution
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture data into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Instantiate student tool using same session
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    
    # Invoke build_dtm_tool with default parameters
    result = build_dtm_tool.invoke({})
    
    # Assertions against known answer #4 in TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert pytest.approx(result["sparsity_pct"], abs=1e-3) == 47.5
    
    # Check vocabulary matches 'alpha', 'always', 'beta', 'delta', 'gamma'
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

def test_term_frequency_tool():
    # Fresh session for isolated test run
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")
    
    # Build DTM first
    build_dtm_tool.invoke({})
    
    # Invoke term_frequency_tool
    result = term_frequency_tool.invoke({"top_n": 20})
    
    # Known answer #14 in TEST_FIXTURE.md:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_frequencies = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3
    }
    
    assert result["total_terms"] == 5
    assert result["term_frequencies"] == expected_frequencies

def test_dtm_heatmap_tool():
    # Fresh session for isolated test run
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")
    
    # Build DTM first
    build_dtm_tool.invoke({})
    
    # 1. Test full matrix slice with default arguments ({}) -> defaults to n_terms=20, n_documents=20
    result_full = dtm_heatmap_tool.invoke({})
    
    # Known answer #21 in TEST_FIXTURE.md (8 docs x 5 terms)
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    
    assert result_full["n_documents"] == 8
    assert result_full["n_terms"] == 5
    assert result_full["included_terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert result_full["doc_indices"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert result_full["matrix_slice"] == expected_full_matrix
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    
    # 2. Test corner slice (n_terms=2, n_documents=3)
    result_corner = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    
    expected_corner_matrix = [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
    
    assert result_corner["n_documents"] == 3
    assert result_corner["n_terms"] == 2
    assert result_corner["included_terms"] == ["alpha", "always"]
    assert result_corner["doc_indices"] == [0, 1, 2]
    assert result_corner["matrix_slice"] == expected_corner_matrix
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
