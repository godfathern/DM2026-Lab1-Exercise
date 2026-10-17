import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool():
    # Fresh session for isolated test run
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
    
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    # Build DTM first
    build_dtm_tool.invoke({})
    
    # Run variance_filter_tool with threshold=0.15 matching known answer #5 in TEST_FIXTURE.md
    result = variance_filter_tool.invoke({"threshold": 0.15})
    
    # Verify summary counts and filtering results
    assert result["total_terms"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1
    assert "always" in result["removed_terms"]
    assert set(result["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    
    # Retrieve full report from session.results_store to assert exact variances from known answer #5
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    
    # Column 1 must be term name, Column 2 must be variance
    assert list(full_report.columns[:2]) == ["term", "variance"]
    
    # Verify rows are strictly sorted descending by variance
    variances_list = full_report["variance"].tolist()
    assert variances_list == sorted(variances_list, reverse=True)
    
    # Check expected term order by variance descending (alpha=1.1875, gamma=1.109375, delta=0.25, beta=0.234375, always=0.109375)
    assert list(full_report["term"]) == ["alpha", "gamma", "delta", "beta", "always"]
    
    # Map term -> variance
    variance_map = dict(zip(full_report["term"], full_report["variance"]))
    
    # Exact known variances from TEST_FIXTURE.md (#5)
    assert pytest.approx(variance_map["alpha"], abs=1e-4) == 1.1875
    assert pytest.approx(variance_map["always"], abs=1e-4) == 0.109375
    assert pytest.approx(variance_map["beta"], abs=1e-4) == 0.234375
    assert pytest.approx(variance_map["delta"], abs=1e-4) == 0.25
    assert pytest.approx(variance_map["gamma"], abs=1e-4) == 1.109375

def test_pearson_filter_tool():
    # Fresh session for isolated test run
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
    
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    build_dtm_tool.invoke({})
    
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    
    # Column 1 must be term name, Column 2 must be pearson_r
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]
    
    pearson_map = dict(zip(full_report["term"], full_report["pearson_r"]))
    
    # Exact known Pearson r values from TEST_FIXTURE.md (#6)
    assert pytest.approx(pearson_map["alpha"], abs=1e-3) == -0.6882
    assert pytest.approx(pearson_map["always"], abs=1e-3) == 0.3780
    assert pytest.approx(pearson_map["beta"], abs=1e-3) == -0.7746
    assert pytest.approx(pearson_map["delta"], abs=1e-3) == 1.0000
    assert pytest.approx(pearson_map["gamma"], abs=1e-3) == 0.8307

def test_spearman_filter_tool():
    # Fresh session for isolated test run
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
    
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")
    
    build_dtm_tool.invoke({})
    
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    
    # Column 1 must be term name, Column 2 must be spearman_r
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]
    
    spearman_map = dict(zip(full_report["term"], full_report["spearman_r"]))
    
    # Exact known Spearman r values from TEST_FIXTURE.md (#6)
    assert pytest.approx(spearman_map["alpha"], abs=1e-3) == -0.7500
    assert pytest.approx(spearman_map["always"], abs=1e-3) == 0.3780
    assert pytest.approx(spearman_map["beta"], abs=1e-3) == -0.7746
    assert pytest.approx(spearman_map["delta"], abs=1e-3) == 1.0000
    assert pytest.approx(spearman_map["gamma"], abs=1e-3) == 0.9363
