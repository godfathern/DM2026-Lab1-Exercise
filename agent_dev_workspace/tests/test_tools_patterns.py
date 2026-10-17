import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def get_loaded_session():
    """Helper to initialize SessionState and load sample_fixture.csv via load_dataset_tool."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def test_mine_patterns_fpgrowth_tfidf():
    """Known Answer #17: FPGrowth with TF-IDF filtering on sample_fixture.csv."""
    session = get_loaded_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # catA -> keeps alpha -> pattern {alpha} with support 3
    res_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_a["total_patterns"] == 1
    assert res_a["patterns"][0]["pattern"] == ["alpha"]
    assert res_a["patterns"][0]["support"] == 3

    # catB -> keeps gamma -> pattern {gamma} with support 4
    res_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_b["total_patterns"] == 1
    assert res_b["patterns"][0]["pattern"] == ["gamma"]
    assert res_b["patterns"][0]["support"] == 4

def test_mine_patterns_zero_terms_edge_case():
    """Known Answer #17 edge case: variance/term_frequency filtering keeps 0 terms."""
    session = get_loaded_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["total_patterns"] == 0
    assert res_var["patterns"] == []
    assert res_var["note"] == "no terms survived filtering"

def test_mine_patterns_topk():
    """Known Answer #20: TopK algorithm (k=1) on TF-IDF filtered data."""
    session = get_loaded_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res_topk_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_a["total_patterns"] == 1
    assert res_topk_a["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_a["patterns"][0]["support"] == 3

def test_mine_patterns_maxfpgrowth():
    """Known Answer #20: MaxFPGrowth algorithm returns 0 patterns on single-term database."""
    session = get_loaded_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res_max_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_a["total_patterns"] == 0
    assert res_max_a["patterns"] == []
