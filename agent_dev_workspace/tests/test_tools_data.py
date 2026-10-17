import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

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

def test_list_files_tool():
    """Known Answer #11 in TEST_FIXTURE.md: list_files_tool(subdirectory='newdataset')"""
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]
    assert result["subdirectory"] == "newdataset"
    assert result["total_directories"] == 0
    assert result["total_files"] == 1

def test_check_missing_tool():
    """Known Answer #2: Exactly 1 row missing/empty text (row 7) out of 8 rows."""
    session = get_loaded_session()
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({"text_column": "text"})

    assert res["total_rows"] == 8
    assert res["n_missing"] == 1
    assert res["missing_indices"] == [7]

def test_check_duplicates_tool_no_drop():
    """Known Answer #3 (check only): Exactly 1 duplicate detected (row 6 is copy of row 5)."""
    session = get_loaded_session()
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": False})

    assert res["n_duplicates"] == 1
    assert res["duplicate_indices"] == [6]
    assert len(session.dataframe) == 8

def test_check_duplicates_tool_with_drop():
    """Known Answer #3 (drop with keep=False): Removes ALL copies of duplicates, leaving 6 rows."""
    session = get_loaded_session()
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": True})

    assert res["n_duplicates"] == 1
    assert res["duplicate_indices"] == [6]
    assert res["dropped"] is True
    # Rows 5 and 6 are both gone when keep=False, dropping total count from 8 to 6
    assert len(session.dataframe) == 6

def test_tokenize_tool():
    """Known Answer #10: Tokenize unigrams on sample_fixture.csv with lowercasing."""
    session = get_loaded_session()
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"text_column": "text", "output_column": "unigrams"})

    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert res["n_documents"] == 8
    assert res["total_tokens"] == sum(len(toks) for toks in expected_unigrams)
    assert res["sample_tokens"] == ["always", "alpha", "alpha", "alpha", "beta"]
    
    # Verify exact unigram lists stored in dataframe and artifacts
    assert session.dataframe["unigrams"].tolist() == expected_unigrams
    assert session.artifacts["unigrams"] == expected_unigrams

def test_sample_data_tool():
    """Known Answer #13: Subsamples n=4 rows with random_state=42 returning indices [1, 5, 0, 7]."""
    session = get_loaded_session()
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["original_rows"] == 8
    assert res["sampled_rows"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert res["random_state"] == 42
    
    # Verify session.dataframe was updated to 4 rows and backup retained original 8 rows
    assert len(session.dataframe) == 4
    assert len(session.artifacts["original_dataframe"]) == 8

def test_describe_data_tool():
    """Known Answer #18: Character length stats overall and per-category with boxplot figure."""
    session = get_loaded_session()
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({"text_column": "text", "category_column": "category_name"})

    # Known character lengths per document in 8-row order
    assert res["text_lengths"] == [29, 23, 17, 30, 24, 18, 18, 0]
    assert res["n_documents"] == 8

    # Overall stats assertions using pytest.approx for floating point numbers
    overall = res["text_length_overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    # Per-category catA stats assertions
    catA = res["text_length_by_category"]["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29

    # Per-category catB stats assertions
    catB = res["text_length_by_category"]["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30

    # Verify box plot figure was set on session.pending_figure
    assert session.pending_figure is not None

def test_inspect_data_tool():
    """Custom test for inspect_data_tool (Known Answer #12: No fixed contract)."""
    session = get_loaded_session()
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Test random sampling mode
    res_rand = inspect_data_tool.invoke({"n_rows": 5, "mode": "random", "random_state": 42})
    assert res_rand["total_rows"] == 8
    assert res_rand["inspected_rows_count"] == 5
    assert res_rand["mode"] == "random"
    assert "text" in res_rand["columns"]
    assert len(res_rand["sample_records"]) == 5

    # Test head mode with short text truncation (max_text_len=10)
    res_head = inspect_data_tool.invoke({"n_rows": 3, "mode": "head", "max_text_len": 10})
    assert res_head["inspected_indices"] == [0, 1, 2]
    assert res_head["inspected_rows_count"] == 3
    # "always alpha alpha alpha beta" (29 chars) truncated to 10 chars + "..."
    assert res_head["sample_records"][0]["text"] == "always alp..."
