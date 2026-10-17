import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool

# Ensure NLTK punkt tokenizer data is available silently
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

def make_tools(session):
    @tool
    def list_files_tool(
        subdirectory: str = "",
        max_items: int = 50
    ) -> Dict[str, Any]:
        """Lists files and folders in a workspace directory to discover dataset paths.
        
        Args:
            subdirectory: Subdirectory path relative to workspace root (default "" for root).
            max_items: Maximum number of files and directories to include in summary lists (default 50).
            
        Returns:
            Dict containing lists of directories and files, along with total item counts.
        """
        base_dir = Path(os.getcwd()).resolve()
        target_dir = (base_dir / subdirectory).resolve()

        # Strict boundary check per LESSONS_LEARNED #6: path check by folder hierarchy
        if target_dir != base_dir and base_dir not in target_dir.parents:
            raise ValueError("Access denied: subdirectory path attempts to traverse outside repository root.")

        if not target_dir.exists():
            raise FileNotFoundError(f"Directory '{subdirectory}' does not exist.")

        if not target_dir.is_dir():
            raise ValueError(f"Path '{subdirectory}' is a file, not a directory.")

        all_dirs = []
        all_files = []

        for entry in sorted(os.listdir(target_dir)):
            # Skip hidden files and directories (starting with '.')
            if entry.startswith("."):
                continue
            
            entry_path = target_dir / entry
            if entry_path.is_dir():
                all_dirs.append(entry)
            elif entry_path.is_file():
                all_files.append(entry)

        # Capped list outputs per LESSONS_LEARNED #1
        summary_dirs = all_dirs[:max_items]
        summary_files = all_files[:max_items]

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "total_directories": len(all_dirs),
            "total_files": len(all_files),
            "directories": summary_dirs,
            "files": summary_files,
            "truncated": (len(all_dirs) > max_items or len(all_files) > max_items)
        }

        args = {"subdirectory": subdirectory, "max_items": max_items}
        session.store_result("list_files_tool", args, summary)
        return summary

    @tool
    def check_missing_tool(
        text_column: str = "text"
    ) -> Dict[str, Any]:
        """Checks for missing or empty text values in session.dataframe.
        
        Args:
            text_column: Column name to inspect for missing/empty values (default "text").
            
        Returns:
            Dict containing count and indices of rows with missing or empty text.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not initialized. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        series = session.dataframe[text_column]
        
        # Identify null, NaN, or whitespace-only/empty string rows
        is_missing = series.isna() | (series.astype(str).str.strip() == "")
        missing_indices = session.dataframe.index[is_missing].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": len(session.dataframe),
            "n_missing": len(missing_indices),
            "missing_indices": missing_indices
        }

        args = {"text_column": text_column}
        session.store_result("check_missing_tool", args, summary)
        return summary

    @tool
    def check_duplicates_tool(
        subset: Optional[List[str]] = None,
        drop: bool = False
    ) -> Dict[str, Any]:
        """Checks for and optionally drops duplicate documents in session.dataframe.
        
        Args:
            subset: List of column names to consider for identifying duplicates (default None for all columns).
            drop: If True, drops duplicate rows from session.dataframe in-place using keep=False (default False).
            
        Returns:
            Dict containing count of duplicate rows, duplicate row indices, and updated dataset row count.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not initialized. Load a dataset first.")

        cols = subset if subset else session.dataframe.columns.tolist()
        for col in cols:
            if col not in session.dataframe.columns:
                raise ValueError(f"Column '{col}' not found in session.dataframe.")

        # Identify duplicate rows across specified subset
        dup_mask = session.dataframe.duplicated(subset=cols, keep="first")
        dup_indices = session.dataframe.index[dup_mask].tolist()
        n_duplicates = len(dup_indices)

        if drop and n_duplicates > 0:
            # Per spec requirement & Known Answer #3: drop using keep=False to remove ALL copies
            session.dataframe = session.dataframe.drop_duplicates(subset=cols, keep=False).reset_index(drop=True)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "total_rows": len(session.dataframe),
            "n_duplicates": n_duplicates,
            "duplicate_indices": dup_indices,
            "dropped": drop
        }

        args = {"subset": subset, "drop": drop}
        session.store_result("check_duplicates_tool", args, summary)
        return summary

    @tool
    def tokenize_tool(
        text_column: str = "text",
        output_column: str = "unigrams",
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """Tokenizes document text into unigrams using nltk.word_tokenize on lowercased text.
        
        Args:
            text_column: Column name in session.dataframe containing raw text (default "text").
            output_column: Column name in session.dataframe to store token lists (default "unigrams").
            sample_size: Maximum number of tokens to return in the summary preview (default 10).
            
        Returns:
            Dict containing document count, total token count, and capped sample tokens preview.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not initialized. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer.")

        tokens_list = []
        total_tokens = 0

        for text_val in session.dataframe[text_column]:
            if pd.isna(text_val) or not str(text_val).strip():
                doc_tokens = []
            else:
                lowercased_text = str(text_val).lower()
                doc_tokens = word_tokenize(lowercased_text)

            tokens_list.append(doc_tokens)
            total_tokens += len(doc_tokens)

        # Store tokenized outputs in dataframe and artifacts
        session.dataframe[output_column] = tokens_list
        session.artifacts[output_column] = tokens_list

        # Preview of first non-empty document's tokens, capped per LESSONS_LEARNED #1
        first_non_empty = next((toks for toks in tokens_list if len(toks) > 0), [])
        sample_tokens = first_non_empty[:sample_size]

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(session.dataframe),
            "text_column": text_column,
            "output_column": output_column,
            "total_tokens": total_tokens,
            "sample_tokens": sample_tokens
        }

        args = {
            "text_column": text_column,
            "output_column": output_column,
            "sample_size": sample_size
        }

        session.store_result("tokenize_tool", args, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        replace: bool = False,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Subsamples session.dataframe down to n rows or a fraction of rows.
        
        Args:
            n: Number of rows to sample (e.g. 4).
            frac: Fraction of rows to sample (e.g. 0.5).
            random_state: Seed for reproducible random sampling (default None).
            replace: Whether to sample with replacement (default False).
            top_n: Maximum number of sampled row indices to include in summary preview (default 20).
            
        Returns:
            Dict containing original row count, sampled row count, capped sampled_indices, and random_state.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not initialized. Load a dataset first.")

        if n is None and frac is None:
            raise ValueError("Must specify either 'n' or 'frac' parameter for sampling.")

        if n is not None and frac is not None:
            raise ValueError("Specify either 'n' or 'frac', not both.")

        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        # Save a copy of full original dataset in session.artifacts if not already present
        if "original_dataframe" not in session.artifacts:
            session.artifacts["original_dataframe"] = session.dataframe.copy()

        original_rows = len(session.dataframe)

        # Sample rows using pandas DataFrame.sample
        sampled_df = session.dataframe.sample(
            n=n,
            frac=frac,
            replace=replace,
            random_state=random_state
        )

        sampled_indices = sampled_df.index.tolist()

        # Update session.dataframe in-place and reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample_data")
        
        # Per LESSONS_LEARNED #1: Cap returned index list at top_n while providing total counts
        capped_sampled_indices = sampled_indices[:top_n]

        summary = {
            "result_id": result_id,
            "original_rows": original_rows,
            "sampled_rows": len(session.dataframe),
            "total_sampled_indices": len(sampled_indices),
            "sampled_indices": capped_sampled_indices,
            "random_state": random_state,
            "truncated": len(sampled_indices) > top_n
        }

        args = {
            "n": n,
            "frac": frac,
            "random_state": random_state,
            "replace": replace,
            "top_n": top_n
        }

        session.store_result("sample_data_tool", args, summary)
        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name"
    ) -> Dict[str, Any]:
        """Computes text length statistics overall and per category, and builds a category box plot.
        
        Args:
            text_column: Column name in session.dataframe containing text (default "text").
            category_column: Column name in session.dataframe containing category labels (default "category_name").
            
        Returns:
            Dict containing overall text length describe stats, per-category describe stats, and document count.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not initialized. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        # Compute character length per document matching Master's X['text_length'] = X['text'].apply(len)
        session.dataframe["text_length"] = session.dataframe[text_column].fillna("").astype(str).apply(len)

        # Overall .describe() statistics
        overall_desc = session.dataframe["text_length"].describe().to_dict()

        category_stats = {}
        if category_column in session.dataframe.columns:
            # Per-category .describe() statistics
            grouped = session.dataframe.groupby(category_column)["text_length"]
            for cat_name, group_series in grouped:
                category_stats[str(cat_name)] = group_series.describe().to_dict()

            # Render box plot grouped by category
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(
                data=session.dataframe,
                x=category_column,
                y="text_length",
                ax=ax
            )
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length (Character Count)")
            plt.tight_layout()

            # Set figure on session.pending_figure per spec plotting requirement
            session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "n_documents": len(session.dataframe),
            "text_length_overall": overall_desc,
            "text_length_by_category": category_stats,
            "text_lengths": session.dataframe["text_length"].tolist()
        }

        args = {
            "text_column": text_column,
            "category_column": category_column
        }

        session.store_result("describe_data_tool", args, summary)
        return summary

    return [
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        describe_data_tool
    ]
