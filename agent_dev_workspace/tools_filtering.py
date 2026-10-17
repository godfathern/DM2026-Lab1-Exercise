from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Flags terms in the document-term matrix by variance against a given threshold.
        
        Args:
            threshold: Minimum variance threshold for keeping a term (default 0.0).
            top_n: Maximum number of kept/removed terms to include in summary lists (default 20).
            
        Returns:
            Dict containing summary of kept/removed term counts and sample term lists.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        n_docs = session.feature_matrix.shape[0]
        if n_docs == 0:
            raise ValueError("session.feature_matrix has 0 documents.")

        # Compute column population variance (ddof=0) entirely in sparse format to prevent OOM
        means = np.asarray(session.feature_matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(session.feature_matrix.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (means ** 2)
        variances = np.clip(variances, 0.0, None)  # Clip tiny floating point precision artifacts

        # Build full report DataFrame with term as 1st col and variance as 2nd col, sorted descending
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Filter terms by threshold
        kept_df = df_report[df_report["variance"] >= threshold]
        removed_df = df_report[df_report["variance"] < threshold]

        kept_terms_all = kept_df["term"].tolist()
        removed_terms_all = removed_df["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms_all),
            "n_removed": len(removed_terms_all),
            "kept_terms": kept_terms_all[:top_n],
            "removed_terms": removed_terms_all[:top_n]
        }

        args = {"threshold": threshold, "top_n": top_n}

        # Store result with full DataFrame report (term 1st col, variance 2nd col, sorted)
        session.store_result("variance_filter_tool", args, summary, full_report=df_report)
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Calculates Pearson correlation between term frequencies and a target class indicator.
        
        Args:
            target_class: Category label to correlate terms against (e.g., 'catB').
            threshold: Minimum absolute correlation threshold for keeping terms (default 0.0).
            top_n: Maximum number of kept/removed terms to include in summary lists (default 20).
            
        Returns:
            Dict containing summary of kept/removed terms and top term correlations.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or lacks 'category_name' column.")

        if target_class not in session.dataframe["category_name"].values:
            raise ValueError(f"Target class '{target_class}' not found in dataset categories.")

        # Binary indicator vector y (1 for target_class, 0 otherwise)
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        n_terms = len(session.feature_names)
        correlations = []

        # Iterate column-by-column to avoid converting full matrix to dense array (OOM safety)
        for j in range(n_terms):
            x = session.feature_matrix[:, j].toarray().ravel().astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r, _ = pearsonr(x, y)
                if np.isnan(r):
                    r = 0.0
            correlations.append(float(r))

        # Build full report DataFrame: term 1st col, pearson_r 2nd col, sorted descending by pearson_r
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": correlations
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        kept_df = df_report[df_report["pearson_r"].abs() >= threshold]
        removed_df = df_report[df_report["pearson_r"].abs() < threshold]

        kept_terms_all = kept_df["term"].tolist()
        removed_terms_all = removed_df["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms_all),
            "n_removed": len(removed_terms_all),
            "kept_terms": kept_terms_all[:top_n],
            "removed_terms": removed_terms_all[:top_n]
        }

        args = {"target_class": target_class, "threshold": threshold, "top_n": top_n}

        # Store result with full DataFrame report (term 1st col, pearson_r 2nd col, sorted)
        session.store_result("pearson_filter_tool", args, summary, full_report=df_report)
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between term frequencies and a target class indicator.
        
        Args:
            target_class: Category label to correlate terms against (e.g., 'catB').
            threshold: Minimum absolute correlation threshold for keeping terms (default 0.0).
            top_n: Maximum number of kept/removed terms to include in summary lists (default 20).
            
        Returns:
            Dict containing summary of kept/removed terms and top term rank correlations.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError("session.dataframe is missing or lacks 'category_name' column.")

        if target_class not in session.dataframe["category_name"].values:
            raise ValueError(f"Target class '{target_class}' not found in dataset categories.")

        # Binary indicator vector y (1 for target_class, 0 otherwise)
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        n_terms = len(session.feature_names)
        correlations = []

        # Iterate column-by-column to avoid converting full matrix to dense array (OOM safety)
        for j in range(n_terms):
            x = session.feature_matrix[:, j].toarray().ravel().astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r, _ = spearmanr(x, y)
                if np.isnan(r):
                    r = 0.0
            correlations.append(float(r))

        # Build full report DataFrame: term 1st col, spearman_r 2nd col, sorted descending by spearman_r
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": correlations
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        kept_df = df_report[df_report["spearman_r"].abs() >= threshold]
        removed_df = df_report[df_report["spearman_r"].abs() < threshold]

        kept_terms_all = kept_df["term"].tolist()
        removed_terms_all = removed_df["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms_all),
            "n_removed": len(removed_terms_all),
            "kept_terms": kept_terms_all[:top_n],
            "removed_terms": removed_terms_all[:top_n]
        }

        args = {"target_class": target_class, "threshold": threshold, "top_n": top_n}

        # Store result with full DataFrame report (term 1st col, spearman_r 2nd col, sorted)
        session.store_result("spearman_filter_tool", args, summary, full_report=df_report)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
