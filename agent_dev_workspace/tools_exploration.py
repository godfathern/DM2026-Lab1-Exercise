from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_idx1: int = 0,
        doc_idx2: int = 1,
        text1: Optional[str] = None,
        text2: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents by matrix index or raw text strings.
        
        Args:
            doc_idx1: Row index of first document in session.feature_matrix (default 0).
            doc_idx2: Row index of second document in session.feature_matrix (default 1).
            text1: Optional raw text for document 1. If provided with text2, vectorizes text directly.
            text2: Optional raw text for document 2. If provided with text1, vectorizes text directly.
            
        Returns:
            Dict containing document indices, similarity score (float 0.0 to 1.0), and result_id.
        """
        if text1 is not None and text2 is not None:
            # Vectorize raw text inputs using count_vectorizer stored in session.artifacts
            if "count_vectorizer" not in session.artifacts:
                raise ValueError("Count vectorizer missing in session.artifacts. Run build_dtm_tool first.")
            vectorizer = session.artifacts["count_vectorizer"]
            v1 = vectorizer.transform([str(text1)])
            v2 = vectorizer.transform([str(text2)])
        else:
            # Slice document vectors directly from session.feature_matrix
            if session.feature_matrix is None:
                raise ValueError("session.feature_matrix is missing. Run build_dtm_tool first.")
            
            n_docs = session.feature_matrix.shape[0]
            if doc_idx1 < 0 or doc_idx1 >= n_docs:
                raise ValueError(f"doc_idx1 ({doc_idx1}) is out of bounds for matrix with {n_docs} documents.")
            if doc_idx2 < 0 or doc_idx2 >= n_docs:
                raise ValueError(f"doc_idx2 ({doc_idx2}) is out of bounds for matrix with {n_docs} documents.")

            v1 = session.feature_matrix[doc_idx1:doc_idx1+1]
            v2 = session.feature_matrix[doc_idx2:doc_idx2+1]

        # Handle zero vector edge cases cleanly (e.g. empty text rows returning 0.0 instead of NaN)
        norm1 = np.linalg.norm(v1.toarray()) if hasattr(v1, "toarray") else np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2.toarray()) if hasattr(v2, "toarray") else np.linalg.norm(v2)

        if norm1 == 0.0 or norm2 == 0.0:
            sim_score = 0.0
        else:
            sim_matrix = cosine_similarity(v1, v2)
            sim_score = float(sim_matrix[0, 0])
            if np.isnan(sim_score):
                sim_score = 0.0

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "similarity": round(sim_score, 4)
        }

        args = {
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "text1": text1,
            "text2": text2
        }

        session.store_result("cosine_similarity_tool", args, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20
    ) -> Dict[str, Any]:
        """Calculates Pearson correlation matrix for top variance terms and generates a heatmap.
        
        Args:
            top_k: Maximum number of top terms by variance to include in correlation matrix (default 20).
            
        Returns:
            Dict containing top terms by variance, term variances, correlation matrix, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        n_docs = session.feature_matrix.shape[0]
        if n_docs == 0:
            raise ValueError("session.feature_matrix has 0 documents.")

        # Per LESSONS_LEARNED #2: Compute variance strictly in sparse format without .toarray() on full matrix
        means = np.asarray(session.feature_matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(session.feature_matrix.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (means ** 2)
        variances = np.clip(variances, 0.0, None)

        # Rank terms by variance descending and take top_k
        df_var = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances,
            "orig_idx": list(range(len(session.feature_names)))
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        k_count = min(top_k, len(df_var))
        top_df = df_var.head(k_count)

        top_terms = top_df["term"].tolist()
        top_variances = top_df["variance"].tolist()
        top_indices = top_df["orig_idx"].tolist()

        # Convert ONLY the sliced top-k columns to dense array for correlation calculation
        X_top = session.feature_matrix[:, top_indices].toarray()
        
        # Handle zero variance columns safely
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(X_top, rowvar=False)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        corr_list = np.round(corr_matrix, 4).tolist()

        # Step 3: Build annotated heatmap figure and set on session.pending_figure per contract
        fig, ax = plt.subplots(figsize=(max(6, len(top_terms) * 0.8), max(5, len(top_terms) * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cbar=True,
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top {len(top_terms)} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "top_terms": top_terms,
            "variances": [round(v, 4) for v in top_variances],
            "correlation_matrix": corr_list
        }

        args = {"top_k": top_k}

        session.store_result("feature_correlation_matrix_tool", args, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
