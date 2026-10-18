from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: int = 1,
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[str] = None
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from text documents in session.dataframe.
        
        Args:
            max_features: Maximum number of terms to keep in the vocabulary.
            min_df: Minimum document frequency threshold for terms.
            ngram_range: List of two integers [min_n, max_n] for n-gram extraction range.
            stop_words: Language for stop words removal (e.g., 'english'), or None.
            
        Returns:
            Dict containing summary stats of the DTM including sparsity metrics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe is not initialized or missing 'text' column. Load a dataset first.")

        # Handle empty/NaN text values by replacing with empty string
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        # Convert list parameter to tuple for sklearn
        ngram_tuple = (ngram_range[0], ngram_range[1])

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_tuple,
            stop_words=stop_words
        )

        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Save artifacts and state in session
        session.artifacts["count_vectorizer"] = vectorizer
        session.feature_matrix = X
        session.feature_names = feature_names

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        # Extract top 10 terms by total document frequency for feature_sample
        if n_cols > 0:
            counts = np.asarray(X.sum(axis=0)).ravel()
            top_indices = np.argsort(-counts)[:10]
            feature_sample = [feature_names[i] for i in top_indices]
        else:
            feature_sample = []

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_sample": feature_sample
        }

        args = {
            "max_features": max_features,
            "min_df": min_df,
            "ngram_range": ngram_range,
            "stop_words": stop_words
        }

        session.store_result("build_dtm_tool", args, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the document-term matrix.
        
        Args:
            top_n: Number of top terms by frequency to include in the summary (default 20).
            
        Returns:
            Dict containing summary of top term frequency counts across all documents.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        # Sum total frequency for each term across all documents (axis 0)
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        
        df_freq = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        # Cap summary output to top_n items to protect LLM context window
        df_summary = df_freq.head(top_n)
        summary_freqs = dict(zip(df_summary["term"], df_summary["frequency"].astype(int)))

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "top_n": top_n,
            "term_frequencies": summary_freqs
        }

        args = {"top_n": top_n}

        # Store result with full DataFrame report so visualize_result_tool can bar-chart it
        session.store_result("term_frequency_tool", args, summary, full_report=df_freq)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Generates a heatmap of a raw positional slice of the document-term matrix.
        
        Args:
            n_terms: Number of terms (columns) to slice from the DTM (default 20).
            n_documents: Number of documents (rows) to slice from the DTM (default 20).
            
        Returns:
            Dict containing the sliced matrix, included term names, and document labels.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix or session.feature_names is missing. Run build_dtm_tool first.")

        if n_terms <= 0 or n_documents <= 0:
            raise ValueError("n_terms and n_documents must be positive integers.")

        # Slice sub-matrix and convert to dense numpy array
        matrix_slice = session.feature_matrix[0:n_documents, 0:n_terms].toarray()
        included_terms = session.feature_names[0:n_terms]
        doc_indices = list(range(matrix_slice.shape[0]))

        # Render heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, len(included_terms) * 0.8), max(4, matrix_slice.shape[0] * 0.5)))
        sns.heatmap(
            matrix_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=included_terms,
            yticklabels=doc_indices,
            cbar=True,
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (First {matrix_slice.shape[0]} Docs x {len(included_terms)} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        # Set figure on session attribute for automatic pipeline rendering
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": matrix_slice.shape[0],
            "n_terms": len(included_terms),
            "included_terms": included_terms,
            "doc_indices": doc_indices,
            "matrix_slice": matrix_slice.tolist()
        }

        args = {"n_terms": n_terms, "n_documents": n_documents}

        session.store_result("dtm_heatmap_tool", args, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
