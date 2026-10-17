from typing import List, Optional, Dict, Any, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensions (PCA/t-SNE/UMAP) and builds a 2D scatter plot colored by category.
        
        Args:
            method: Reduction algorithm - "pca", "tsne", or "umap" (default "pca").
            n_components: Target dimensions, must be 2 for 2D plotting (default 2).
            random_state: Seed for reproducible random initialization (default 42).
            perplexity: t-SNE perplexity parameter for neighbor manifold density (default 30.0).
            n_neighbors: UMAP n_neighbors parameter for local cluster structure (default 15).
            top_n: Maximum number of coordinate rows to include in summary preview (default 20).
            
        Returns:
            Dict containing method, capped 2D coordinates preview, total sample count, explained_variance_ratio (PCA), and result_id.
        """
        if session.feature_matrix is None:
            raise ValueError("session.feature_matrix is missing. Run build_dtm_tool first.")

        if n_components != 2:
            raise ValueError("n_components must be 2 for 2D visualization.")

        if method not in ["pca", "tsne", "umap"]:
            raise ValueError("method must be 'pca', 'tsne', or 'umap'.")

        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        # Convert sparse feature matrix to dense numpy array
        X_dense = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else session.feature_matrix
        n_samples = X_dense.shape[0]

        if n_samples < 2:
            raise ValueError("At least 2 documents are required for dimension reduction.")

        explained_variance_ratio = None

        # Step 1: Perform requested dimension reduction
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()

        elif method == "tsne":
            # Adjust perplexity if sample size is too small for default
            effective_perplexity = perplexity
            if effective_perplexity >= n_samples:
                effective_perplexity = max(1.0, float(n_samples - 1))

            reducer = TSNE(
                n_components=n_components,
                perplexity=effective_perplexity,
                random_state=random_state,
                init="pca"
            )
            coords = reducer.fit_transform(X_dense)

        else:  # "umap"
            # Adjust n_neighbors if sample size is too small for default
            effective_neighbors = n_neighbors
            if effective_neighbors >= n_samples:
                effective_neighbors = max(2, n_samples - 1)

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=effective_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X_dense)

        # Store ALL reduced coordinates in session.artifacts
        coords_list = coords.tolist()
        session.artifacts["reduced_coordinates"] = coords_list

        # Step 2: Build 2D scatter plot colored strictly by session.labels per contract
        category_labels = session.labels
        if category_labels is None and session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                category_labels = session.dataframe["category_name"].values
            elif "label" in session.dataframe.columns:
                category_labels = session.dataframe["label"].values

        df_plot = pd.DataFrame({
            "Dim1": coords[:, 0],
            "Dim2": coords[:, 1]
        })
        if category_labels is not None:
            df_plot["Category"] = category_labels

        fig, ax = plt.subplots(figsize=(8, 6))
        if "Category" in df_plot.columns:
            sns.scatterplot(data=df_plot, x="Dim1", y="Dim2", hue="Category", style="Category", s=100, ax=ax)
        else:
            sns.scatterplot(data=df_plot, x="Dim1", y="Dim2", s=100, ax=ax)

        ax.set_title(f"Dimension Reduction via {method.upper()}")
        ax.set_xlabel("Component 1" if method == "pca" else "Dimension 1")
        ax.set_ylabel("Component 2" if method == "pca" else "Dimension 2")
        plt.tight_layout()

        # Set figure on session attribute for automatic pipeline rendering
        session.pending_figure = fig

        # Per LESSONS_LEARNED #1: Cap returned coordinates list at top_n for summary preview
        capped_coords = coords_list[:top_n]

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_samples": n_samples,
            "total_coordinates": len(coords_list),
            "coordinates": capped_coords,
            "explained_variance_ratio": explained_variance_ratio,
            "truncated": len(coords_list) > top_n
        }

        args = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "top_n": top_n
        }

        session.store_result("reduce_dimensions_tool", args, summary)
        return summary

    @tool
    def binarize_labels_tool(
        category_column: str = "category_name",
        output_key: str = "one_hot_labels",
        top_n: int = 20
    ) -> Dict[str, Any]:
        """One-hot encodes categorical labels in session.labels or session.dataframe into binary matrix.
        
        Args:
            category_column: Column name in session.dataframe containing string category labels (default "category_name").
            output_key: Artifacts dictionary key to store the one-hot binary matrix (default "one_hot_labels").
            top_n: Maximum number of binarized rows to include in summary preview (default 20).
            
        Returns:
            Dict containing document count, unique class names, matrix shape, and capped sample binarized rows.
        """
        raw_labels = session.labels

        if raw_labels is None:
            if session.dataframe is None:
                raise ValueError("Neither session.labels nor session.dataframe is initialized. Load a dataset first.")

            col = category_column
            if col not in session.dataframe.columns:
                if "label" in session.dataframe.columns:
                    col = "label"
                else:
                    raise ValueError(f"Category column '{category_column}' not found in session.dataframe.")

            raw_labels = session.dataframe[col].values

        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        categories = np.asarray(raw_labels, dtype=str).reshape(-1, 1)

        # OneHotEncoder guarantees 1 column per category regardless of unique category count
        ohe = OneHotEncoder(sparse_output=False)
        binarized_matrix = ohe.fit_transform(categories).astype(int)
        
        # Extract unique class names (e.g. ["catA", "catB"])
        classes = [str(c) for c in ohe.categories_[0].tolist()]

        # Store encoder and binarized matrix in session.artifacts
        session.artifacts["label_encoder"] = ohe
        binarized_list = binarized_matrix.tolist()
        session.artifacts[output_key] = binarized_list

        # Capped summary preview per LESSONS_LEARNED #1
        capped_matrix = binarized_list[:top_n]

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "n_documents": len(raw_labels),
            "n_classes": len(classes),
            "classes": classes,
            "shape": list(binarized_matrix.shape),
            "binarized_matrix": capped_matrix,
            "truncated": len(binarized_list) > top_n
        }

        args = {
            "category_column": category_column,
            "output_key": output_key,
            "top_n": top_n
        }

        session.store_result("binarize_labels_tool", args, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
