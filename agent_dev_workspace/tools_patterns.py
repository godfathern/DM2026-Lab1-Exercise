import os
import tempfile
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI module imports
import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpgrowth_module
import PAMI.frequentPattern.topk.FAE as topk_module
import PAMI.frequentPattern.maximal.MaxFPGrowth as maxfpgrowth_module

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Mines frequent word patterns for a specific document category using PAMI algorithms.
        
        Args:
            category_name: Category label in session.dataframe to filter and mine (e.g. "catA").
            filtering_method: Term selection method - "tfidf", "variance", or "term_frequency" (default "tfidf").
            algorithm: PAMI mining algorithm - "fpgrowth", "topk", or "maxfpgrowth" (default "fpgrowth").
            min_sup: Minimum support count threshold required for "fpgrowth" and "maxfpgrowth" (default 1).
            k: Number of top patterns to return for "topk" algorithm (default 10).
            top_n: Maximum number of mined patterns to return in summary preview (default 20).
            
        Returns:
            Dict containing mined patterns list, pattern count, category name, filtering method, and algorithm.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not initialized. Load a dataset first.")

        if "category_name" not in session.dataframe.columns and "label" not in session.dataframe.columns:
            raise ValueError("Category column ('category_name' or 'label') missing in session.dataframe.")

        cat_col = "category_name" if "category_name" in session.dataframe.columns else "label"
        cat_df = session.dataframe[session.dataframe[cat_col] == category_name]

        if len(cat_df) == 0:
            raise ValueError(f"Category '{category_name}' not found in {cat_col} column.")

        if filtering_method not in ["tfidf", "variance", "term_frequency"]:
            raise ValueError("filtering_method must be 'tfidf', 'variance', or 'term_frequency'.")

        if algorithm not in ["fpgrowth", "topk", "maxfpgrowth"]:
            raise ValueError("algorithm must be 'fpgrowth', 'topk', or 'maxfpgrowth'.")

        if top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        # Step 1: Build fresh per-category DTM with English stop words removal
        texts = cat_df["text"].fillna("").astype(str).tolist()
        vectorizer = CountVectorizer(stop_words="english")
        
        try:
            X_cat = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out().tolist()
        except ValueError:
            # Empty vocabulary after stop words removal
            feature_names = []

        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns": 0,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            args = {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}
            session.store_result("mine_patterns_tool", args, summary)
            return summary

        # Step 2: Apply chosen vocabulary filtering method
        X_dense = X_cat.toarray()
        
        if filtering_method == "variance":
            # Keep middle 90% (5th to 95th percentile) by variance
            variances = np.var(X_dense, axis=0)
            if len(variances) <= 2:
                kept_indices = []
            else:
                p5, p95 = np.percentile(variances, 5), np.percentile(variances, 95)
                kept_indices = [i for i, v in enumerate(variances) if p5 <= v <= p95]

        elif filtering_method == "term_frequency":
            # Keep middle 90% (5th to 95th percentile) by column counts
            counts = np.sum(X_dense, axis=0)
            if len(counts) <= 2:
                kept_indices = []
            else:
                p5, p95 = np.percentile(counts, 5), np.percentile(counts, 95)
                kept_indices = [i for i, v in enumerate(counts) if p5 <= v <= p95]

        else:  # "tfidf"
            # Keep top 80% (>= 20th percentile) by mean TF-IDF
            tfidf_trans = TfidfTransformer()
            X_tfidf = tfidf_trans.fit_transform(X_cat).toarray()
            mean_tfidf = np.mean(X_tfidf, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            kept_indices = [i for i, v in enumerate(mean_tfidf) if v >= p20]

        # Step 3: Handle zero terms surviving filtering cleanly
        if len(kept_indices) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            args = {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}
            session.store_result("mine_patterns_tool", args, summary)
            return summary

        surviving_terms = [feature_names[i] for i in kept_indices]
        X_filtered = X_dense[:, kept_indices]

        # Filter out empty documents (all zeros) for transactional database
        row_sums = np.sum(X_filtered, axis=1)
        X_active = X_filtered[row_sums > 0]

        if X_active.shape[0] == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns": 0,
                "patterns": [],
                "note": "no non-empty transactions after filtering"
            }
            args = {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}
            session.store_result("mine_patterns_tool", args, summary)
            return summary

        # Step 4: Convert surviving terms to transactional database using DF2DB
        df_filtered = pd.DataFrame(X_active, columns=surviving_terms)
        
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp_file:
            db_path = tmp_file.name

        try:
            converter = db_converter.DF2DB(df_filtered)
            converter.convert2TransactionalDatabase(db_path, ">=", 1)

            # Step 5: Run requested PAMI algorithm using module.Class(...) syntax
            if algorithm == "fpgrowth":
                miner = fpgrowth_module.FPGrowth(iFile=db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = topk_module.FAE(iFile=db_path, k=k, sep="\t")
            else:  # "maxfpgrowth"
                miner = maxfpgrowth_module.MaxFPGrowth(iFile=db_path, minSup=min_sup, sep="\t")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            patterns_list = []
            if patterns_df is not None and not patterns_df.empty:
                # Standardize PAMI dataframe column names ('itemset' / 'Patterns', 'support' / 'Support')
                col_itemset = "itemset" if "itemset" in patterns_df.columns else patterns_df.columns[0]
                col_support = "support" if "support" in patterns_df.columns else patterns_df.columns[1]

                for _, row in patterns_df.iterrows():
                    item_str = str(row[col_itemset]).strip()
                    items = [it.strip() for it in item_str.split() if it.strip()]
                    sup_val = int(row[col_support])
                    patterns_list.append({
                        "pattern": items,
                        "support": sup_val
                    })

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

        # Store pattern list in session.artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        # Capped summary output per LESSONS_LEARNED #1
        capped_patterns = patterns_list[:top_n]

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "total_patterns": len(patterns_list),
            "patterns": capped_patterns,
            "truncated": len(patterns_list) > top_n
        }

        args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
            "top_n": top_n
        }

        session.store_result("mine_patterns_tool", args, summary)
        return summary

    return [mine_patterns_tool]
