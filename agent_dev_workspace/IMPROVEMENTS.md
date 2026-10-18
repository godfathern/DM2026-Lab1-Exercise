# Tool Improvement Proposals

The following tool improvements were identified while executing the data mining pipeline on the real-world 20 Newsgroups dataset (`alt.atheism`, `comp.graphics`, `sci.med`, `soc.religion.christian`). Running the agent pipeline against real text data exposed key edge cases, performance bottlenecks, and terminology inconsistencies that should be updated in the agent-dev notebook:

### 1. `mine_patterns_tool` — Change `min_sup` default from `1` to `60`
* **Issue:** The default `min_sup` of `1` attempts to mine every single term co-occurrence across the entire dataset.
* **Finding on Real Data:** On real text documents, `min_sup=1` causes a combinatorial explosion in FP-Growth, causing execution to hang for over 15 minutes.
* **Fix:** Change the default `min_sup` to `60` (or a dynamic threshold based on a percentage of dataset size) to prune noise early, drastically reduce execution time, and surface meaningful patterns.

### 2. `build_dtm_tool` — Improve `vocabulary_sample` selection logic
* **Issue:** `vocabulary_sample` currently extracts the first 10 terms alphabetically from the feature array.
* **Finding on Real Data:** Because digits sort before letters alphabetically, the output preview for real text vectorizers consists entirely of numeric noise (`['00', '000', '01', '02', '03', '0358', '04', '05', '06', '07']`), offering no insight into the actual vocabulary.
* **Fix:** Update sampling logic to draw a random selection of feature names, sample from the middle/end of the feature list, or filter out purely numeric strings before generating the sample preview.

### 3. `build_dtm_tool` — Rename `vocabulary_sample` to `feature_sample`
* **Issue:** Inconsistent terminology in the return dictionary keys.
* **Finding on Real Data:** In matrix construction and machine learning workflows, matrix columns are standardly referred to as **features** (feature matrix, feature names).
* **Fix:** Rename the `vocabulary_sample` dictionary key and function parameters to `feature_sample` to maintain consistency with standard NLP and machine learning conventions.

### 4. `reduce_dimensions_tool` — Add 3D Plotting Support
* **Issue:** `reduce_dimensions_tool` restricts `n_components` strictly to `2` due to hardcoded 2D scatter plot rendering.
* **Finding on Real Data:** High-dimensional text data (7,100+ terms) suffers from heavy overlap when forced onto a 2D linear plane (PCA captured only ~23.8% total variance in 2D).
* **Fix:** Extend the tool's plotting backend (e.g., using `mpl_toolkits.mplot3d`) to render 3D scatter plots when `n_components=3` is passed.

### 5. `binarize_labels_tool` — Clarify Use-Case in Docstring
* **Issue:** Unclear tool role in the pipeline workflow.
* **Finding on Real Data:** Downstream filtering tools (`pearson_filter_tool`, `spearman_filter_tool`) and visualization functions convert category labels internally on the fly.
* **Fix:** Update the docstring to explicitly clarify that `binarize_labels_tool` produces an $N \times K$ one-hot matrix intended for external multi-class/multi-label machine learning classifiers (e.g., Scikit-Learn models), rather than being a required prerequisite for internal lab tools.

### 6. `inspect_data_tool` & Matrix Tools — Support Index-Based Row Lookup
* **Issue:** `inspect_data_tool` only supports `"head"` or `"random"` inspection modes, and matrix previews are capped at the first $N$ rows.
* **Finding on Real Data:** Debugging specific document anomalies (e.g., cross-checking why documents #1313 or #2078 produced specific similarity or classification scores) was impossible without being able to query specific row indices.
* **Fix:** Add an `indices` parameter (e.g., `indices=[1313, 2078]`) to allow direct row-level lookup across both text inspection and matrix preview tools.