# Lessons learned

Errors found in review. Apply to every new tool.

1. **Cap lists in the summary.** Default `top_n = 20`, plus the full counts. Uncapped lists blow up on real data. (term_frequency, variance, list_files)
2. **No `.toarray()` on the full matrix.** Too big on real data; compute on the sparse matrix. (variance, pearson, spearman)
3. **Summary must match the docstring.** If the docstring says it's returned, return it. (dtm_heatmap `doc_indices`)
4. **Honest labels.** A positional slice is "First N", not "Top N". (dtm_heatmap)
5. **Test what's claimed.** If it says "sorted", assert the order. (variance)
6. **Path check by folder, not text prefix.** `startswith` lets `../DM2026-Lab1-Exercise-x` through. (list_files)
7. **Validate list params.** `ngram_range` `[1]`, `[2,1]`, `[1,2,3]` break or pass silently. (build_dtm)
