# Supplementary material package

The final CSSP supplementary file should contain the material needed to audit the claims without overloading the main text:

1. dataset sources, dimensions, labels, preprocessing, licenses, and loader variable names;
2. full algorithm pseudocode and the derivation of bilateral orthogonal-complement updates;
3. initialization, stopping, zero-vector fallback, denominator guards, and complexity;
4. complete hyperparameter table and q-selection rule;
5. raw per-repeat results, seeds, split indices, corruption masks and configuration hashes;
6. complete mean/SD/95% CI tables and paired statistical tests;
7. sensitivity to q, block size, corruption type, initialization and stopping tolerance;
8. convergence traces, runtime, memory, failure runs and NaN handling;
9. test-oracle results clearly separated from validation-selected main results;
10. high-resolution figures and `paper_table_figure_map.csv`.

The directory is a documentation placeholder in the review candidate. It must not be filled by copying legacy outputs without a new generation manifest.
