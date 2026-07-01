# 6 · Test-set inference & error analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/06_analysis/06_error_analysis.ipynb)

Runs the fine-tuned model on the test set and reports **WER / CER** (the
training metrics) plus a phoneme-level error analysis: insertions/deletions per
phoneme, a substitution confusion matrix, and word-conditioned error tables.

**Input:** test manifest + saved model (Notebook 5).
**Output:** `errors_df`, `insertions_deletions.png`, `substitution_confusion.png`.
**Official test result:** WER 14.63% · CER 11.75% (467 utterances, 17,455 phonemes).
