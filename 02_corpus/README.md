# 2 · Corpus & speaker statistics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/02_corpus/02_corpus_statistics.ipynb)

Quantifies the digitised corpus: token/type counts and type–token ratio,
per-CD word distribution, speaker demographics (age histogram, neighbourhood
counts), and the 44-phoneme inventory distribution across train/valid/test.

### Run
- Provide `metadata.csv` (`speaker_id, cd, age, neighbourhood, transcript`,
  optionally `ipa` + `split`).
- Run all cells. The notebook ships with demo data so it runs out of the box.

**Output:** corpus statistics + `kolsch_phoneme_distribution.png`.
