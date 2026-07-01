# 7 · Word-level recognition (IPA & orthographic)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/07_word_level/07_word_level_recognition.ipynb)

An **alternative recognition target** to the phoneme model (Notebook 5). Trains
character-level `Wav2Vec2CTCTokenizer` models (space → `|` word delimiter) on
whole words, reported with **WER / CER**. Reuses stages 1–4.

Set `TARGET` at the top:
- **`"ipa"`** — labels are word-form IPA (`dat hœʁt`); vocab = IPA characters.
- **`"orthography"`** — labels are Kölsch spelling (`un dann hammer`); vocab =
  letters, with `normalize_text` (digit-stripping included).

Both learn word boundaries natively (no post-processing). A single wrong
character fails the whole word, so word-level WER sits **above** the phoneme
error rate from Notebook 5 — expected, not a bug.

**Input:** segment manifest (Notebook 3) + word-form IPA (Notebook 4) or raw text.
**Output:** saved model + WER/CER, plus a character-distribution figure.
