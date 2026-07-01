# 4 · Phonological normalisation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/04_normalisation/04_phonological_normalisation.ipynb)

Cleans the semi-phonetic book orthography into phonology and tokenises each IPA
word into the 44-symbol Kölsch inventory.

**Rules:** (a) degemination · (b) double-vowel → long vowel · (c) silent
*Dehnungs-h* → length (intervocalic *h* kept). A longest-match tokeniser then
splits words into phonemes (`p h | p h`), with a zero-change safety pass.

**Input:** IPA column · **Output:** `phonetic` training labels for Notebook 5.
