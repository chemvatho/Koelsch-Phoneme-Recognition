# 3 · Audio segmentation & forced alignment (MMS)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/Koelsch-Phoneme-Recognition/blob/main/03_segmentation/03_mms_segmentation.ipynb)

Aligns 2–5-minute Kölsch narratives to their transcripts with **Meta's MMS
forced aligner**, then cuts them into trainable utterances using three
strategies: 5-word, 10-word, and **prosodic breath-group** (adopted).

The prosodic splitter cuts at sentence punctuation (hard) and at commas before
Kölsch clause-starters / after discourse particles (soft), targeting 3–10-word
intonation phrases.

### Run
- Provide recordings + corrected transcripts (from Notebook 1).
- Run all cells; use `prosodic_chunks(...)` then `export_segments(...)`.

**Output:** 16-kHz WAV clips + a manifest (`audio_path, text`) → input to Notebook 5.

### Note on torchaudio versions
`forced_align` / `merge_tokens` APIs differ across torchaudio releases. If yours
differs, the [`ctc-forced-aligner`](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
package (`pip install ctc-forced-aligner`) wraps MMS with a stable API.
