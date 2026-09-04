# 1 · OCR digitisation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/Koelsch-Phoneme-Recognition/blob/main/01_ocr/01_ocr_digitisation.ipynb)

Converts the printed **Alles Kölsch** (Bhatt & Lindlar, 1998) transcription
pages into clean Unicode text, preserving the dialectal orthography.

**Engines:** Tesseract → EasyOCR → **Gemini 2.5 Pro** (selected). Gemini reads
pages in context and is prompted to transcribe faithfully without "correcting"
the dialect, so elision apostrophes (`d'r`, `m'r`), dialect spellings and
special characters are preserved.

### Run
1. Put scanned page images in `pages/` (`.png`/`.jpg`, sub-folders OK).
2. Set `GEMINI_API_KEY` (get one at aistudio.google.com/app/apikey).
3. Run all cells, then call `batch_ocr()`.
4. Proofread the `.txt` files in `ocr_txt/`.

**Input:** scanned pages · **Output:** mirrored `.txt` tree → used by notebooks 2 & 3.
