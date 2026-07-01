# Kölsch Tandem Project — dialect phoneme recognition (→ TTS)

Speech technology for **Kölsch** (Ripuarian German, Cologne), developed as part
of the **CIF Tandem Fellowship** at **IfL-Phonetik, University of Cologne**, in
tandem with **Simon Rössig**.

This repository holds the reproducible, six-stage pipeline that turns a printed
dialect corpus and its audio CDs into a fine-tuned **phoneme recogniser** for
Kölsch (Wav2Vec2 XLS-R-300M). It is stage one of the Tandem project; the same
data foundation feeds the planned **text-to-speech** work.

Each stage is a self-contained, Colab-ready notebook with its own README, so the
pipeline can be run end-to-end, reused for another dialect, or cited stage by
stage.

> **Tandem Fellowship.** A tandem pairs an incoming researcher with a local host
> to build a shared, transferable resource. Here the shared resource is an
> open, documented Kölsch speech-technology stack — usable by the IfL, the Royal
> Academy of Cambodia's tooling work, and the wider low-resource phoneme-recognition and TTS
> community.

> Source corpus: *Alles Kölsch* (Bhatt & Lindlar, 1998) — ~4 h of narrative
> speech, 125 speakers across 49 Cologne neighbourhoods.

---

## The pipeline

```
 book + CDs ─► 1 OCR ─► 2 Corpus ─► 3 Segment ─► 4 Normalise ─► 5 Fine-tune ─► 6 Analyse ─► phonemes
```

| # | Stage | What it does | Notebook |
|---|-------|--------------|----------|
| 1 | **OCR** | Digitise the printed transcriptions (Tesseract → EasyOCR → **Gemini 2.5 Pro**) | [`01_ocr/`](01_ocr/01_ocr_digitisation.ipynb) |
| 2 | **Corpus** | Token/type counts, speaker demographics, 44-phoneme distribution | [`02_corpus/`](02_corpus/02_corpus_statistics.ipynb) |
| 3 | **Segment** | MMS forced alignment + 5-word / 10-word / **prosodic** segmentation | [`03_segmentation/`](03_segmentation/03_mms_segmentation.ipynb) |
| 4 | **Normalise** | Phonological rules (a/b/c) + IPA phoneme tokeniser | [`04_normalisation/`](04_normalisation/04_phonological_normalisation.ipynb) |
| 5 | **Fine-tune** | Wav2Vec2 XLS-R-300M + CTC head | [`05_finetune/`](05_finetune/05_wav2vec2_finetune.ipynb) |
| 6 | **Analyse** | Test-set WER/CER + phoneme error analysis | [`06_analysis/`](06_analysis/06_error_analysis.ipynb) |
| 7 | **Word-level** *(alt. target)* | IPA word-level & orthographic recognition | [`07_word_level/`](07_word_level/07_word_level_recognition.ipynb) |

### Open in Colab
- 1 OCR — https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/01_ocr/01_ocr_digitisation.ipynb
- 2 Corpus — https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/02_corpus/02_corpus_statistics.ipynb
- 3 Segment — https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/03_segmentation/03_mms_segmentation.ipynb
- 4 Normalise — https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/04_normalisation/04_phonological_normalisation.ipynb
- 5 Fine-tune — https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/05_finetune/05_wav2vec2_finetune.ipynb
- 6 Analyse — https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/06_analysis/06_error_analysis.ipynb
- 7 Word-level — https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/07_word_level/07_word_level_recognition.ipynb

*(Replace `chemvatho/kolsch-tandem` with your own GitHub path once you push.)*

---

## Recognition targets

Stages 1–4 are shared data preparation. From there, three recognition targets are
available — all Wav2Vec2 XLS-R-300M + CTC, differing only in the label and vocab:

| Target | Notebook | Unit | Tokenizer | Output | Metric |
|--------|----------|------|-----------|--------|--------|
| **Phonemes** (primary) | 5 | IPA phoneme | `Wav2Vec2PhonemeCTCTokenizer` | `d a t \| ə s ʊ` | WER/CER (phoneme-level) |
| **IPA word-level** | 7 (`TARGET="ipa"`) | character | `Wav2Vec2CTCTokenizer` | `dat əsʊ` | WER/CER (word-level) |
| **Orthographic** | 7 (`TARGET="orthography"`) | character | `Wav2Vec2CTCTokenizer` | `dat esu` | WER/CER (word-level) |

The phoneme recogniser is the primary system. The word-level notebook is an
alternative target: a single wrong character fails the whole word, so word-level
WER sits above the phoneme error rate — expected, not a regression.

---

## Example data & scaling to many files

Every notebook is **global**: it iterates the registry `data/index.csv` (one row
per recording) and globs `data/pages/*.png` / `data/audio/*.wav`. Nothing is
hard-coded to a single file.

The repo ships **one worked example** — `page_1.png` (a scanned page) and
`track1_mono.wav` (its audio), which are the same recording: CD 1 / 01,
*"Tee mit Schuss"*, speaker Aldorf Klaus (51, Altstadt-Süd). It flows through
all stages: OCR the page → corpus stats → forced-align the audio to the text →
segment → phonemise → train → analyse.

**To scale up:** drop more scans in `data/pages/`, more recordings in
`data/audio/`, add a row per recording to `data/index.csv` — every notebook then
processes them all. See [`data/README.md`](data/README.md).

```
data/
├── index.csv          registry (id, page_image, audio, transcript, speaker, metadata)
├── pages/*.png        scanned pages     -> 1 OCR
├── audio/*.wav        recordings        -> 3 Segmentation
├── transcripts/*.txt  OCR output        -> 2 Corpus, 3 Segmentation
└── segments/          clips + manifest  -> 4 Normalise, 5 Fine-tune, 6/7
```

---

## Results

On the held-out test set (467 utterances, 17,455 reference phonemes), the
fine-tuned **Wav2Vec2 XLS-R-300M** reaches:

| Metric | Test |
|--------|------|
| **WER** (over phoneme tokens) | **14.63 %** |
| **CER** (over the IPA stream) | **11.75 %** |

Best validation checkpoint: 15.2 % WER / 11.8 % CER (selected on validation WER,
not loss). Error composition: 1,252 substitutions, 877 deletions, 425
insertions. For context, an off-the-shelf multilingual Wav2Vec2Phoneme scores
~33 % PER on comparable German-dialect material (Xu et al., 2022) — this
dialect-specific fine-tune more than halves that.

WER and CER are the metrics logged during training; because the target is a
phoneme sequence, this WER is a phoneme-token error rate (equivalent to PER).

---

## Roadmap

- [x] **Stage 1 — Phoneme recognition.** OCR → corpus → segmentation → normalisation → XLS-R-300M
      fine-tune → error analysis (this repo).
- [ ] **Stage 2 — TTS.** Reuse the aligned, phoneme-normalised corpus to train a
      Kölsch text-to-speech voice.
- [ ] **Shared resources.** Publish the pronunciation dictionary and the
      fine-tuned checkpoints for public use.

---

## Quickstart

```bash
git clone https://github.com/chemvatho/kolsch-tandem.git
cd kolsch-tandem
pip install -r requirements.txt
```

Then open any notebook in Jupyter or click its Colab badge. Stages are
independent: each reads the previous stage's output (described in its README)
and you can start from whichever stage you have data for.

**Data flow between stages**

```
scans ─►[1]─► ocr_txt/  ─►[2]─► statistics
                         └►[3]─► segments/ + manifest.csv ─┐
ipa  ──────────────────►[4]─► phonetic labels ────────────┤
                                                           ▼
                                          [5]─► model ─►[6]─► WER/CER + figures
```

---

## Data & licence

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Corpus:** the *Alles Kölsch* (Bhatt & Lindlar, 1998) text and audio are
  copyrighted by the rights holder and are **not** distributed here. You need
  your own licensed copy to reproduce the data stages. The notebooks contain
  small demo stubs so they run without the private corpus.

## Citation

```bibtex
@misc{chem2026kolschtandem,
  title  = {K\"olsch Tandem Project: dialect speech technology from print to phonemes},
  author = {Chem, Vatho and R\"ossig, Simon and Greisbach, Reinhold},
  year   = {2026},
  note   = {CIF Tandem Fellowship, IfL-Phonetik, University of Cologne},
  howpublished = {\url{https://github.com/chemvatho/kolsch-tandem}}
}
```

## Acknowledgements

Developed under the **CIF Tandem Fellowship** at IfL-Phonetik, University of
Cologne, in tandem with **Simon Rössig**. With thanks to Martine Grice,
Constantijn Kaland, and Reinhold Greisbach (co-author of the phoneme-recognition
study). Built on Meta AI's Wav2Vec2 / XLS-R and MMS, and Hugging Face
Transformers.
