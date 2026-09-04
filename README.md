# Kölsch Phoneme Recognition

**Speech technology for Kölsch** (Ripuarian German, Cologne) — a reproducible
pipeline that turns a printed 1998 dialect corpus and its four audio CDs into a
fine-tuned phoneme recogniser, an orthographic recogniser, and phone-level
forced alignments.

Built during the **CIF Tandem Fellowship** at **IfL-Phonetik, University of
Cologne**, in tandem with **Jun.-Prof. Dr. Simon Rössig** and **Prof. Dr.
Reinhold Greisbach**, with the **Akademie för uns kölsche Sproch** as corpus
partner.

> **Kölsch has no public speech dataset and no standardised spelling.** Against
> the pretrained 152,766-word `german_mfa` dictionary, **94.1 % of Kölsch word
> tokens are out of vocabulary** — the dialect is simply not German text. Every
> design decision here follows from that.

> **Tandem Fellowship.** A tandem pairs an incoming researcher with a local host
> to build a shared, transferable resource. Here that resource is an open,
> documented Kölsch speech-technology stack — usable by the IfL, by the Royal
> Academy of Cambodia's tooling work, and by the wider low-resource
> phoneme-recognition and TTS community.

> **Source corpus:** *Alles Kölsch* (Bhatt & Lindlar, 1998) — ~4 h of narrative
> speech, 125 speakers across 49 Cologne neighbourhoods.

---

## Contents

| | |
|---|---|
| [Quick start](#quick-start) | get an alignment out of it in ten minutes |
| [The pipeline](#the-pipeline) | nine notebooks, stage by stage |
| [**Forced alignment — start with `vc-onset`**](#forced-alignment--start-with-absorbvc-onset) | the recommended setting, and why |
| [Two recognition targets](#two-recognition-targets) | IPA phonemes vs orthography |
| [OCR](#ocr-why-the-character-set-decides-everything) | why the character set decides the project |
| [Where it ran and where it didn't](#where-it-ran-and-where-it-didnt) | honest test status |
| [Data, rights and consent](#data-rights-and-consent) | what is in `data/` and whose it is |
| [Install](#install) · [Layout](#repository-layout) · [Citation](#citation) | |

---

## Quick start

```bash
git clone https://github.com/chemvatho/Koelsch-Phoneme-Recognition.git
cd Koelsch-Phoneme-Recognition
pip install -r requirements.txt
```

Then run the notebooks in order. Stages 3 → 4 must run before 5, 7, 8 or 9,
because stage 3 writes the segment manifest and stage 4 adds the phonetic
columns to it.

```
3  segment   →  4  normalise  →  5  fine-tune  →  6  analyse
                              →  8  orthography (w2v-BERT)
                              →  9  forced alignment      ← start here if you
                                                            already have a model
```

Model weights are **not** in this repository — the checkpoint is ~1.2 GB, well
past what git should carry. Train your own in stage 5, or point
`KOLSCH_MODEL` at a local directory or a Hugging Face repo id.

---

## The pipeline

```
book + CDs ─► 1 OCR ─► 2 Corpus ─► 3 Segment ─► 4 Normalise ─┬─► 5 Fine-tune ─► 6 Analyse
                                                             ├─► 7 Word-level
                                                             ├─► 8 Orthography (w2v-BERT)
                                                             └─► 9 Forced alignment
```

| # | Stage | What it does | Notebook |
|---|-------|--------------|----------|
| 1 | **OCR** | Digitise the printed transcriptions (Tesseract → EasyOCR → **Gemini 2.5 Pro**) | [`01_ocr/`](01_ocr/01_ocr_digitisation.ipynb) |
| 2 | **Corpus** | Token/type counts, speaker demographics, phoneme distribution | [`02_corpus/`](02_corpus/02_corpus_statistics.ipynb) |
| 3 | **Segment** | MMS forced alignment + fixed-window / **prosodic** segmentation | [`03_segmentation/`](03_segmentation/03_mms_segmentation.ipynb) |
| 4 | **Normalise** | Phonological rules + IPA phoneme tokeniser, dictionary-first G2P | [`04_normalisation/`](04_normalisation/04_phonological_normalisation.ipynb) |
| 5 | **Fine-tune** | Wav2Vec2 **XLS-R-300M** + CTC head → IPA phonemes | [`05_finetune/`](05_finetune/05_wav2vec2_finetune.ipynb) |
| 6 | **Analyse** | Test-set PER/CER + phoneme error analysis | [`06_analysis/`](06_analysis/06_error_analysis.ipynb) |
| 7 | **Word-level** | IPA word-level & orthographic recognition on XLS-R | [`07_word_level/`](07_word_level/07_word_level_recognition.ipynb) |
| 8 | **Orthography** | **W2v-BERT 2.0** + CTC head → Kölsch spelling | [`08_orthography/`](08_orthography/08_w2vbert_orthography.ipynb) |
| 9 | **Alignment** | Phone boundaries in time → Praat TextGrids | [`09_alignment/`](09_alignment/09_forced_alignment.ipynb) |

---

## Forced alignment — start with `absorb="vc-onset"`

**`vc-onset` is the recommended setting. If you read one section of this
README, read this one.**

CTC is **peaky**. The model emits one confident frame per phone and blanks in
between, so the labelled frames cover only **14–21 %** of the timeline on the
reference material (31 % on the shipped example). That means:

> **Roughly four fifths of every phone duration in a CTC-derived TextGrid is not
> measured. It is a rule deciding who gets the blank frames.**

Which rule you pick therefore matters more than the model does — and the obvious
rules are all wrong in the same way. They assume each CTC spike sits in the
middle of its phone. It does not:

| | spike sits … through its own segment |
|---|---|
| **vowels** | **71 %** — late |
| **consonants** | **10 %** — early |

*Measured against MFA over one recording and its two halves, n = 27 vowels /
35 consonants. Unstable in detail — the consonant median is 29 % on the full file
and 6 % on the chunks, because MFA re-segments each independently — but the
direction is stable.*

So the blank run between a consonant spike and the following vowel spike starts
at the **beginning** of the consonant and ends **two thirds into** the vowel.
Split it down the middle and the consonant eats half the vowel.

![why the consonants grew](docs/figures/why_consonants_grew.png)

That is exactly what happened: `/h/` in *høːt* came out **307 ms** where MFA said
10 ms and MAUS 100 ms, and `/b/` in *bɛsɐ* took **197 ms** while its own vowel
kept 42.

> Take MFA's 10 ms `/h/` as an illustration, not a target. It is below MFA's own
> ~30 ms duration floor — 4 of the 38 segments in that file are — and MAUS puts
> the same `/h/` at 100 ms. Where `/h/` really ends is not something these two
> references agree on.

### The gap rules

| mode | what happens to a blank run |
|---|---|
| `none` | nothing — phones keep only their labelled frames, and the TextGrid has holes |
| `even` | split down the middle |
| `hybrid` | cut at the **spectral-change peak** inside a word; posterior-weighted across word edges |
| `vc` | word-internally, C→V and V→C runs go to the **vowel**; everything else as `hybrid` |
| **`vc-onset`** ← **use this** | `vc`, plus word boundaries placed at the end of the pause |

![what vc changes](docs/figures/vc_vs_hybrid.png)

Red arrows mark every internal boundary that moved. **Word onsets are untouched
by construction**, so cross-system comparisons stay valid.

### What `vc` does not fix

That last sentence is a methodological virtue and an accuracy problem at the same
time, and the second half deserves saying out loud: **the largest remaining error
is the one the rule refuses to touch.** Split the same alignment by position:

| onset position | vs MFA | vs MAUS |
|---|---|---|
| **word-internal** — where `vc` applies | **11.7 ms** | **18.0 ms** |
| **word-initial** — where it does not | **113.0 ms** | **66.5 ms** |

*Median \|Δ\|, one recording plus its two halves, n = 52 internal / 21 initial.*

Word-internal boundaries are close to both references. Word-initial ones are
roughly ten times worse. The `/h/` of *høːt* is an ordinary instance: `vc` starts
it at 0.341 s where MFA starts it at 0.460 — **119 ms early** — because that
boundary is a *word onset* and falls back to the posterior-weighted split.

This also explains the vowel. `vc` gives *øː* 240 ms against MFA's 290, but its
**right** edge is already right (0.762 vs 0.760 s, 2 ms apart). The vowel is
short at its **left** edge, because the `/h/` in front of it starts too early.
One cause, two symptoms.

### The fix: `absorb="vc-onset"`

Placing a word onset is a different problem. Inside a word you are deciding
which of two phones owns a blank run. Across a word boundary the gap usually
**contains an actual pause**, and the question is where the next word starts —
which the energy answers directly and the CTC posteriors do not. MFA and MAUS
both get this for free because they model silence explicitly; `vc` did not.

`vc-onset` keeps `vc` word-internally and puts each word-boundary gap at **the
end of its silence**, falling back to the flux peak when the words run together.
Validated on **941 word-initial onsets across the 84 helga recordings**, against
both references independently:

| word-initial onset | vs MFA | vs MAUS |
|---|---|---|
| `vc` — posterior-weighted | 62.2 ms | 72.9 ms |
| **`vc-onset` — speech onset** | **42.5 ms** | **42.9 ms** |

Word-internal boundaries are untouched, to 0.1 ms. On the `/h/` of *høːt* the
onset moves from 0.341 s to **0.455** against MFA's 0.460 — **5 ms out, from
119** — and `/h/` shrinks from 181 ms to 67, now between MFA's 10 and MAUS's 100
rather than far past both.

> **One rule was tried and rejected, and it is worth knowing why.** Applying the
> vowel rule at word onsets looked good on wenker2 — 113.0 → 90.0 ms against MFA.
> On helga's 941 boundaries it is **worse than doing nothing** (62.2 → 67.5). It
> was fitting one speaker. The same happened to a class-rule fallback for
> pause-less gaps: better on wenker2, 42.5 → 50.0 on helga. **Where 21 boundaries
> and 941 boundaries disagree, believe the 941.** This is the same trap `vc`
> itself was fitted in, and only the larger set catches it.

### What is still not fixed

`vc-onset` fixes the word onset. It does **not** fix the boundary *between* a
consonant and the vowel after it: `/h/` still ends at 0.522 s where MFA ends it
at 0.470, so *øː* stays 241 ms against MFA's 290. The C→V rule hands the blank
run to the vowel starting at the consonant's spike **end**, and for a fricative
that spike runs on as long as the frication does.

There is no agreed target to fix it against: MFA says `/h/` is 10 ms, MAUS says
100. Beyond this point the honest move is to learn the boundary from the
**acoustics of the specific transition** — release burst, formant transition,
frication onset, voicing — rather than from a phone-class lookup. That needs a
hand-corrected reference first, and it is the reason one is the top item under
[Roadmap](#roadmap).

### Does it help?

Over **84 field recordings, 3,742 phones**. Both runs hold word onsets fixed,
so this isolates the word-internal change — it is a `hybrid` vs `vc`
comparison, and `vc-onset` differs from `vc` only at word boundaries:

| | `hybrid` | `vc` |
|---|---|---|
| three-way spread, median | 72.9 ms | **67.4 ms** |
| all three systems within 50 ms | 38.2 % | **41.4 %** |
| wav2vec2 vs MFA, median \|Δ\| | 37.5 ms | **34.0 ms** |
| wav2vec2 vs MAUS, median \|Δ\| | 49.6 ms | **43.7 ms** |

**68 recordings improve, 15 get worse**, median change −3.0 ms. On material where
the phone chain comes from a real manual transcription rather than a prompt text,
the effect is larger: median three-way spread **55.0 → 42.9 ms**.

**Read the per-class split before believing the headline.**

| | change in three-way spread |
|---|---|
| diphthong | −13.5 ms |
| vowel_long | −11.7 |
| vowel_short | −7.5 |
| affricate | −1.5 |
| plosive | +0.8 |
| fricative | +1.1 |
| approximant | +1.8 |
| nasal | +2.1 |

Every vocalic class improves; every consonantal class is flat or very slightly
worse. Vowels gain more than consonants lose, so the total moves the right way —
but **"vc is better" is a net claim, not a uniform one.**

### Comparing against MFA and MAUS

![three aligners on one chain](docs/figures/three_aligners.png)

Give all three systems the **same phone chain**, or you are measuring three
pronunciation dictionaries rather than three aligners. Notebook 9 documents both;
neither runs by default.

- **MFA** — Montreal Forced Aligner 2.2.17 + `german_mfa`. Runs entirely offline.
  Expect the 94.1 % OOV rate unless you supply a Kölsch lexicon.
- **MAUS** — the BAS webservice, `deu-DE`. **This uploads your audio.** It is an
  academic service in Munich, not a commercial one, but the recordings still
  leave your machine. That was an acceptable trade here for a comparison
  baseline; for recordings your speakers did not consent to share, it is not.
  Everything else in this repository runs locally.

**These are agreement figures, not accuracy figures.** With no hand-corrected
reference, "which aligner is right" is not answerable. The three-way spread says
how far apart the systems place the same boundary; the median-of-three leans
toward MFA and MAUS, which share an HMM-GMM lineage, so a wav2vec2-specific
improvement is *understated* by it.

Against the only human-placed boundaries available — three edges of two
hand-cut excerpts — wav2vec2 sat **7.2 ms** away on average, MFA 123.7 ms and
MAUS 80.8 ms. **Three boundaries from one speaker is an anecdote, not a result.**
It is reported because it points the same way as the 84-recording comparison, not
as evidence on its own.

---

## Two recognition targets

Stages 1–4 are shared. From there the corpus feeds two different products:

| | **stage 5 — phonemes** | **stage 8 — orthography** |
|---|---|---|
| encoder | XLS-R-300M (~315M) | **w2v-BERT 2.0** (~580M) |
| input | raw waveform | 80-bin log-mel, stride 2 |
| target | IPA phoneme (48 symbols) | Kölsch grapheme |
| output | `d a t \| ə s ʊ` | `dat esu` |
| headline metric | **PER** | **CER** |

**They are different products, not competing runs.** Use phonemes for anything
needing a phone inventory — TTS, alignment, dialectometry — and orthography when
you want text a Kölsch reader can read.

Two things to know before comparing them:

1. **W2v-BERT gets no German warm-start.** `facebook/w2v-bert-2.0` is a raw
   pretrained checkpoint with no CTC head, so `lm_head` *and* the conv adapter
   initialise randomly. The stronger multilingual pretraining has to pay for
   that. Do not assume the bigger model wins — measure it.
2. **CER is the headline for orthography, not WER.** With no spelling standard,
   *janz*/*ganz* and *zusamme*/*zosamme* are one word spelled two ways. WER
   charges full price. Notebook 8 also reports a scoring-only variant folding
   *alongside* the raw numbers — never instead of them.

### Results — phoneme model

Held-out test set, **467 utterances / 17,455 reference phonemes**:

| metric | test |
|---|---|
| **WER** (over phoneme tokens — i.e. PER) | **14.63 %** |
| **CER** (over the IPA stream) | **11.75 %** |

Best validation checkpoint 15.2 % WER / 11.8 % CER. Error composition: 1,252
substitutions, 877 deletions, 425 insertions. For context, an off-the-shelf
multilingual Wav2Vec2Phoneme scores ~33 % PER on comparable German-dialect
material (Xu et al., 2022) — this dialect-specific fine-tune more than halves
that.

Training cost roughly 23,400 steps / 200 epochs in 9 h 32 m on four hours of
audio. **Validation loss bottoms out around step 6,000 and then climbs while the
error rate keeps falling** — select the checkpoint on error rate, not on loss.

### Roadmap

- [x] **Stage 1 — phoneme recognition.** OCR → corpus → segmentation →
      normalisation → XLS-R-300M fine-tune → error analysis.
- [x] **Stage 2 — orthography and alignment.** W2v-BERT 2.0 grapheme model
      (notebook 8) and phone-level forced alignment (notebook 9).
- [ ] **Stage 3 — TTS.** Reuse the aligned, phoneme-normalised corpus to train a
      Kölsch text-to-speech voice.
- [ ] **Shared resources.** Publish the pronunciation dictionary and the
      fine-tuned checkpoints to the Hugging Face Hub.

---

## OCR: why the character set decides everything

![the OCR pipeline](docs/figures/ocr_pipeline.png)

A classical OCR engine can only output a character that is **in its character
set**. A glyph it does not know is not flagged — it is silently replaced by the
nearest one it does know.

That single fact decided the project. *Alles Kölsch* ships two versions of every
text: a quasi-orthographic one in standard German letters, and a phonetic one in
Rheinische Dokumenta. The phonetic version is the linguistically richer source
and the one OCR destroys, because its diacritics are exactly the characters no
German-trained engine has.

![why the phonetic route failed](docs/figures/ocr_why_fails.png)

Three engines were compared. **Tesseract** and **EasyOCR** both dropped the
elision apostrophes (`d'r`, `m'r`) and mangled the diacritics; **Gemini 2.5 Pro**,
prompted page by page, was selected — and **every page was then proofread against
the scan by a human**. OCR output is never taken as final.

---

## Where it ran and where it didn't

Every notebook was executed in a clean kernel on this machine
(Python 3.12, torch 2.13 + CUDA, transformers 5.14.1, one NVIDIA GB10) using
[`tools/run_notebook.py`](tools/run_notebook.py). Status as tested:

| notebook | status | time | note |
|---|---|---|---|
| 2 · Corpus | **pass** | 2 s | |
| 3 · Segment | **pass** | 14 s | downloads MMS_FA, writes 55 segments |
| 4 · Normalise | **pass** | 4 s | must run *after* 3 — stage 3 rewrites the manifest |
| 5 · Fine-tune | **pass** | 812 s | full 150-epoch run on the example |
| 6 · Analyse | **pass** | 21 s | |
| 7 · Word-level | **pass** | 635 s | |
| 8 · Orthography | **pass** | 35 s | smoke run (`KOLSCH_SMOKE=1`), 2 optimiser steps, downloads w2v-BERT 2.0 |
| 9 · Alignment | **pass** | 31 s | 55 TextGrids written |
| 1 · OCR | **not run** | — | needs the Tesseract **binary** and a `GEMINI_API_KEY`; neither is available in a headless test |

Three real bugs surfaced and were fixed rather than worked around:

- **`group_by_length` was removed in transformers 5.0**, so stage 5 raised
  `TypeError` on any current install. The cell now filters its kwargs against
  `TrainingArguments`' actual signature and prints what it dropped — one notebook
  that works on both 4.x and 5.x, instead of a version pin that will rot.
- **Stage 6's confusion plot crashed when there were no substitutions**
  (`M.max()` on a 0×0 array raises rather than returning 0). It now says so and
  returns.
- **`opencv-python` was missing from `requirements.txt`** although stage 1
  imports `cv2`. Added, along with `scikit-learn` and a note that the Tesseract
  binary cannot come from pip.

**On the shipped example, training notebooks prove the wiring, not the model.**
`data/` holds one recording by one speaker, so there is nothing to hold out: the
split falls back to random, train and test share a voice, and any score is
memorisation. Both notebook 8 and notebook 7 say so at runtime. Scale `data/` up
before believing a number.

Notebook 9's demo on that example measured CTC covering **31 %** of the span and
`vc` moving vowels **+15.8 ms** and consonants **−6.9 ms** on average — the
expected direction, on 33 phones.

---

## Data, rights and consent

`data/` ships **one worked example** so the pipeline runs out of the box:

| file | what it is |
|---|---|
| `pages/page_1.png` | scan of the printed transcription page |
| `audio/track1_mono.wav` | CD 1, track 01 — 2.8 min, *"Tee mit Schuss"* |
| `transcripts/page_1.txt` | the OCR output for that page, human-proofread |
| `index.csv` | one row per recording, with the book's own speaker metadata |
| `lexicon.csv` | Kölsch orthography → IPA, generated by `kolsch_g2p.py` |

**Source and rights.** The page and the audio are from *Alles Kölsch*
(Bhatt & Lindlar, 1998), published by the **Akademie för uns kölsche Sproch**,
which holds the rights. One page and one track are included as a worked example
so the pipeline runs out of the box.

> **They are excluded from this repository's MIT licence, which covers the code
> only.** Nothing here grants you a licence to the corpus. To use or
> redistribute the page or the audio — or to obtain the full corpus, which is
> not published here — contact the Akademie för uns kölsche Sproch.

**Speaker metadata.** `index.csv` carries the speaker's name, age, occupation and
neighbourhood. These are reproduced from the book's own published speaker table,
where they have been in print since 1998; nothing here discloses more than the
publication does. If you extend `data/` with recordings that are *not* already
published, do not copy this pattern — use an opaque speaker id.

**Everything downstream runs locally.** The recogniser is fine-tuned in-house,
alignment runs offline, and audio, lexicon, phone inventory and TextGrids never
leave your machine. The two exceptions are opt-in and named as such: Gemini reads
the **printed page** in notebook 1, and MAUS is an **optional** comparison
baseline in notebook 9. A workable rule: *published text may travel; speaker
recordings should not.*

---

## Install

```bash
pip install -r requirements.txt
```

Notebook 1 additionally needs the Tesseract **binary**, which pip cannot install:

```bash
apt-get install tesseract-ocr tesseract-ocr-deu     # Debian/Ubuntu
brew install tesseract tesseract-lang               # macOS
```

and a Gemini key in the environment:

```bash
export GEMINI_API_KEY=...        # never commit this
```

Notebook 9 needs a trained checkpoint:

```bash
export KOLSCH_MODEL=/path/to/models/kolsch_wav2vec2_model_all
export KOLSCH_PROCESSOR=/path/to/processor      # defaults to KOLSCH_MODEL
```

Every notebook locates the repository root from `kolsch_paths.py`, so it runs
identically in VS Code, Jupyter and Colab regardless of the working directory.

---

## Repository layout

```
01_ocr/ … 09_alignment/     one notebook + README per stage
data/                       the worked example (see rights above)
docs/figures/               figures used by this README
tools/run_notebook.py       executes a notebook and reports where it stops
kolsch_g2p.py               rule-based Kölsch grapheme→phoneme converter
kolsch_paths.py             single source of truth for paths
requirements.txt
```

`models/` is generated and git-ignored. Publish checkpoints to the Hugging Face
Hub rather than committing them — the phoneme model alone is 1.2 GB.

---

## Citation

```bibtex
@misc{chem2026koelsch,
  author = {Chem, Vatho and R\"ossig, Simon and Greisbach, Reinhold},
  title  = {K\"olsch Phoneme Recognition: an open pipeline from a printed
            dialect corpus to phoneme recognition and forced alignment},
  year   = {2026},
  note   = {CIF Tandem Fellowship, IfL-Phonetik, University of Cologne},
  howpublished = {\url{https://github.com/chemvatho/Koelsch-Phoneme-Recognition}}
}
```

Corpus: Bhatt, C. & Lindlar, M. (1998). *Alles Kölsch: eine Dokumentation der
heutigen Sprache in Köln*. Akademie för uns kölsche Sproch.

## Licence

**MIT** for the code and notebooks. **Not** for `data/` — see
[Data, rights and consent](#data-rights-and-consent).

## Acknowledgements

Developed under the **CIF Tandem Fellowship** at IfL-Phonetik, University of
Cologne, in tandem with **Simon Rössig**. With thanks to **Martine Grice**,
**Constantijn Kaland**, and **Reinhold Greisbach** (co-author of the
phoneme-recognition study), and to the **Akademie för uns kölsche Sproch** as
corpus partner. Built on Meta AI's Wav2Vec2 / XLS-R / w2v-BERT and MMS, and
Hugging Face Transformers.
