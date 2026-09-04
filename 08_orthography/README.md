# 8 · Orthographic recognition with W2v-BERT 2.0

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/Koelsch-Phoneme-Recognition/blob/main/08_orthography/08_w2vbert_orthography.ipynb)

The second training target. Stage 5 fine-tunes **XLS-R-300M** to IPA phonemes;
this fine-tunes **`facebook/w2v-bert-2.0`** to Kölsch **orthography** — the
spelling used in *Alles Kölsch*.

| | stage 5 | stage 8 |
|---|---|---|
| encoder | XLS-R-300M (~315M) | w2v-BERT 2.0 (~580M) |
| input | raw waveform | 80-bin log-mel, stride 2 |
| target | IPA phoneme | grapheme |
| headline | PER | **CER** |

## Two things before you compare them

**No German warm-start here.** `facebook/w2v-bert-2.0` is a raw pretrained
checkpoint with no CTC head, so `lm_head` *and* the conv adapter initialise
randomly. Stage 5 starts from a German-adapted checkpoint. The bigger model has
to pay for that difference — measure it, do not assume it wins.

**CER, not WER.** Kölsch has no spelling standard, so *janz*/*ganz* is one word
written two ways and WER charges full price. A scoring-only variant folding is
reported **alongside** the raw numbers, never instead of them; the fold is
guarded so real minimal pairs (`denn`/`den`, `jott`/`jot`) cannot collapse.

## Notes

- Apostrophes are **glued, not spaced**: `d'r` → `dr`. Kölsch elision writes one
  spoken word with an apostrophe inside it.
- `add_adapter=True` halves the frame rate to **40 ms**. Invisible in CER,
  very visible if you feed this model to notebook 9 — set it `False` if you
  intend to align with it.
- `KOLSCH_SMOKE=1` runs two optimiser steps to prove the wiring without
  pretending to be a training run.

**Input:** segment manifest (Notebook 3). **Output:** model + processor in
`models/kolsch_w2vbert_orthography/` (git-ignored).

On the shipped one-speaker example the split falls back to random and train and
test share a voice. The notebook says so at runtime; any score there is
memorisation.
