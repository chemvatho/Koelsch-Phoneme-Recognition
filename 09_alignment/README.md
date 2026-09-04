# 9 · Forced alignment — phone boundaries in time

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/Koelsch-Phoneme-Recognition/blob/main/09_alignment/09_forced_alignment.ipynb)

Stage 6 says **what** was said; this says **when** each phone starts and ends.
The Kölsch model from stage 5 plus `torchaudio.functional.forced_align`, exported
as Praat **TextGrids** with a word tier and a phone tier.

No pronunciation dictionary needed beyond `kolsch_g2p.py` — which is the point.
MFA and MAUS both need a German lexicon, and **94.1 % of Kölsch word tokens are
OOV** against the 152,766-word `german_mfa` dictionary.

## Use `absorb="vc-onset"`

CTC is peaky: the labelled frames cover only 14–21 % of the timeline, so **most
of every phone duration in the output is a rule, not a measurement.** The obvious
rules assume each CTC spike sits mid-phone. It does not — vowel spikes sit ~71 %
through their segment, consonant spikes ~10 % — so an even split lets each
consonant eat half the following vowel.

`vc` hands each word-internal blank run to the **vowel**. Word onsets are
untouched by construction, so cross-system comparisons stay valid.

| mode | blank run goes to |
|---|---|
| `none` | nobody — the TextGrid keeps its holes |
| `even` | split down the middle |
| `hybrid` | the spectral-change peak inside a word |
| `vc` | the **vowel**, at C→V and V→C; otherwise as `hybrid` |
| **`vc-onset`** | `vc`, plus word boundaries at the end of the pause |

`vc` leaves word-boundary gaps alone, and that is where the error concentrates:
word-internal onsets sit 11.7 ms from MFA, word-initial ones 113.0. A word
boundary is a different question — where does speech resume after a pause — and
the energy answers it directly. Over **941 word-initial onsets on the 84 helga
recordings**, `vc-onset` cuts that error from 62.2 to **42.5 ms** against MFA and
72.9 to **42.9** against MAUS, leaving word-internal boundaries untouched.

Applying the vowel rule at word onsets instead was tried and rejected: better on
one recording, *worse than doing nothing* on helga. It was fitting one speaker.

Over 84 recordings / 3,742 phones: median three-way spread **72.9 → 67.4 ms**,
68 improve and 15 get worse. Every vocalic class improves and every consonantal
class is flat or slightly worse — a **net** gain, not a uniform one.

**Input:** segment manifest (Notebook 3) + a trained checkpoint (Notebook 5).
**Output:** one TextGrid per segment in `data/textgrids/` (git-ignored).

```bash
export KOLSCH_MODEL=/path/to/models/kolsch_wav2vec2_model_all
```

MFA and MAUS comparisons are documented in the notebook and **do not run by
default**. MAUS uploads your audio to the BAS webservice in Munich; everything
else here runs offline.
