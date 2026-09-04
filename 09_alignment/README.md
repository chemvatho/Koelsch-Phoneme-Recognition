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
| `vc-onset` | `vc`, plus word boundaries at the end of the pause |
| **`vc-sil`** | `vc-onset` **plus explicit silence** — supersedes it |

`vc` leaves word-boundary gaps alone, and that is where the error concentrates:
word-internal onsets sit 11.7 ms from MFA, word-initial ones 113.0.

A word boundary is not one event but three — the previous word ends, there is
silence, the next word begins. `vc-onset` cuts once at the end of the pause,
which fixes the onset and leaves the pause inside the *previous* phone: the final
schwa of *ʃnaɪə* came out **800 ms** against MFA's 160. `vc-sil` ends the previous
phone at the pause's start and begins the next at its end, leaving an empty
interval — what MFA and MAUS do. The schwa drops to **245 ms**.

Over **941 word boundaries on the 84 helga recordings**, median \|Δ\|:

| | onset vs MFA | end vs MFA | onset vs MAUS | end vs MAUS |
|---|---|---|---|---|
| `vc` | 62.1 ms | 57.2 ms | 73.9 ms | 70.3 ms |
| `vc-onset` | **40.0** | 55.0 | **45.4** | 55.4 |
| **`vc-sil`** | **40.0** | **50.0** | **45.4** | **52.9** |

Silence emitted per recording (median): `vc` and `vc-onset` 0 ms, `vc-sil` 250,
MAUS 100, MFA 415.

**But both lose to `vc` against the only human-placed boundaries here** — 7.2 ms
for `vc` against 62.0 for both onset-aware modes. They move toward MFA and MAUS,
which share an HMM-GMM lineage, and away from the person. `vc` therefore stays
the default; use `vc-sil` when you want a TextGrid whose phones do not span
pauses, which is usually what you want for TTS data or for Praat.

Two rules were tried and rejected on the way: the vowel rule at word onsets
(better on one recording, *worse than doing nothing* on helga) and a class-rule
fallback for pause-less gaps. Both were fitting one speaker.

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
