# 9 · Forced alignment — phone boundaries in time

Three notebooks:

| | | |
|---|---|---|
| **9** | [`09_forced_alignment.ipynb`](09_forced_alignment.ipynb) | produce TextGrids |
| **9b** | [`09b_gap_rules_compared.ipynb`](09b_gap_rules_compared.ipynb) | decide which gap rule to trust, measured |
| **9c** | [`09c_periodic_energy.ipynb`](09c_periodic_energy.ipynb) | add an acoustic cue where there was only a rule — and check it exists in your audio first |

All three import `kolsch_align.py` at the repo root, so the rules are implemented once and the notebooks cannot drift apart.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/Koelsch-Phoneme-Recognition/blob/main/09_alignment/09_forced_alignment.ipynb)

Stage 6 says **what** was said; this says **when** each phone starts and ends.
The Kölsch model from stage 5 plus `torchaudio.functional.forced_align`, exported
as Praat **TextGrids** with a word tier and a phone tier.

No pronunciation dictionary needed beyond `kolsch_g2p.py` — which is the point.
MFA and MAUS both need a German lexicon, and **94.1 % of Kölsch word tokens are
OOV** against the 152,766-word `german_mfa` dictionary.

## Which `absorb` mode — `vc` by default

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
| `pe` | `vc-sil`, plus obstruent→sonorant onsets at **voicing onset**, read off a periodic-energy curve. Needs a quiet recording — see below |

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

Over 84 recordings / 3,742 phones `vc` gives a median three-way spread of
**72.9 → 67.4 ms**, 68 improving and 15 getting worse. Every vocalic class
improves and every consonantal class is flat or slightly worse — a **net** gain,
not a uniform one.

## `pe` — the one boundary the signal can answer directly

Every rule above decides a gap from the CTC posteriors, the phone classes, or a
broadband spectral-change curve. **None of them can tell noise from voicing**, so
they place the edge of a fricative by the same logic they use for a nasal.

`pe` adds a measurement: **periodic energy**, after
[ProPer](https://osf.io/28ea5/) (Albert, Cangemi, Ellison & Grice, IfL Phonetik /
SFB 1252, Cologne) — how much of the signal's energy is *periodic*, with the zero
of the scale set by the loudest **aperiodic** frame in the recording, so a
fricative sits at 0 dB by construction. Where a gap runs from an obstruent to a
sonorant, the boundary goes at **voicing onset**. After a stop that is the end of
VOT, so closure, burst and aspiration stay inside the consonant.

![the rule firing, and the rule declining to fire](../docs/figures/periodic_energy_cases.png)

Over the same 84 recordings — 941 word boundaries, 2717 word-internal ones —
median \|Δ\|, **never worse than `vc-sil` on any bucket against either
reference**:

| | vc-sil | **pe** |
|---|---|---|
| MFA · word-initial / internal / final-end | 40.0 / 26.0 / 50.0 | **37.5 / 22.7 / 47.5** |
| MAUS · word-initial / internal / final-end | 42.9 / 34.3 / 52.9 | 42.9 / **32.1** / 52.9 |
| voiceless fricative → vowel | 30.4 / 39.0 | **21.6 / 33.5** |

![where it helps and where it does nothing](../docs/figures/periodic_class_pairs.png)

**Four variants that sounded just as good and lost:** the mirror rule at voicing
*offset* (voicing offset is gradual — devoicing, creak — so it is not a
location); ProPer's own steepest-rise landmark (lands late into the vowel); the
landmarks applied to every gap (between two sonorants the curve is flat); and
trusting the crossing on *voiced* obstruents (there is no onset — intervocalic
/h/ never stops being voiced).

**One honest confound.** The gain at *voiced* fricative→vowel is **not** the
curve. Dropping an unrelated clause of the `vc` rule — its exception for
fricatives between two vowels — gets there with no curve at all. The curve earns
its place at *voiceless* fricatives, where dropping that clause instead makes
things worse.

### It needs a quiet recording, and it will not tell you

Broadband noise fills in the AMDF minimum that periodicity is read from. This
project has one corpus of each kind:

| | noise floor | periodicity when loud | vowel vs voiceless fricative |
|---|---|---|---|
| 84 field recordings, 16 kHz | −46 dB | 0.85 | d′ = **1.1** |
| 55 archival CD cuts *(the sample shipped here)* | −11 to −27 dB | 0.64 | d′ = **0.08** |

On the second kind `pe` moves boundaries on the strength of a curve that is
measuring hiss. Call `kolsch_periodic.cue_strength()` — section 1 of notebook 9c
does it for you and refuses to export in `pe` if the cue is absent, which on the
shipped sample is what happens (3 of 55 recordings pass).

**Input:** segment manifest (Notebook 3) + a trained checkpoint (Notebook 5).
**Output:** one TextGrid per segment in `data/textgrids/` (git-ignored).

```bash
export KOLSCH_MODEL=/path/to/models/kolsch_wav2vec2_model
```

MFA and MAUS comparisons are documented in the notebook and **do not run by
default**. MAUS uploads your audio to the BAS webservice in Munich; everything
else here runs offline.
