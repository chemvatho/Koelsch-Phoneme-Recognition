# 9 · Forced alignment — phone boundaries in time

Stage 6 says **what** was said; this says **when** each phone starts and ends.
The Kölsch model from stage 5 plus `torchaudio.functional.forced_align`, exported
as Praat **TextGrids** with a word tier and a phone tier.

No pronunciation dictionary needed beyond `kolsch_g2p.py` — which is the point.
MFA and MAUS both need a German lexicon, and **94.1 % of Kölsch word tokens are
OOV** against the 152,766-word `german_mfa` dictionary.

| | | |
|---|---|---|
| **9** | [`09_forced_alignment.ipynb`](09_forced_alignment.ipynb) | produce TextGrids |
| **9b** | [`09b_gap_rules_compared.ipynb`](09b_gap_rules_compared.ipynb) | every gap rule measured on your own data |
| **9c** | [`09c_periodic_energy.ipynb`](09c_periodic_energy.ipynb) | voicing onset as a cue — and whether your audio carries it |

All three import `kolsch_align.py` at the repo root, so the rules are implemented
once and the notebooks cannot drift apart.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/Koelsch-Phoneme-Recognition/blob/main/09_alignment/09_forced_alignment.ipynb)

**Input:** segment manifest (notebook 3) + a trained checkpoint (notebook 5).
**Output:** one TextGrid per segment in `data/textgrids/` (git-ignored).

```bash
export KOLSCH_MODEL=/path/to/models/kolsch_wav2vec2_model
```

---

## Start with `absorb="vc"`

**`vc` is the recommended setting. If you read one section of this page,
read this one.**

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

![why the consonants grew](../docs/figures/why_consonants_grew.png)

That is exactly what happened: `/h/` in *høːt* came out **307 ms** where MFA said
10 ms and MAUS 100 ms, and `/b/` in *bɛsɐ* took **197 ms** while its own vowel
kept 42.

> Take MFA's 10 ms `/h/` as an illustration, not a target. It is below MFA's own
> ~30 ms duration floor — 4 of the 38 segments in that file are — and MAUS puts
> the same `/h/` at 100 ms. Where `/h/` really ends is not something these two
> references agree on.

## The gap rules

| mode | what happens to a blank run |
|---|---|
| `none` | nothing — phones keep only their labelled frames, and the TextGrid has holes |
| `even` | split down the middle |
| `hybrid` | cut at the **spectral-change peak** inside a word; posterior-weighted across word edges |
| **`vc`** ← **use this** | word-internally, C→V and V→C runs go to the **vowel**; everything else as `hybrid` |
| `vc-onset` | `vc`, plus word boundaries at the end of the pause |
| `vc-sil` | `vc-onset`, **plus explicit silence** — supersedes it; see the caveat below |

![what vc changes](../docs/figures/vc_vs_hybrid.png)

All three wav2vec2 gap rules on one chain. **Red arrows** are the 17
word-**internal** boundaries `vc` moves (from `hybrid`); **teal arrows** are the
9 **word onsets** `vc-onset` moves (from `vc`). The two sets are disjoint by
construction — `vc` never fires on a word-separator gap, and `vc-onset` fires
only there. So in `vc`, **word onsets are untouched** and cross-system
comparisons stay valid.

## What `vc` does not fix

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

## `vc-onset` and `vc-sil`, and why neither is the default

Placing a word onset is a different problem. Inside a word you decide which of
two phones owns a blank run. Across a word boundary the gap usually **contains a
real pause**, and the question is where the next word starts — which the energy
answers directly and the CTC posteriors do not.

`vc-onset` puts each word-boundary gap at the end of its silence. That fixes the
onset and leaves a worse bug behind: **the pause ends up inside the previous
phone.** On *wenker2* the final schwa of *ʃnaɪə* came out **800 ms** against
MFA's 160, because the 620 ms MFA labels as silence had nowhere else to go.

`vc-sil` fixes that properly. A word boundary is not one event but three — the
previous word ends, there is silence, the next word begins — so it ends the
previous phone at the **start** of the pause and begins the next at its **end**,
leaving a real hole that the TextGrid renders as an empty interval. That is what
MFA and MAUS do. The schwa drops from 800 ms to **245**.

Over **941 word boundaries on the 84 helga recordings**, against both references:

| median \|Δ\| | onset vs MFA | end vs MFA | onset vs MAUS | end vs MAUS |
|---|---|---|---|---|
| `vc` | 62.1 ms | 57.2 ms | 73.9 ms | 70.3 ms |
| `vc-onset` | **40.0** | 55.0 | **45.4** | 55.4 |
| **`vc-sil`** | **40.0** | **50.0** | **45.4** | **52.9** |

Median silence emitted per recording: `vc` and `vc-onset` **0 ms**, `vc-sil`
250, MAUS 100, MFA 415. **`vc-sil` dominates `vc-onset` everywhere** — if you
want one of the two, take `vc-sil`.

> ### And yet both lose against the only human boundaries we have
>
> Two hand-cut excerpts give three speech-flanked edges a person placed in Praat.
> Mean distance to the nearest system boundary:
>
> | | mean | dann offset | wööd onset | wööd offset |
> |---|---|---|---|---|
> | **`vc`** | **7.2 ms** | 3.2 | 6.8 | 11.7 |
> | `vc-onset` / `vc-sil` / `pe` | 62.0 ms | 36.0 | 26.0 | 124.0 |
> | MAUS | 80.8 ms | 78.1 | 68.1 | 96.1 |
> | MFA | 123.7 ms | 121.0 | 111.0 | 139.0 |
>
> ![the hand-cut excerpt edges](../docs/figures/handcut_excerpts.png)
>
> The onset-aware modes move *towards MFA and MAUS and away from the human*. The
> obvious reading is that they are learning a convention those two share — common
> HMM-GMM lineage, both put a boundary at the edge of a silence interval — rather
> than getting closer to the truth. Agreement with them is not accuracy.
>
> **Three boundaries from one speaker is an anecdote**, and 941 boundaries
> against two related systems is not ground truth either. They disagree, and
> neither settles it.

**[Notebook 9b](09_alignment/09b_gap_rules_compared.ipynb) runs every gap rule
on your own data** and measures the three things that need no reference — how
much of the TextGrid is invented, where each rule puts the blank run, and whether
phones swallow the pauses — plus a reference comparison that activates if you
supply TextGrids from MFA, MAUS or a human.

So **`vc` stays the default**, and `vc-sil` is there for when you want a TextGrid
whose phones do not span pauses — which for TTS training data or for anything a
phonetician will open in Praat is usually what you want. The question is open
until there is a hand-corrected reference, which is why one heads the
[Roadmap](../README.md#roadmap).

## What none of them fixes

The boundary *between* a consonant and the vowel after it. `/h/` still ends at
0.522 s where MFA ends it at 0.470, so *øː* stays 241 ms against MFA's 290. The
C→V rule hands the blank run to the vowel starting at the consonant's spike
**end**, and for a fricative that spike runs on as long as the frication does.

There is no agreed target to fix it against — MFA says `/h/` is 10 ms, MAUS says
100. Past this point the honest move is to learn the boundary from the
**acoustics of the specific transition** — release burst, formant transition,
frication onset, voicing — rather than a phone-class lookup.

[`pe`](#pe--voicing-onset-as-a-measurement) does that for one of those cues,
voicing, and it does move the class of boundary this section is about: voiceless
fricative → vowel goes 30.4 → 21.6 ms against MFA across 205 instances. **It does
not move this particular token** — `/h/` in *høːt* ends at 0.525 s under `pe`
against 0.522 under `vc-sil`, a 3 ms change in the wrong direction. The rule
improves a distribution; it does not repair every member of it, and the token
this section was written around remains one it gets wrong.

## Does it help?

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

## `pe` — voicing onset as a measurement

Every rule above decides a blank run from the CTC posteriors, the phone classes,
or a broadband spectral-change curve. **None of them can tell noise from
voicing**, so all of them place the edge of a fricative by the logic they use for
a nasal.

`pe` adds a measurement. Following **ProPer — PROsodic analysis with PERiodic
energy** (Albert, Cangemi, Ellison & Grice, IfL Phonetik / SFB 1252, University
of Cologne, <https://osf.io/28ea5/>) it measures not how loud the signal is but
**how much of its energy is periodic**, with the zero of the scale set by the
loudest *aperiodic* frame in the recording — so a fricative sits at 0 dB by
construction, however loud it is. Where a gap runs from an obstruent to a
sonorant, the boundary goes at **voicing onset**. After a stop that is the end of
VOT, so closure, burst and aspiration stay inside the consonant instead of being
scored as vowel.

Periodicity comes from the average magnitude difference function rather than
Praat, so the whole thing is one dependency-free file, `kolsch_periodic.py`.

![the rule firing, and the rule declining to fire](../docs/figures/periodic_energy_cases.png)

**It knows when not to apply itself.** A voiced obstruent has no voicing onset —
intervocalic `/h/` is breathy voiced and never stops being periodic — so where
the curve never approaches the floor there is nothing to find, and the blank goes
to the sonorant instead. The right-hand panel above is that case.

Same 84 recordings, 941 word boundaries and 2717 word-internal ones, median
\|Δ\|. **Never worse than `vc-sil` on any bucket against either reference:**

| | `vc-sil` | **`pe`** |
|---|---|---|
| MFA · word-initial / internal / final-end | 40.0 / 26.0 / 50.0 | **37.5 / 22.7 / 47.5** |
| MAUS · word-initial / internal / final-end | 42.9 / 34.3 / 52.9 | 42.9 / **32.1** / 52.9 |
| voiceless fricative → vowel | 30.4 / 39.0 | **21.6 / 33.5** |
| voiced fricative → vowel | 30.9 / 33.4 | **20.1 / 17.7** |

![where it helps and where it does nothing](../docs/figures/periodic_class_pairs.png)

Four variants that sounded just as good were implemented, measured and rejected:
the mirror rule at voicing *offset* (offset is gradual — devoicing, creak — so it
is not a location); ProPer's own steepest-rise landmark (lands late into the
vowel); the landmarks at every gap (between two sonorants the curve is flat); and
trusting the crossing on voiced obstruents. `scripts/12_eval_periodic.py` in the
comparison harness runs all of them.

**One honest confound.** The gain at *voiced* fricative → vowel above is **not**
the periodic curve. Dropping an unrelated clause of the `vc` rule — its exception
for fricatives between two vowels — reaches 20.4 / 18.2 ms there with no curve at
all. The curve earns its place at *voiceless* fricatives, where dropping that
clause instead makes things **worse** (35.1 / 48.3). Two effects on two different
sets of phones, and only a control column separates them.

**And it needs a quiet recording, which it will not tell you unasked.**
Broadband noise fills in the AMDF minimum that periodicity is read from. This
project happens to have one corpus of each kind:

| | noise floor | periodicity when loud | vowel vs voiceless fricative |
|---|---|---|---|
| 84 field recordings, 16 kHz | −46 dB | 0.85 | d′ = **1.1** |
| 55 archival CD cuts *(the sample shipped here)* | −11 to −27 dB | 0.64 | d′ = **0.08** |

On the second kind `pe` moves boundaries on the strength of a curve that is
measuring hiss, and is strictly worse than `vc-sil`. `kolsch_periodic.cue_strength()`
answers this in one call; notebook 9c runs it first and **refuses to export in
`pe` if the cue is absent** — which on the shipped sample is what happens, 3 of
55 recordings passing. `vc` remains this repository's default for that reason
among others.

## Comparing against MFA and MAUS

![three aligners on one chain](../docs/figures/three_aligners.png)

Five rows, three aligners — wav2vec2 appears three times because the gap rule
changes what it can express. Give all of them the **same phone chain**, or you
are measuring three pronunciation dictionaries rather than three aligners.

The same sentence cut into its two prosodic segments and aligned independently,
on a shared time scale so one second is the same width in both panels:

![both segments, five systems](../docs/figures/three_aligners_chunks.png)

**Look at where the rows have holes.** MFA and MAUS model silence explicitly, so
a pause is an interval belonging to nobody. `vc` cannot say "silence" at all — its
phones tile the signal end to end, so every pause is inside whichever phone
happens to border it. `vc-sil` leaves the pauses unlabelled, and its holes line up
with the other two. **That is what makes its row structurally comparable to
theirs**; `vc`'s row is not, however close its boundaries land.

This matters for more than tidiness. A phone that swallows a 620 ms pause is
wrong as annotation whatever the boundary metric says — see the *ʃnaɪə* schwa
above, 800 ms under `vc-onset` against MFA's 160.

Neither MFA nor MAUS runs by default; notebook 9 documents both.

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
improvement is *understated* by it — and a change that merely makes wav2vec2
behave more like those two is *overstated*. Both distortions are live here:
`vc-sil` gains on the 941-boundary comparison and loses on the human edges, and
you cannot tell from these numbers alone how much of that gain is accuracy and
how much is conformity.

Against the only human-placed boundaries available — three edges of two hand-cut
excerpts — `vc` sat **7.2 ms** away on average, `vc-sil` and `pe` 62.0 (`pe`
changes none of those three edges), MAUS 80.8 and MFA
123.7. **Three boundaries from one speaker is an anecdote, not a result**, and it
points the opposite way from the 84-recording comparison. Getting past that
standoff needs a hand-corrected reference, which is the top
[Roadmap](../README.md#roadmap) item.

---


---

## Running MFA and MAUS yourself

Neither runs by default; notebook 9 documents both.

- **MFA** — Montreal Forced Aligner 2.2.17 + `german_mfa`. Runs entirely
  offline. Expect the 94.1 % OOV rate unless you supply a Kölsch lexicon.
- **MAUS** — the BAS webservice, `deu-DE`. **This uploads your audio.** It is an
  academic service in Munich, not a commercial one, but the recordings still
  leave your machine. That was an acceptable trade here for a comparison
  baseline; for recordings your speakers did not consent to share, it is not.
  Everything else in this repository runs locally.
