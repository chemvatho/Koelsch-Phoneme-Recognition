"""Forced alignment with the Kölsch CTC model, and the gap rules that decide it.

    from kolsch_align import Aligner, MODES
    al = Aligner(model_dir, processor_dir)
    phones, words, dur = al.align("clip.wav", "Als Kind han ich", mode="vc")

WHY THIS IS A MODULE AND NOT A CELL. CTC is peaky: the model emits one confident
frame per phone and blanks in between, so labelled frames cover only 14-21 % of
the timeline on the reference material. **Roughly four fifths of every phone
duration in a CTC-derived TextGrid is not measured -- it is a rule deciding who
gets the blank frames.** Which rule you pick therefore matters more than the
model does, notebooks 9 and 9b both need the rules, and two copies would drift.

THE MODES, in the order they were arrived at:

  none        phones keep only their labelled frames; the TextGrid has holes
  even        every blank run split down the middle
  weighted    split in proportion to the two CTC posteriors
  hybrid      cut at the spectral-change peak inside a word; weighted across
              word edges. The notebook default this project shipped with.
  vc          hybrid, plus: word-internally a C->V or V->C blank run goes
              entirely to the VOWEL
  vc-onset    vc, plus: a word-boundary gap is cut at the END of its pause
  vc-sil      vc-onset, plus: the pause is EMITTED as a hole rather than being
              handed to the phone in front of it

WHY vc. Every rule above it assumes each CTC spike sits mid-phone. Measured
against MFA over one recording and its two halves, a spike sits at 71 % through a
vowel (n=27) and 10 % through a consonant (n=35). So the blank between a
consonant spike and the following vowel spike runs from the START of the
consonant to two thirds into the vowel, and any midpoint-ish cut lands deep
inside the vowel: /h/ in høːt came out 307 ms under hybrid where MFA said 10.

WHY vc-sil. vc leaves word boundaries alone, and that is where the error
concentrates -- word-internal onsets sit 11.7 ms from MFA, word-initial ones
113.0. But a word boundary is not one event; it is three: the previous word ends,
there is silence, the next word begins. Cutting once at the end of the pause
(vc-onset) fixes the onset and leaves the pause inside the PREVIOUS phone -- the
final schwa of ʃnaɪə came out 800 ms against MFA's 160. vc-sil ends the previous
phone at the pause's start and begins the next at its end.

WHICH TO USE, honestly: vc is the default. Over 941 word boundaries on 84 helga
recordings vc-sil beats it against both MFA and MAUS (onset 62.1 -> 40.0 ms vs
MFA, 73.9 -> 45.4 vs MAUS), but against the only human-placed boundaries in this
material -- three hand-cut excerpt edges -- vc is 7.2 ms out and vc-sil is 62.0.
The onset-aware modes move toward MFA and MAUS, which share an HMM-GMM lineage,
and away from the person. Three boundaries is an anecdote; 941 against two
related systems is not ground truth. They disagree and neither settles it.
Use vc-sil when you want a TextGrid whose phones do not span pauses.
"""
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

MODES = ("none", "even", "weighted", "hybrid", "vc", "vc-onset", "vc-sil", "pe")
SEP = "|"
SAMPLE_RATE = 16_000

VOWEL_SHORT = set("a e œ ɐ ɔ ə ɛ ɪ ʊ ʏ y".split())
VOWEL_LONG = set("aː eː iː oː uː yː øː ɛː".split())
DIPHTHONG = set("aɪ aʊ ɔɪ ɔʏ ɛɪ".split())
# Both g's on purpose. The model published in models/ was trained with U+0261
# LATIN SMALL LETTER SCRIPT G, the IPA character; the larger model this project
# trains locally uses ASCII g. They look identical and are not, and the
# constructor check below turns that into an error rather than a phone silently
# classified as "not a plosive".
PLOSIVE = set("b d g ɡ k p t".split())
AFFRICATE = set("p͡f t͡s t͡ʃ".split())
FRICATIVE = set("f h s v z ç ʃ ʒ χ".split())
NASAL = set("m n ŋ".split())
APPROXIMANT = set("j l ʁ".split())

VOCALIC = VOWEL_SHORT | VOWEL_LONG | DIPHTHONG
SUSTAINED = FRICATIVE | AFFRICATE
KNOWN = VOCALIC | PLOSIVE | AFFRICATE | FRICATIVE | NASAL | APPROXIMANT
_SPECIAL = {SEP, "[PAD]", "[UNK]", "<s>", "</s>"}


@dataclass
class Segment:
    label: str
    start: float
    end: float
    score: float


# --------------------------------------------------------------------------- #
# signal
# --------------------------------------------------------------------------- #
def spectral_flux(wav, sr, hop_ms=2.5, n_mfcc=13):
    """Frame-to-frame spectral change: where the signal says a boundary is.

    c0 is dropped so this responds to spectral SHAPE -- formant and manner
    change -- rather than loudness, which would just track the envelope.
    """
    hop = max(1, int(hop_ms / 1000 * sr))
    m = librosa.feature.mfcc(y=np.asarray(wav, dtype=np.float32), sr=sr,
                             n_mfcc=n_mfcc, hop_length=hop,
                             n_fft=min(1024, max(256, 4 * hop)))[1:]
    m = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-8)
    flux = np.concatenate([[0.0], np.sqrt((np.diff(m, axis=1) ** 2).sum(axis=0))])
    return flux, np.arange(len(flux)) * hop / sr


def energy_envelope(wav, sr, hop_ms=5.0, win_ms=20.0, smooth=3):
    """Median-smoothed dB energy per frame, and each frame's time.

    dB against a robust floor rather than a fraction of peak amplitude: a linear
    threshold does not transfer between a clean studio recording and a field one.
    """
    hop, win = max(1, int(hop_ms / 1000 * sr)), max(1, int(win_ms / 1000 * sr))
    n = (len(wav) - win) // hop + 1
    if n < 2:
        return np.zeros(0), np.zeros(0)
    e = np.sqrt(np.array([np.mean(wav[i * hop:i * hop + win] ** 2)
                          for i in range(n)]))
    db = 20 * np.log10(e + 1e-8)
    if smooth > 1:
        k = smooth | 1
        pad = np.pad(db, k // 2, mode="edge")
        db = np.array([np.median(pad[i:i + k]) for i in range(n)])
    return np.arange(n) * hop / sr, db


def pause_span_in_gap(t0, t1, env_t, env_db, thr, min_ms=60.0):
    """Longest sub-threshold run in [t0, t1] as (start, end), or None."""
    if len(env_t) == 0:
        return None
    lo, hi = np.searchsorted(env_t, t0), np.searchsorted(env_t, t1)
    if hi <= lo:
        return None
    quiet, best, run = env_db[lo:hi] < thr, None, None
    for k, q in enumerate(quiet):
        if q:
            run = k if run is None else run
        elif run is not None:
            if best is None or k - run > best[1] - best[0]:
                best = (run, k)
            run = None
    if run is not None and (best is None or len(quiet) - run > best[1] - best[0]):
        best = (run, len(quiet))
    if best is None:
        return None
    a, b = float(env_t[lo + best[0]]), float(env_t[lo + best[1] - 1])
    return (a, b) if (b - a) * 1000 >= min_ms else None


# --------------------------------------------------------------------------- #
# the gap rules
# --------------------------------------------------------------------------- #
def merge_repeats(tokens, scores, blank_id, tokenizer):
    segs, i, n = [], 0, len(tokens)
    while i < n:
        tok = int(tokens[i])
        j = i
        while j < n and int(tokens[j]) == tok:
            j += 1
        if tok != blank_id:
            segs.append(Segment(tokenizer.convert_ids_to_tokens(tok), i, j,
                                scores[i:j].mean().item()))
        i = j
    return segs


def absorb_gaps(segments, flux, ftimes, spf, mode="vc",
                env_t=None, env_db=None, env_thr=None, drop_label=SEP,
                sustained_exception=True):
    """Decide who owns each blank run. See the module docstring for the modes.

    sustained_exception=False disables one clause of the vc rule -- the one that
    declines to hand a C->V blank to the vowel when the consonant is a fricative
    or affricate BETWEEN two vowels. It exists only so that the periodic-energy
    evaluation can tell its own effect apart from the effect of dropping that
    clause; see 12_eval_periodic.py. Leave it True.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    word_internal_only = mode in ("hybrid", "vc", "vc-onset", "vc-sil")
    class_split = mode in ("vc", "vc-onset", "vc-sil")

    spans_sep, k = set(), -1
    for seg in segments:
        if seg.label == drop_label:
            spans_sep.add(k)
        else:
            k += 1
    kept = [s for s in segments if s.label != drop_label]
    if not kept:
        return []
    if mode == "none":
        return [Segment(s.label, float(s.start), float(s.end), s.score)
                for s in kept]

    bounds, sil = [float(kept[0].start)], {}
    for i in range(len(kept) - 1):
        a, b = kept[i], kept[i + 1]
        if b.start <= a.end:
            bounds.append(float(a.end))
            continue
        gap = b.start - a.end

        if mode == "even":
            bounds.append(a.end + gap / 2)
            continue
        if mode == "weighted" or (i in spans_sep and not word_internal_only):
            tot = a.score + b.score
            bounds.append(a.end + gap * (a.score / tot if tot > 0.5 else 1))
            continue

        if i in spans_sep:                       # a word boundary
            if mode in ("vc-onset", "vc-sil") and env_t is not None:
                span = pause_span_in_gap(a.end * spf, b.start * spf,
                                         env_t, env_db, env_thr)
                if span is not None:
                    if mode == "vc-sil":
                        sil[i] = (span[0] / spf, span[1] / spf)
                    bounds.append(span[1] / spf)
                    continue
                lo, hi = (np.searchsorted(ftimes, a.end * spf),
                          np.searchsorted(ftimes, b.start * spf))
                if hi > lo:
                    bounds.append(ftimes[lo + int(np.argmax(flux[lo:hi]))] / spf)
                    continue
            tot = a.score + b.score
            bounds.append(a.end + gap * (a.score / tot if tot > 0.5 else 1))
            continue

        if class_split:                          # word-internal C<->V
            va, vb = a.label in VOCALIC, b.label in VOCALIC
            if va and not vb:
                bounds.append(b.start)
                continue
            if vb and not va and not (sustained_exception
                                      and a.label in SUSTAINED and i
                                      and kept[i - 1].label in VOCALIC):
                bounds.append(a.end)
                continue
        lo, hi = (np.searchsorted(ftimes, a.end * spf),
                  np.searchsorted(ftimes, b.start * spf))
        bounds.append(ftimes[lo + int(np.argmax(flux[lo:hi]))] / spf
                      if hi > lo else float(a.end + gap / 2))
    bounds.append(float(kept[-1].end))

    out = []
    for i, seg in enumerate(kept):
        st = sil[i - 1][1] if (i - 1) in sil else bounds[i]
        en = sil[i][0] if i in sil else bounds[i + 1]
        out.append(Segment(seg.label, st, max(st, en), seg.score))
    return out


def absorb_gaps_pe(segments, flux, ftimes, spf, pt, pp, env, env_thr,
                   env_t=None, env_db=None, drop_label=SEP,
                   cross_only=True, closure=True, closure_ms=25.0,
                   sustained_only=False, onset="0.5", offset=False,
                   aperiodic_db=6.0, base=None):
    """vc-sil, overruled by periodic energy at the one edge it can see clearly.

    Every rule before this one decides a gap from the CTC posteriors, from the
    phone classes, or from broadband spectral change. None can tell noise from
    voicing, so all of them place a fricative-vowel edge by the same logic they
    use for a nasal-vowel one. Periodic energy -- see kolsch_periodic, after
    ProPer -- separates those, and this mode applies it to:

      obstruent -> sonorant   VOICING ONSET, taken as the halfway crossing of
                              the periodic-energy rise across the gap. After a
                              stop that is the end of VOT, so closure, burst and
                              aspiration stay inside the consonant; after /h/ or
                              /s/ it is where the vowel starts rather than where
                              the noise gets loud.

    and, by default, to nothing else. Three restrictions were not designed but
    measured, on 941 word boundaries and 2717 word-internal ones across the 84
    helga recordings, scored against MFA and MAUS separately:

      * ONSETS ONLY (offset=False). The mirror rule -- voicing offset at
        sonorant -> obstruent -- looked equally principled and is not: it costs
        9 ms at nasal-stop boundaries against both references. Voicing onset is
        an abrupt event; voicing offset is gradual, so "where the curve falls"
        is not a location. offset=True restores it.
      * HALFWAY, NOT STEEPEST (onset="0.5"). The steepest rise, which is the
        threshold-free landmark ProPer uses for cycle boundaries, lands late
        into the vowel: 22.5 ms from MFA at stop-vowel edges against 16.6 for
        the halfway crossing. onset="rise" restores it.
      * SONORANT/OBSTRUENT ONLY (cross_only=True). Between two sonorants the
        periodic curve is flat, and a landmark read off a flat curve is noise:
        applying these rules everywhere cost 12 ms at nasal-vowel and V-V
        boundaries.

    Silence is handled by position:

      * between two words a pause is EMITTED as a hole -- vc-sil's behaviour,
        inherited unchanged;
      * inside a word it is not a pause but a closure, and it belongs to the
        consonant that FOLLOWS it, so the boundary goes at the silence's START.
        A word-internal silence never becomes a hole and never extends the
        phone in front of it.

    Net, against vc-sil: better or equal on all six of {MFA, MAUS} x
    {word-initial onset, word-internal onset, word-final end}, with no bucket
    made worse by either reference. The gain concentrates where it was designed
    to: voiceless fricative -> vowel goes 30.4 -> 21.2 ms against MFA and
    39.0 -> 32.1 against MAUS.
    """
    from kolsch_periodic import crossing, fall, quiet_span, rise

    def _onset(t0, t1):
        if onset == "rise":
            return rise(t0, t1, pt, pp)
        return crossing(t0, t1, pt, pp, frac=float(onset))

    # `base` lets a caller that already has its own vc-sil supply it, so this
    # periodic layer stays one implementation across two codebases without
    # either having to trust the other's vc-sil.
    if base is None:
        base = absorb_gaps(segments, flux, ftimes, spf, mode="vc-sil",
                           env_t=env_t, env_db=env_db, env_thr=env_thr,
                           drop_label=drop_label)
    if len(base) < 2:
        return base
    kept = [s for s in segments if s.label != drop_label]
    spans_sep, k = set(), -1
    for seg in segments:
        if seg.label == drop_label:
            spans_sep.add(k)
        else:
            k += 1

    out = [Segment(s.label, s.start, s.end, s.score) for s in base]
    for i in range(len(out) - 1):
        a, b = kept[i], kept[i + 1]
        if b.start <= a.end or out[i].end < out[i + 1].start - 1e-9:
            continue                      # no blank run, or vc-sil left a hole
        sa, sb = a.label in VOCALIC or a.label in NASAL or a.label in APPROXIMANT, \
            b.label in VOCALIC or b.label in NASAL or b.label in APPROXIMANT
        if cross_only and sa == sb:
            continue
        if sustained_only and (a.label if sb else b.label) not in SUSTAINED:
            continue
        t0, t1 = a.end * spf, b.start * spf

        t = None
        if closure and not sb and i not in spans_sep:
            span = quiet_span(t0, t1, pt, pp, env, env_thr, min_ms=closure_ms)
            if span is not None:
                t = span[0]
        if t is None and sb and not sa:
            # Is there anything to find? A VOICED obstruent -- intervocalic /h/,
            # /v/, /z/ -- keeps the curve well above the floor for the whole
            # gap, so there is no voicing onset inside it: voicing never
            # stopped. Cutting at a crossing of a curve that only wobbles
            # invents a landmark. Hand the blank to the sonorant instead, which
            # is where the CTC spike says the obstruent ended.
            lo, hi = (int(np.searchsorted(pt, t0)), int(np.searchsorted(pt, t1)))
            t = (a.end * spf if hi - lo >= 2 and pp[lo:hi].min() >= aperiodic_db
                 else _onset(t0, t1))
        if t is None:
            t = fall(t0, t1, pt, pp) if sa and not sb and offset else None
        if t is None:
            continue
        t = min(max(t / spf, a.end), b.start)
        out[i] = Segment(out[i].label, out[i].start, max(out[i].start, t),
                         out[i].score)
        out[i + 1] = Segment(out[i + 1].label, min(t, out[i + 1].end),
                             out[i + 1].end, out[i + 1].score)
    return out


# --------------------------------------------------------------------------- #
# TextGrid
# --------------------------------------------------------------------------- #
def write_textgrid(path, dur, tiers):
    """Praat short-text format. Holes become empty intervals, as Praat requires."""
    def pad(ivs):
        out, t = [], 0.0
        for iv in ivs:
            if iv["start"] > t + 1e-6:
                out.append({"label": "", "start": t, "end": iv["start"]})
            out.append(iv)
            t = iv["end"]
        if t < dur - 1e-6:
            out.append({"label": "", "start": t, "end": dur})
        return out

    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
         "0", f"{dur:.6f}", "<exists>", str(len(tiers))]
    for name, ivs in tiers:
        ivs = pad(ivs)
        L += ['"IntervalTier"', f'"{name}"', "0", f"{dur:.6f}", str(len(ivs))]
        for iv in ivs:
            L += [f'{iv["start"]:.6f}', f'{iv["end"]:.6f}',
                  '"' + str(iv["label"]).replace('"', '""') + '"']
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def read_textgrid_tier(path, tier="phones"):
    """Minimal reader for the short-text format, for scoring against a reference."""
    toks = re.findall(r'"[^"]*"|[-\d.]+', Path(path).read_text(encoding="utf-8"))
    out, i, cur = [], 0, None
    while i < len(toks):
        t = toks[i]
        if t.startswith('"') and t.strip('"') in ("IntervalTier", "TextTier"):
            cur = toks[i + 1].strip('"')
            i += 2
            continue
        if cur == tier and re.fullmatch(r"[-\d.]+", t) and i + 2 < len(toks) \
                and toks[i + 2].startswith('"'):
            try:
                a, b = float(t), float(toks[i + 1])
            except ValueError:
                i += 1
                continue
            lab = toks[i + 2].strip('"')
            if b > a and lab not in ("", "sil", "sp", "<p:>"):
                out.append({"label": lab, "start": a, "end": b})
            i += 3
            continue
        i += 1
    return out


# --------------------------------------------------------------------------- #
# the aligner
# --------------------------------------------------------------------------- #
class Aligner:
    def __init__(self, model_dir, processor_dir=None, device=None, lexicon=None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(
            processor_dir or model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(self.device).eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank_id = self.processor.tokenizer.pad_token_id
        self.syms = sorted((s for s in self.vocab if s not in _SPECIAL),
                           key=len, reverse=True)
        self.lex = {}
        if lexicon and os.path.exists(lexicon):
            for row in csv.DictReader(open(lexicon, encoding="utf-8")):
                self.lex[row["kolsch"]] = row["ipa"]
        missing = {s for s in self.syms if s not in KNOWN}
        if missing:
            raise ValueError(
                f"symbols with no phone class: {sorted(missing)} — classify them "
                "in kolsch_align.py or absorb='vc' silently treats them as "
                "consonants")

    # -- text -> phones ----------------------------------------------------- #
    @staticmethod
    def _clean(w):
        for a in "'’ʼʻ`":
            w = str(w).lower().replace(a, "")
        return re.sub(r"[^a-zäöüßï]", "", str(w).lower())

    def ipa_to_phones(self, ipa):
        out, i = [], 0
        while i < len(ipa):
            for s in self.syms:
                if ipa.startswith(s, i):
                    out.append(s)
                    i += len(s)
                    break
            else:
                i += 1
        return out

    def text_to_chain(self, text):
        """-> ([[phones of word 1], ...], [orthographic words])"""
        from kolsch_g2p import word_to_ipa
        words, chain = [], []
        for tok in str(text).split():
            w = self._clean(tok)
            if not w:
                continue
            ph = self.ipa_to_phones(self.lex.get(w) or word_to_ipa(w))
            if ph:
                words.append(tok)
                chain.append(ph)
        return chain, words

    # -- alignment ---------------------------------------------------------- #
    def emissions(self, wav):
        with torch.inference_mode():
            logits = self.model(torch.tensor(wav, device=self.device)[None]).logits
        return torch.log_softmax(logits, dim=-1).cpu()

    def align(self, wav_path, text, mode="vc", chain=None):
        """-> (phone intervals, word intervals, duration) in SECONDS.

        forced_align is a Viterbi pass over the CTC lattice given the chain you
        supply. It cannot skip, insert or reorder a phone: a wrong chain gives a
        confidently wrong alignment, which is why the text comes from a
        transcript and not from the model's own decode.
        """
        wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        dur = len(wav) / SAMPLE_RATE
        if chain is None:
            chain, words = self.text_to_chain(text)
        else:
            words = list(text) if not isinstance(text, str) else text.split()
        if not chain:
            raise ValueError(f"no phones for {text!r}")

        tokens = []
        for k, w in enumerate(chain):
            if k:
                tokens.append(SEP)
            tokens.extend(w)
        ids = torch.tensor([[self.vocab[t] for t in tokens]], dtype=torch.int32)

        logp = self.emissions(wav)
        spf = dur / logp.shape[1]
        path, sc = torchaudio.functional.forced_align(logp, ids,
                                                      blank=self.blank_id)
        segs = merge_repeats(path[0], sc[0].exp(), self.blank_id,
                             self.processor.tokenizer)

        flux, ftimes = spectral_flux(wav, SAMPLE_RATE)
        env_t = env_db = env_thr = None
        if mode in ("vc-onset", "vc-sil", "pe"):
            env_t, env_db = energy_envelope(wav, SAMPLE_RATE)
            if len(env_db):
                floor, peak = np.percentile(env_db, 15), env_db.max()
                env_thr = floor + 0.15 * (peak - floor)
            else:
                env_t = None
        if mode == "pe":
            from kolsch_periodic import on_grid, periodic_power
            pt, pp, _, _ = periodic_power(wav, SAMPLE_RATE)
            if len(pt) == 0 or env_t is None:
                raise ValueError(f"{wav_path}: too short for a periodic curve")
            segs = absorb_gaps_pe(segs, flux, ftimes, spf, pt, pp,
                                  on_grid(env_t, env_db, pt), env_thr,
                                  env_t=env_t, env_db=env_db)
        else:
            segs = absorb_gaps(segs, flux, ftimes, spf, mode=mode,
                               env_t=env_t, env_db=env_db, env_thr=env_thr)

        phones = [{"label": s.label, "start": s.start * spf,
                   "end": s.end * spf, "score": s.score} for s in segs]
        word_iv, i = [], 0
        for orth, w in zip(words, chain):
            grp = phones[i:i + len(w)]
            i += len(w)
            if grp:
                word_iv.append({"label": orth, "start": grp[0]["start"],
                                "end": grp[-1]["end"]})
        return phones, word_iv, dur


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def coverage(phones, dur):
    """Fraction of the timeline the phones actually cover."""
    return sum(p["end"] - p["start"] for p in phones) / dur if dur else 0.0


def silence_ms(phones):
    """Total unlabelled time between consecutive phones."""
    return sum(max(0.0, b["start"] - a["end"])
               for a, b in zip(phones, phones[1:])) * 1000


def word_positions(chain):
    """-> (word-initial phone indices, word-final phone indices)."""
    ini, fin, i = set(), set(), 0
    for w in chain:
        ini.add(i)
        i += len(w)
        fin.add(i - 1)
    return ini, fin


def boundary_errors(phones, ref, chain=None):
    """Median |delta| in ms against a reference, split by position.

    The reference must be the SAME phone chain, index for index -- otherwise this
    compares two pronunciation dictionaries, not two aligners.
    """
    n = min(len(phones), len(ref))
    if n < 2:
        return {}
    ini, fin = word_positions(chain) if chain else (set(), set())
    buckets = {}
    for k in range(1, n):
        key = "word-initial onset" if k in ini else "word-internal onset"
        buckets.setdefault(key, []).append(
            abs(phones[k]["start"] - ref[k]["start"]) * 1000)
    for k in range(n - 1):
        if k in fin:
            buckets.setdefault("word-final end", []).append(
                abs(phones[k]["end"] - ref[k]["end"]) * 1000)
    return {k: float(np.median(v)) for k, v in buckets.items() if v}
