"""Periodic energy, and the acoustic landmarks a CTC aligner cannot see.

    from kolsch_periodic import periodic_power, rise, fall, trough
    t, pp, per, f0 = periodic_power(wav, 16000)

WHY. A CTC aligner emits one confident frame per phone. Everything between two
spikes is a rule, and until now every rule in this project answered "who owns
this blank run?" out of either the posteriors or a broadband spectral-change
curve. Neither knows the difference between a fricative and a vowel, so three
things stayed broken:

  * VOT is invisible. /t/ + /a/ is closure, burst, aspiration, then voicing.
    Spectral flux peaks at the BURST, so the vowel was made to start there and
    the aspiration was scored as vowel.
  * Fricative-vowel edges are soft. /h/ and /s/ are loud, so an energy
    threshold puts the vowel too early; /h/ especially, being breathy voiced,
    looks like a weak vowel to anything that only measures amplitude.
  * A pause inside a word is not a pause. Closure silence belongs to the stop
    that follows it, not to the phone in front.

WHAT PERIODIC ENERGY IS. Following ProPer -- PROsodic analysis with PERiodic
energy, Albert, Cangemi, Ellison & Grice, IfL Phonetik / SFB 1252, Cologne,
https://osf.io/28ea5/ -- we measure not how loud the signal is but how much of
its energy is PERIODIC. Turbulent noise, however loud, contributes nothing.
Two consequences are exactly what the three problems above need:

  * a fricative is near zero on this scale even at full volume, so the
    fricative-vowel edge becomes a rise rather than a plateau;
  * the rise marks VOICING ONSET, which is the end of VOT by definition.

Two details are taken from the paper rather than invented here. The floor is set
by the signal itself: "we measure purely voiceless portions and set the maximal
periodic energy value of those voiceless portions as the constant floor
denominator of the log variable" (Albert et al. 2018:805) -- so the zero of the
scale is the loudest thing in this recording that is not periodic. And boundaries
are read off the curve's shape, not a threshold: "minima (and maxima) in the
periodic energy function are indices of boundaries (and peaks) of periodic
cycles" (ibid.:807).

WHERE WE DIVERGE. ProPer runs on Praat's periodicity via the APP detector and
smooths with LOESS in R. This module has to run inside a notebook with no Praat
and no R, so periodicity comes from the average magnitude difference function --
AMDF, the cheapest of the classical period detectors:

    AMDF(tau) = mean_n |x[n] - x[n+tau]|

which falls to near zero when tau is the period and stays near its average for
noise. Normalising by the median over the lag range gives a periodicity in
roughly [0, 1] with no calibration. It is less accurate than Praat on F0 -- we
are not using it for F0 -- and entirely adequate for "is this frame periodic",
which is all the alignment rules ask.

The AMDF also answers the second question cheaply: silence is where both
periodic energy and broadband energy are at the floor. That is what separates a
pause from a voiceless fricative, which an energy threshold alone cannot do.

CHECK cue_strength() BEFORE BELIEVING ANY OF IT. This measure needs a quiet
recording. Broadband noise fills in the AMDF minimum that periodicity is read
from, so on a noisy transfer the vowels stop looking periodic and the whole
curve flattens. It works on this project's 16 kHz field recordings, whose noise
floor sits 46 dB below peak, and it does NOT work on its archival CD cuts, which
are 20-35 dB noisier: there, vowels and voiceless fricatives are 0.2 dB apart on
this scale (d' = 0.08) and every landmark below would be read off noise. The
--absorb pe rule inherits that limitation exactly.
"""
import numpy as np

PSR = 8000          # periodicity is computed here: F0 <= 500 Hz needs no more
F0_MIN, F0_MAX = 60.0, 500.0
HOP_MS, WIN_MS = 5.0, 40.0


# --------------------------------------------------------------------------- #
# periodicity
# --------------------------------------------------------------------------- #
def _bandpass(y, sr, lo=60.0, hi=1100.0):
    """Keep F0 and its first harmonics; drop rumble and fricative noise.

    Fricative energy lives mostly above 4 kHz, so removing it here is what makes
    a loud /s/ read as silence on the periodic scale.
    """
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [lo / (sr / 2), min(hi, sr / 2 - 1) / (sr / 2)],
                 btype="band", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _frames(y, win, hop):
    """(n, win) view, one row per analysis frame."""
    n = 1 + max(0, (len(y) - win) // hop)
    if n < 1:
        return np.zeros((0, win), dtype=np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def periodicity(wav, sr, hop_ms=HOP_MS, win_ms=WIN_MS,
                f0_min=F0_MIN, f0_max=F0_MAX):
    """-> (times, periodicity in [0,1], f0 in Hz, band RMS).

    periodicity is 1 - min(AMDF) / median(AMDF) over the lag range implied by
    f0_min..f0_max. The median rather than the mean because the minima we are
    looking for would otherwise drag the normaliser down with them.
    """
    import librosa
    y = librosa.resample(np.asarray(wav, dtype=np.float32),
                         orig_sr=sr, target_sr=PSR) if sr != PSR else \
        np.asarray(wav, dtype=np.float32)
    y = _bandpass(y, PSR)

    hop = max(1, int(round(hop_ms / 1000 * PSR)))
    win = max(8, int(round(win_ms / 1000 * PSR)))
    lo_lag, hi_lag = int(PSR / f0_max), int(PSR / f0_min)
    need = win + hi_lag
    if len(y) < need + hop:
        return (np.zeros(0),) * 4

    F = _frames(y, need, hop)                     # (n, win + hi_lag)
    seg = F[:, :win]
    amdf = np.empty((F.shape[0], hi_lag - lo_lag + 1), dtype=np.float32)
    for k, tau in enumerate(range(lo_lag, hi_lag + 1)):
        amdf[:, k] = np.abs(seg - F[:, tau:tau + win]).mean(axis=1)

    med = np.median(amdf, axis=1)
    arg = amdf.argmin(axis=1)
    per = np.clip(1.0 - amdf.min(axis=1) / (med + 1e-12), 0.0, 1.0)
    f0 = PSR / (arg + lo_lag).astype(np.float64)
    rms = np.sqrt((seg ** 2).mean(axis=1))
    # a frame with no energy has no period either; AMDF is meaningless there
    per[rms < 1e-6] = 0.0
    return np.arange(F.shape[0]) * hop / PSR, per, f0, rms


def _smooth(x, k):
    if k < 2 or len(x) < k:
        return x
    k |= 1
    return np.convolve(np.pad(x, k // 2, mode="edge"),
                       np.ones(k) / k, mode="valid")


def periodic_power(wav, sr, hop_ms=HOP_MS, smooth=5, per_floor=0.45):
    """-> (times, periodic power in dB above the voiceless floor, per, f0).

    power = band RMS * periodicity, i.e. the share of the energy that is
    periodic, then log-scaled against the loudest APERIODIC frame in this
    recording so that fricatives and noise land at 0 dB by construction. Frames
    below that floor are clipped to 0 rather than going negative: how far below
    the noise something is carries no information here.
    """
    t, per, f0, rms = periodicity(wav, sr, hop_ms=hop_ms)
    if len(t) == 0:
        return (np.zeros(0),) * 4
    pp = rms * per
    aper = per < per_floor
    if aper.sum() >= 5:
        floor = np.percentile(pp[aper], 95)
    else:                       # an utterance with no voiceless stretch at all
        floor = np.percentile(pp, 5)
    floor = max(float(floor), 1e-7)
    db = np.clip(20 * np.log10(np.maximum(pp, 1e-12) / floor), 0.0, None)
    return t, _smooth(db, smooth), per, f0


def cue_strength(wav, sr):
    """-> dict: is periodic energy usable on this recording at all?

    THE METHOD HAS A PRECONDITION AND IT IS NOT MILD. AMDF finds a period by
    looking for a deep minimum, and broadband noise fills that minimum in. On a
    quiet recording a vowel reaches a periodicity around 0.85 and a voiceless
    fricative stays near 0.3, which is the contrast every rule here depends on.
    Add 20 dB of background hiss and the vowels fall to about 0.6 while the
    fricatives do not move: the contrast is buried, the floor is computed from
    frames that are not actually voiceless, and the curve becomes flat noise.

    Measured on this project's two corpora, through identical code:

      84 field recordings   noise floor -46 dB, periodicity 0.85 when loud,
                            vowel vs voiceless fricative d' = 1.1   -> usable
      55 archival CD cuts   noise floor -11 to -27 dB, periodicity 0.6 when
                            loud, d' = 0.08                          -> NOT

    `usable` is False below 0.75 because that is where the two corpora fall
    either side, not because 0.75 is principled. Treat it as a warning to go and
    look, not a certificate.
    """
    t, per, _, rms = periodicity(wav, sr)
    if len(t) < 8:
        return {"usable": False, "reason": "too short"}
    loud, quiet = rms >= np.percentile(rms, 70), rms <= np.percentile(rms, 15)
    pl = float(per[loud].mean())
    return {"noise_floor_db": float(20 * np.log10(
                np.percentile(rms, 15) / (rms.max() + 1e-12) + 1e-12)),
            "per_loud": pl, "per_quiet": float(per[quiet].mean()),
            "usable": pl >= 0.75,
            "reason": "" if pl >= 0.75 else
                      f"periodicity only {pl:.2f} in the loudest frames; "
                      "the recording is too noisy for this cue"}


# --------------------------------------------------------------------------- #
# landmarks
# --------------------------------------------------------------------------- #
def _window(t0, t1, times):
    lo, hi = int(np.searchsorted(times, t0)), int(np.searchsorted(times, t1))
    return (lo, hi) if hi - lo >= 2 else (None, None)


def rise(t0, t1, times, curve):
    """Steepest RISE of `curve` in [t0, t1], or None.

    Threshold-free on purpose: an absolute cut-off would have to be calibrated
    per recording, and the thing we want -- where periodic energy starts -- is a
    property of the curve's shape, not of its level.
    """
    lo, hi = _window(t0, t1, times)
    if lo is None:
        return None
    d = np.diff(curve[lo:hi])
    if not len(d) or d.max() <= 0:
        return None
    k = int(d.argmax())
    return float((times[lo + k] + times[lo + k + 1]) / 2)


def fall(t0, t1, times, curve):
    """Steepest FALL of `curve` in [t0, t1], or None."""
    lo, hi = _window(t0, t1, times)
    if lo is None:
        return None
    d = np.diff(curve[lo:hi])
    if not len(d) or d.min() >= 0:
        return None
    k = int(d.argmin())
    return float((times[lo + k] + times[lo + k + 1]) / 2)


def trough(t0, t1, times, curve):
    """Deepest MINIMUM of `curve` in [t0, t1] -- ProPer's cycle boundary."""
    lo, hi = _window(t0, t1, times)
    if lo is None:
        return None
    return float(times[lo + int(curve[lo:hi].argmin())])


def crossing(t0, t1, times, curve, frac=0.5):
    """First upward crossing of min + frac*(max-min) in [t0, t1].

    Kept as a sensitivity check on `rise`, which is the rule actually used.
    """
    lo, hi = _window(t0, t1, times)
    if lo is None:
        return None
    seg = curve[lo:hi]
    lvl = seg.min() + frac * (seg.max() - seg.min())
    above = np.where(seg >= lvl)[0]
    return float(times[lo + int(above[0])]) if len(above) else None


def on_grid(env_t, env_db, times):
    """Put a broadband envelope on the periodic curve's time base."""
    if len(env_t) == 0:
        return np.full(len(times), np.inf)
    return np.interp(times, env_t, env_db)


def quiet_span(t0, t1, times, pp, env, env_thr, pp_thr=1.0, min_ms=40.0):
    """Longest run in [t0, t1] that is quiet on BOTH scales, or None.

    Both, because either alone misfires: a voiceless fricative is silent on the
    periodic scale but not on the broadband one, and a murmured nasal is the
    other way round. A pause is the intersection. `env` must already be on
    `times` -- see on_grid().
    """
    lo, hi = _window(t0, t1, times)
    if lo is None:
        return None
    q = (pp[lo:hi] <= pp_thr) & (env[lo:hi] < env_thr)
    best, run = None, None
    for k, v in enumerate(list(q) + [False]):
        if v:
            run = k if run is None else run
        elif run is not None:
            if best is None or k - run > best[1] - best[0]:
                best = (run, k)
            run = None
    if best is None:
        return None
    a, b = float(times[lo + best[0]]), float(times[lo + best[1] - 1])
    return (a, b) if (b - a) * 1000 >= min_ms else None
