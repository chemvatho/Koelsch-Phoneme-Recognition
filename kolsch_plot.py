"""Draw an alignment the way the README does: waveform, word tier, phone tier.

    from kolsch_plot import plot_alignment
    plot_alignment("clip.wav", phones, words, dur, out="clip.png")

Separate from kolsch_align so that importing the aligner never drags matplotlib
in -- alignment runs fine on a machine with no display and no plotting stack,
and the notebooks that do want a picture ask for one.
"""
from pathlib import Path

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"
C_WORD, C_PHONE = "#0d9488", "#2a78d6"


def plot_alignment(wav_path, phones, words, dur, out=None, title=None,
                   subtitle=None, ax=None, max_label_ms=12.0):
    """-> the Axes. Pass `ax` to place this inside a bigger figure.

    `phones` and `words` are the interval lists Aligner.align() returns:
    dicts with label/start/end in SECONDS. Intervals shorter than
    max_label_ms are drawn but not lettered, because at a typical figure
    width their text would overprint the neighbours rather than inform.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    made_fig = ax is None
    if made_fig:
        fig, ax = plt.subplots(figsize=(13.2, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    w, sr = sf.read(str(wav_path))
    if getattr(w, "ndim", 1) > 1:
        w = w.mean(axis=1)
    t = np.arange(len(w)) / sr
    ax.plot(t, w / (np.abs(w).max() or 1) * 0.36 + 0.60, lw=0.45,
            color="#c3c2be", zorder=0)

    for name, items, y, col in (("words", words, -0.30, C_WORD),
                                ("phones", phones, -0.92, C_PHONE)):
        for p in items:
            ax.add_patch(plt.Rectangle((p["start"], y), p["end"] - p["start"],
                                       0.52, facecolor=col, alpha=0.20,
                                       edgecolor=col, linewidth=1.0))
            if (p["end"] - p["start"]) * 1000 >= max_label_ms:
                ax.annotate(str(p["label"]),
                            ((p["start"] + p["end"]) / 2, y + 0.26),
                            ha="center", va="center", fontsize=9.5, color=INK)
        ax.annotate(name, (-0.006, y + 0.26), ha="right", va="center",
                    fontsize=9.5, color=col, fontweight="bold",
                    xycoords=("axes fraction", "data"), annotation_clip=False)

    ax.set_xlim(0, dur)
    ax.set_ylim(-1.10, 1.12)
    ax.set_yticks([])
    ax.set_xlabel("time (s)", fontsize=9, color=INK2)
    ax.grid(axis="x", alpha=0.55, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if title:
        ax.set_title(title if not subtitle else f"{title}\n{subtitle}",
                     loc="left", fontsize=11, fontweight="bold", pad=8,
                     color=INK)

    if made_fig and out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
    return ax
