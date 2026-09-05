#!/usr/bin/env python3
"""Try both Kölsch models on your own audio. Two tasks, one script.

    # 1 - orthography: what was said, in Kölsch spelling  (w2v-BERT 2.0)
    python tools/try_models.py transcribe data/segments/*.wav

    # 2 - IPA + forced alignment: a TextGrid and a picture  (wav2vec2 XLS-R-300M)
    python tools/try_models.py align data/segments/cd1_track01_000.wav

`align` transcribes the clip itself and then aligns the audio to that
transcription, so it needs nothing but a wav. Pass --text if you already have a
transcript -- a forced aligner can only return what you gave it, so a supplied
transcript is always the better input when you have one.

MODELS. Set KOLSCH_ORTHO_MODEL and KOLSCH_MODEL to local directories or to
Hugging Face repo ids; the defaults below are the published ids. Audio is
resampled to 16 kHz automatically, so any sample rate is fine.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IPA_MODEL = os.environ.get("KOLSCH_MODEL", "chemvatho/koelsch-wav2vec2-ipa")
IPA_PROC = os.environ.get("KOLSCH_PROCESSOR", IPA_MODEL)
ORTHO_MODEL = os.environ.get("KOLSCH_ORTHO_MODEL",
                             "chemvatho/koelsch-w2vbert-orthography")
SR = 16_000


def load_audio(path):
    import librosa
    wav, _ = librosa.load(str(path), sr=SR)
    return wav, len(wav) / SR


def cmd_transcribe(args):
    """Task 1 - orthography, w2v-BERT 2.0."""
    import torch
    from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

    print(f"loading {ORTHO_MODEL}", file=sys.stderr)
    proc = Wav2Vec2BertProcessor.from_pretrained(ORTHO_MODEL)
    model = Wav2Vec2BertForCTC.from_pretrained(ORTHO_MODEL).eval()
    if torch.cuda.is_available():
        model.cuda()

    for p in args.audio:
        wav, dur = load_audio(p)
        inp = proc(wav, sampling_rate=SR, return_tensors="pt")
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.inference_mode():
            ids = model(**inp).logits.argmax(-1)
        print(f"{Path(p).name:28} {dur:5.2f}s  {proc.batch_decode(ids)[0]}")
    return 0


def decode_chain(al, wav_path):
    """Greedy CTC decode -> [[phones of word 1], ...].

    Not tokenizer.decode(): that treats '|' as the word delimiter and drops it,
    so the whole utterance comes back as one word and the alignment gets a
    single word interval. Collapsing the argmax by hand keeps the separators the
    model actually predicted -- it does predict them, ~8 per clip here.
    """
    import torch
    from kolsch_align import SEP

    wav, _ = load_audio(wav_path)
    with torch.inference_mode():
        ids = al.model(torch.tensor(wav, device=al.device)[None]
                       ).logits.argmax(-1)[0]
    tok = al.processor.tokenizer
    pad = tok.pad_token
    seq, prev = [], None
    for t in tok.convert_ids_to_tokens([int(i) for i in ids]):
        if t == pad:
            prev = None
            continue
        if t != prev:
            seq.append(t)
        prev = t
    chain, cur = [], []
    for t in seq:
        if t == SEP:
            if cur:
                chain.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        chain.append(cur)
    return chain


def cmd_align(args):
    """Task 2 - IPA + forced alignment, wav2vec2 XLS-R-300M."""
    from kolsch_align import Aligner, write_textgrid

    lex = ROOT / "data" / "lexicon.csv"
    al = Aligner(IPA_MODEL, IPA_PROC, lexicon=str(lex) if lex.exists() else None)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in args.audio:
        stem = Path(p).stem
        text = args.text
        if text is None:
            # No transcript supplied: decode the audio, then align to that.
            # Recognition errors become alignment errors here -- the aligner
            # cannot place a phone the decode never proposed.
            chain = decode_chain(al, p)
            if not chain:
                print(f"{stem}: nothing decoded, skipped")
                continue
            words = ["".join(w) for w in chain]
            print(f"{stem}: decoded  {' | '.join(words)}")
            phones, word_iv, dur = al.align(p, words, mode=args.absorb,
                                            chain=chain)
        else:
            phones, word_iv, dur = al.align(p, text, mode=args.absorb)

        tg = out_dir / f"{stem}.TextGrid"
        write_textgrid(tg, dur, [("words", word_iv), ("phones", phones)])
        print(f"{stem}: {len(phones)} phones, {len(word_iv)} words "
              f"-> {tg.relative_to(ROOT) if tg.is_relative_to(ROOT) else tg}")

        if not args.no_plot:
            from kolsch_plot import plot_alignment
            png = out_dir / f"{stem}.png"
            plot_alignment(p, phones, word_iv, dur, out=png,
                           title=f"{stem}  —  wav2vec2 XLS-R-300M, "
                                 f"--absorb {args.absorb}",
                           subtitle=" ".join(w["label"] for w in word_iv))
            print(f"{' ' * len(stem)}  -> {png.relative_to(ROOT) if png.is_relative_to(ROOT) else png}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="task 1: Kölsch spelling (w2v-BERT)")
    t.add_argument("audio", nargs="+")
    t.set_defaults(func=cmd_transcribe)

    a = sub.add_parser("align", help="task 2: IPA + TextGrid + plot (wav2vec2)")
    a.add_argument("audio", nargs="+")
    a.add_argument("--text", default=None,
                   help="transcript to align to; without it the clip is decoded "
                        "first")
    a.add_argument("--absorb", default="vc",
                   help="gap rule; see 09_alignment/ (default: vc)")
    a.add_argument("--out", default="out_alignment")
    a.add_argument("--no-plot", action="store_true")
    a.set_defaults(func=cmd_align)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
