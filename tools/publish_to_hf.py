#!/usr/bin/env python3
"""Publish the two Kölsch checkpoints to the Hugging Face Hub.

    huggingface-cli login                # or: export HF_TOKEN=hf_...
    python tools/publish_to_hf.py --owner chemvatho --dry-run
    python tools/publish_to_hf.py --owner chemvatho --push

WHY NOT GITHUB. GitHub rejects any file over 100 MB outright. These two
checkpoints are 1.26 GB and 2.42 GB, so neither can be pushed, and Git LFS does
not rescue it either: the free tier is 1 GB of storage and 1 GB of bandwidth a
month, which one clone of one model would exhaust. The Hub is free for public
models, has no such limit, and -- the practical part -- makes
from_pretrained("owner/name") work with no download step for the user.

NOTHING LEAVES THIS MACHINE WITHOUT --push. The default prints exactly what
would be uploaded and stops.

RIGHTS. The weights are derived from the Alles Koelsch corpus, which belongs to
the Akademie foer uns koelsche Sproch. Weights are not the corpus and this
script uploads no audio and no transcripts, but a heritage-corpus partner may
still have a view on derived models being downloadable worldwide. That is a
conversation to have before --push, not after.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CARD = """---
language: [ksh, de]
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags: [koelsch, kolsch, ripuarian, dialect, low-resource, ctc, {tags}]
---

# {title}

{summary}

Trained during the **CIF Tandem Fellowship** at IfL-Phonetik, University of
Cologne, on *Alles Kölsch* (Bhatt & Lindlar 1998) — 4,670 utterances, 4.5 hours
of **spontaneous** Cologne dialect speech from 105 speakers aged 10–88 across 49
neighbourhoods.

Kölsch is Ripuarian German. It has no public speech dataset and no standardised
spelling: **94.1 % of Kölsch word tokens are out of vocabulary** against the
152,766-word `german_mfa` dictionary. It is not German text.

## Use it

```python
{usage}
```

Full pipeline, notebooks and the alignment tooling:
<https://github.com/chemvatho/Koelsch-Phoneme-Recognition>

```bash
python tools/try_models.py {cli}
```

## How it scores

{results}

> **These numbers are optimistic and it is worth knowing why.** The test split
> is **not speaker-disjoint** — {leakage} So they measure how well the model
> transcribes voices it has already heard, which is a different and easier
> question than generalisation. Treat them as an upper bound. A speaker-held-out
> re-split is on the project roadmap.

## Limitations

- **Spontaneous dialect speech, archival source.** The training audio is a CD
  transfer whose median noise floor sits ~28 dB below peak. Expect degradation
  on noisier input, and note that studio recordings are *out* of domain too.
- **No language model.** Greedy CTC decoding only.
- **Kölsch, not German.** Scored against a Standard German reference this model
  looks bad — it is penalised for correctly writing the dialect it heard.
{extra_limits}

## Licence and rights

The **code** in the repository is MIT. **These weights are released for research
use** under CC-BY-NC-SA-4.0. The underlying *Alles Kölsch* corpus is **not**
redistributed here and is not covered by either: it belongs to the **Akademie
för uns kölsche Sproch**, who should be contacted for corpus access.

## Citation

```bibtex
@misc{{chem2026koelsch,
  author = {{Chem, Vatho and R\\"ossig, Simon and Greisbach, Reinhold}},
  title  = {{K\\"olsch Phoneme Recognition: an open pipeline from a printed
            dialect corpus to phoneme recognition and forced alignment}},
  year   = {{2026}},
  note   = {{CIF Tandem Fellowship, IfL-Phonetik, University of Cologne}},
  howpublished = {{\\url{{https://github.com/chemvatho/Koelsch-Phoneme-Recognition}}}}
}}
```
"""

IPA_USAGE = """import torch, librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

proc  = Wav2Vec2Processor.from_pretrained("{repo}")
model = Wav2Vec2ForCTC.from_pretrained("{repo}").eval()

wav, _ = librosa.load("clip.wav", sr=16000)
with torch.inference_mode():
    ids = model(proc(wav, sampling_rate=16000,
                     return_tensors="pt").input_values).logits.argmax(-1)
print(proc.batch_decode(ids)[0])
# -> ɪ ç v ə l ə s ɔ χ n ɪ t m iː v ɪ d ɐ d ʊ n"""

ORTHO_USAGE = """import torch, librosa
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

proc  = Wav2Vec2BertProcessor.from_pretrained("{repo}")
model = Wav2Vec2BertForCTC.from_pretrained("{repo}").eval()

wav, _ = librosa.load("clip.wav", sr=16000)
inp = proc(wav, sampling_rate=16000, return_tensors="pt")
with torch.inference_mode():
    ids = model(**inp).logits.argmax(-1)
print(proc.batch_decode(ids)[0])
# -> ich well es och nit mih widder dun"""

MODELS = {
    "ipa": dict(
        name="koelsch-wav2vec2-ipa",
        src=ROOT.parents[2] / "best_model" / "kolsch_wav2vec2_model_all",
        proc=ROOT.parents[2] / "best_model" / "kolsch_final_model_all",
        title="Kölsch phoneme recogniser — wav2vec2 XLS-R-300M (IPA)",
        tags="phoneme-recognition, forced-alignment, ipa",
        summary=(
            "IPA phoneme recognition for **Kölsch** (Ripuarian German, "
            "Cologne). Fine-tuned from `facebook/wav2vec2-xls-r-300m` with a "
            "CTC head over a 48-symbol IPA inventory.\n\n"
            "**This is also the forced-alignment model.** Its frame-level CTC "
            "posteriors drive `torchaudio.functional.forced_align` to produce "
            "Praat TextGrids with word and phone tiers — see notebook 9 in the "
            "repository."),
        usage=IPA_USAGE,
        cli="align clip.wav        # TextGrid + plot",
        results=(
            "| | |\n|---|---|\n"
            "| **PER** | **15.3 %** |\n"
            "| CER over the IPA stream | 15.3 % |\n\n"
            "467 held-out utterances, 21,463 reference phones, recomputed from "
            "the stored test predictions. For context, an off-the-shelf "
            "multilingual Wav2Vec2Phoneme scores ~33 % PER on comparable "
            "German-dialect material."),
        leakage="103 of its 105 speakers also appear in training.",
        extra_limits=(
            "- **For alignment, roughly four fifths of every phone duration is a "
            "rule, not a measurement.** CTC is peaky: labelled frames cover only "
            "14–21 % of the timeline, and which rule fills the gaps matters more "
            "than the model does. Read `09_alignment/` before taking any "
            "duration from a TextGrid this produces."),
    ),
    "ortho": dict(
        name="koelsch-w2vbert-orthography",
        src=ROOT.parents[2] / "Compare_helga_glidehaus" / "Kolsch_Dataset"
            / "kolsch_w2vbert_ortho_model",
        proc=None,
        title="Kölsch orthographic recogniser — w2v-BERT 2.0",
        tags="orthography, transcription",
        summary=(
            "Transcribes **Kölsch** speech into Kölsch spelling. Fine-tuned "
            "from `facebook/w2v-bert-2.0` with a CTC head over the book's "
            "German-letter orthography.\n\n"
            "Note that `w2v-bert-2.0` ships no CTC head, so `lm_head` *and* the "
            "conv adapter were initialised randomly — the stronger multilingual "
            "pretraining had to pay for that."),
        usage=ORTHO_USAGE,
        cli="transcribe clip.wav",
        results=(
            "| | |\n|---|---|\n"
            "| **CER** | **11.3 %** |\n"
            "| WER | 34.0 % |\n\n"
            "582 held-out utterances, recomputed from the stored test "
            "predictions.\n\n"
            "**CER is the headline here, not WER.** Kölsch has no spelling "
            "standard, so *janz*/*ganz* and *zusamme*/*zosamme* are one word "
            "spelled two ways and WER charges full price for the difference."),
        leakage="22 % of its test clips (128/582) come from a training speaker.",
        extra_limits="",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True, help="your Hugging Face username")
    ap.add_argument("--only", choices=list(MODELS), help="publish just one")
    ap.add_argument("--private", action="store_true",
                    help="create the repos private; make them public later")
    ap.add_argument("--push", action="store_true",
                    help="actually upload. Without it, nothing leaves this "
                         "machine.")
    args = ap.parse_args()

    picks = [args.only] if args.only else list(MODELS)
    cards = {}
    ok = True
    for key in picks:
        m = MODELS[key]
        repo = f"{args.owner}/{m['name']}"
        src = Path(m["src"])
        files = sorted(p for p in src.glob("*") if p.is_file()) if src.is_dir() else []
        size = sum(p.stat().st_size for p in files) / 1e9
        print(f"\n=== {repo} ===")
        if not files:
            print(f"  !! {src} not found or empty")
            ok = False
            continue
        print(f"  from {src}")
        print(f"  {len(files)} files, {size:.2f} GB")
        for p in files:
            print(f"     {p.stat().st_size / 1e6:9.1f} MB  {p.name}")
        if m["proc"]:
            pf = sorted(Path(m["proc"]).glob("*"))
            print(f"  + processor from {m['proc']}: "
                  f"{', '.join(p.name for p in pf)}")
        cards[key] = CARD.format(
            title=m["title"], tags=m["tags"], summary=m["summary"],
            usage=m["usage"].format(repo=repo), cli=m["cli"],
            results=m["results"], leakage=m["leakage"],
            extra_limits=m["extra_limits"])

    card_dir = ROOT / "docs" / "model_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    for key, text in cards.items():
        p = card_dir / f"{MODELS[key]['name']}.md"
        p.write_text(text, encoding="utf-8")
        print(f"\ncard -> {p.relative_to(ROOT)}")

    if not ok:
        return 1
    if not args.push:
        print("\nDry run — nothing uploaded. Re-run with --push to publish.")
        return 0

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        print("\nNo HF_TOKEN in the environment. Run `huggingface-cli login` "
              "first,\nor export HF_TOKEN=hf_...", file=sys.stderr)
        return 1
    from huggingface_hub import HfApi
    api = HfApi()
    for key in picks:
        m = MODELS[key]
        repo = f"{args.owner}/{m['name']}"
        print(f"\nuploading {repo} …")
        api.create_repo(repo, repo_type="model", exist_ok=True,
                        private=args.private)
        api.upload_folder(folder_path=str(m["src"]), repo_id=repo,
                          repo_type="model",
                          ignore_patterns=["*.bin.index.json", "checkpoint-*"])
        if m["proc"]:
            api.upload_folder(folder_path=str(m["proc"]), repo_id=repo,
                              repo_type="model")
        api.upload_file(path_or_fileobj=(card_dir / f"{m['name']}.md"),
                        path_in_repo="README.md", repo_id=repo,
                        repo_type="model")
        print(f"  https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
