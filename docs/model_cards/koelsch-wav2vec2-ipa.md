---
language: [ksh, de]
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags: [koelsch, kolsch, ripuarian, dialect, low-resource, ctc, phoneme-recognition, forced-alignment, ipa]
---

# Kölsch phoneme recogniser — wav2vec2 XLS-R-300M (IPA)

IPA phoneme recognition for **Kölsch** (Ripuarian German, Cologne). Fine-tuned from `facebook/wav2vec2-xls-r-300m` with a CTC head over a 48-symbol IPA inventory.

**This is also the forced-alignment model.** Its frame-level CTC posteriors drive `torchaudio.functional.forced_align` to produce Praat TextGrids with word and phone tiers — see notebook 9 in the repository.

Trained during the **CIF Tandem Fellowship** at IfL-Phonetik, University of
Cologne, on *Alles Kölsch* (Bhatt & Lindlar 1998) — 4,670 utterances, 4.5 hours
of **spontaneous** Cologne dialect speech from 105 speakers aged 10–88 across 49
neighbourhoods.

Kölsch is Ripuarian German. It has no public speech dataset and no standardised
spelling: **94.1 % of Kölsch word tokens are out of vocabulary** against the
152,766-word `german_mfa` dictionary. It is not German text.

## Use it

```python
import torch, librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

proc  = Wav2Vec2Processor.from_pretrained("Vatho/koelsch-wav2vec2-ipa")
model = Wav2Vec2ForCTC.from_pretrained("Vatho/koelsch-wav2vec2-ipa").eval()

wav, _ = librosa.load("clip.wav", sr=16000)
with torch.inference_mode():
    ids = model(proc(wav, sampling_rate=16000,
                     return_tensors="pt").input_values).logits.argmax(-1)
print(proc.batch_decode(ids)[0])
# -> ɪ ç v ə l ə s ɔ χ n ɪ t m iː v ɪ d ɐ d ʊ n
```

Full pipeline, notebooks and the alignment tooling:
<https://github.com/chemvatho/Koelsch-Phoneme-Recognition>

```bash
python tools/try_models.py align clip.wav        # TextGrid + plot
```

## How it scores

| | |
|---|---|
| **PER** | **15.3 %** |
| CER over the IPA stream | 15.3 % |

467 held-out utterances, 21,463 reference phones, recomputed from the stored test predictions. For context, an off-the-shelf multilingual Wav2Vec2Phoneme scores ~33 % PER on comparable German-dialect material.

> **These numbers are optimistic and it is worth knowing why.** The test split
> is **not speaker-disjoint** — 103 of its 105 speakers also appear in training. So they measure how well the model
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
- **For alignment, roughly four fifths of every phone duration is a rule, not a measurement.** CTC is peaky: labelled frames cover only 14–21 % of the timeline, and which rule fills the gaps matters more than the model does. Read `09_alignment/` before taking any duration from a TextGrid this produces.

## Licence and rights

The **code** in the repository is MIT. **These weights are released for research
use** under CC-BY-NC-SA-4.0. The underlying *Alles Kölsch* corpus is **not**
redistributed here and is not covered by either: it belongs to the **Akademie
för uns kölsche Sproch**, who should be contacted for corpus access.

## Citation

```bibtex
@misc{chem2026koelsch,
  author = {Chem, Vatho and R\"ossig, Simon and Greisbach, Reinhold},
  title  = {K\"olsch Phoneme Recognition: an open pipeline from a printed
            dialect corpus to phoneme recognition and forced alignment},
  year   = {2026},
  note   = {CIF Tandem Fellowship, IfL-Phonetik, University of Cologne},
  howpublished = {\url{https://github.com/chemvatho/Koelsch-Phoneme-Recognition}}
}
```
