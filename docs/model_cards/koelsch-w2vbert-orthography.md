---
language: [ksh, de]
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags: [koelsch, kolsch, ripuarian, dialect, low-resource, ctc, orthography, transcription]
---

# Kölsch orthographic recogniser — w2v-BERT 2.0

Transcribes **Kölsch** speech into Kölsch spelling. Fine-tuned from `facebook/w2v-bert-2.0` with a CTC head over the book's German-letter orthography.

Note that `w2v-bert-2.0` ships no CTC head, so `lm_head` *and* the conv adapter were initialised randomly — the stronger multilingual pretraining had to pay for that.

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
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

proc  = Wav2Vec2BertProcessor.from_pretrained("Vatho/koelsch-w2vbert-orthography")
model = Wav2Vec2BertForCTC.from_pretrained("Vatho/koelsch-w2vbert-orthography").eval()

wav, _ = librosa.load("clip.wav", sr=16000)
inp = proc(wav, sampling_rate=16000, return_tensors="pt")
with torch.inference_mode():
    ids = model(**inp).logits.argmax(-1)
print(proc.batch_decode(ids)[0])
# -> ich well es och nit mih widder dun
```

Full pipeline, notebooks and the alignment tooling:
<https://github.com/chemvatho/Koelsch-Phoneme-Recognition>

```bash
python tools/try_models.py transcribe clip.wav
```

## How it scores

| | |
|---|---|
| **CER** | **11.3 %** |
| WER | 34.0 % |

582 held-out utterances, recomputed from the stored test predictions.

**CER is the headline here, not WER.** Kölsch has no spelling standard, so *janz*/*ganz* and *zusamme*/*zosamme* are one word spelled two ways and WER charges full price for the difference.

> **These numbers are optimistic and it is worth knowing why.** The test split
> is **not speaker-disjoint** — 22 % of its test clips (128/582) come from a training speaker. So they measure how well the model
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
