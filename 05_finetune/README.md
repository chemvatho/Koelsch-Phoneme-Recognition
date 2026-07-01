# 5 · Fine-tuning Wav2Vec2 XLS-R-300M

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemvatho/kolsch-tandem/blob/main/05_finetune/05_wav2vec2_finetune.ipynb)

Fine-tunes `facebook/wav2vec2-xls-r-300m` with a CTC head for Kölsch phoneme
recognition: builds the phoneme vocabulary, prepares the dataset, and trains
with the project configuration (lr 3e-5 cosine, warm-up 500, effective batch 16,
fp16 + gradient checkpointing, 150 epochs, best checkpoint on **validation WER**,
feature encoder frozen, `ctc_zero_infinity`).

**Input:** segment manifest (Notebook 3) + `phonetic` labels (Notebook 4).
**Output:** saved model + processor → evaluated in Notebook 6.
**Result:** best ~15.2% val WER; **14.63% WER / 11.75% CER** on test. Needs a GPU.
