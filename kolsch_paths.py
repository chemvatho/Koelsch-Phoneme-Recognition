"""Single source of truth for paths — import this from any notebook or script.

Works identically in VS Code, Jupyter and Google Colab because it locates the
repository from the location of THIS file, not from the current working dir.

    from kolsch_paths import ROOT, DATA, PAGES, AUDIO, TRANS, SEG, INDEX, LEXICON, MODELS
"""
import os
from pathlib import Path

ROOT    = str(Path(__file__).resolve().parent)   # this file lives at the repo root
DATA    = os.path.join(ROOT, "data")
PAGES   = os.path.join(DATA, "pages")
AUDIO   = os.path.join(DATA, "audio")
TRANS   = os.path.join(DATA, "transcripts")
SEG     = os.path.join(DATA, "segments")
INDEX   = os.path.join(DATA, "index.csv")
LEXICON = os.path.join(DATA, "lexicon.csv")
MODELS  = os.path.join(ROOT, "models")

for _d in (PAGES, AUDIO, TRANS, SEG, MODELS):
    os.makedirs(_d, exist_ok=True)
