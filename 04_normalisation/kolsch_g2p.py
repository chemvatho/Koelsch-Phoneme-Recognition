#!/usr/bin/env python3
"""Kölsch grapheme-to-phoneme (G2P): orthography -> IPA.

Rule-based, deterministic converter for Kölsch (Ripuarian, Cologne). This is the
engine behind the pronunciation dictionary — it maps any Kölsch spelling to a
word-form IPA string, so it also covers out-of-vocabulary words.

    from kolsch_g2p import word_to_ipa, text_to_ipa
    word_to_ipa("kölsch")     -> "kœlʃ"
    text_to_ipa("dat es su")  -> "dat es su" mapped word by word

Notes
-----
* Output is *word-form* IPA (phonemes glued within a word). Split into phonemes
  downstream with tokenize_ipa (Notebook 4).
* <ch> uses the ich-/ach-laut split: ç after front vowels & consonants, χ after
  back vowels. Set CH_ALWAYS_UVULAR=True to force χ everywhere (older behaviour).
* Rules: Kölsch diphthongs (eï→ɛɪ, äu/eu→ɔʏ …), long vowels, uvular /ʁ/,
  <z>/<tz>→/t͡s/, <w>→/v/, <j>→/j/, and word-final devoicing (b→p, d→t, g→k).
"""
import re

CH_ALWAYS_UVULAR = False
FRONT = set("eiäöüïy")          # vowels that trigger the ich-laut (ç)
VOWELS_IPA = set("aɛeɪiɔoʊuʏyœøəɐ")

# Ordered grapheme rules — longest match first. Each is (grapheme, ipa).
RULES = [
    # 3–4 char
    ("tsch", "t͡ʃ"), ("dsch", "d͡ʒ"), ("sch", "ʃ"),
    # doubled / long vowels
    ("aa", "aː"), ("ee", "eː"), ("oo", "oː"), ("uu", "uː"),
    ("ää", "ɛː"), ("öö", "øː"), ("üü", "yː"), ("ie", "iː"),
    # Kölsch + German diphthongs (order matters: ï-forms before plain)
    ("eï", "ɛɪ"), ("aï", "aɪ"), ("oï", "ɔɪ"), ("uï", "ʊɪ"),
    ("äu", "ɔʏ"), ("eu", "ɔʏ"), ("au", "aʊ"),
    ("ei", "aɪ"), ("ai", "aɪ"), ("oi", "ɔɪ"),
    # consonant digraphs
    ("ck", "k"), ("ch", "ç"), ("ng", "ŋ"), ("pf", "p͡f"), ("ph", "f"),
    ("qu", "kv"), ("th", "t"), ("ts", "t͡s"), ("tz", "t͡s"), ("ss", "s"),
    # single graphemes
    ("a", "a"), ("e", "ɛ"), ("i", "ɪ"), ("o", "ɔ"), ("u", "ʊ"),
    ("ä", "ɛ"), ("ö", "œ"), ("ü", "ʏ"), ("y", "ʏ"),
    ("b", "b"), ("c", "k"), ("d", "d"), ("f", "f"), ("g", "ɡ"),
    ("h", "h"), ("j", "j"), ("k", "k"), ("l", "l"), ("m", "m"),
    ("n", "n"), ("p", "p"), ("r", "ʁ"), ("s", "s"), ("t", "t"),
    ("v", "v"), ("w", "v"), ("x", "ks"), ("z", "t͡s"), ("ß", "s"),
]
_MAX = max(len(g) for g, _ in RULES)
_APOS = "'’ʼʻ`"

def _clean(word):
    w = word.lower().strip()
    for a in _APOS:
        w = w.replace(a, "")            # elisions: d'r -> dr, m'r -> mr
    return re.sub(r"[^a-zäöüßï]", "", w)

def word_to_ipa(word):
    """Convert one Kölsch orthographic word to word-form IPA (glued phonemes)."""
    w = _clean(word)
    if not w:
        return ""
    out, i, n = [], 0, len(w)
    while i < n:
        matched = False
        for length in range(min(_MAX, n - i), 0, -1):
            g = w[i:i+length]
            for gr, ipa in RULES:
                if gr == g:
                    # ich-/ach-laut split for <ch>
                    if g == "ch" and not CH_ALWAYS_UVULAR:
                        prev = w[i-1] if i > 0 else ""
                        ipa = "χ" if prev in "aouå" else "ç"
                    out.append(ipa); i += length; matched = True
                    break
            if matched:
                break
        if not matched:
            i += 1                       # skip an unmappable char
    ipa = "".join(out)
    ipa = _schwa(w, ipa)
    ipa = _final_devoice(ipa)
    return ipa

def _schwa(orth, ipa):
    # final unstressed -e -> ə ; final -er -> ɐ  (typical German/Kölsch reduction)
    if orth.endswith("er") and ipa.endswith("ɛʁ"):
        return ipa[:-2] + "ɐ"
    if orth.endswith("e") and ipa.endswith("ɛ") and len(orth) > 1:
        return ipa[:-1] + "ə"
    return ipa

def _final_devoice(ipa):
    # Auslautverhärtung on the final phoneme: b->p, d->t, ɡ->k, v->f, z->s
    dev = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s"}
    for src, tgt in dev.items():
        if ipa.endswith(src):
            return ipa[:-len(src)] + tgt
    return ipa

def text_to_ipa(text):
    """Convert a whitespace-tokenised Kölsch string to word-form IPA."""
    return " ".join(w for w in (word_to_ipa(t) for t in str(text).split()) if w)

if __name__ == "__main__":
    for w in ["un", "ich", "dat", "kölsch", "Schneï", "eïmol", "Wasser", "maache"]:
        print(f"{w:10s} -> {word_to_ipa(w)}")
