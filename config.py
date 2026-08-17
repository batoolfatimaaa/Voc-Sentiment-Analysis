"""
Paths + label canonicalization for the VOC classification pipeline.

Why canonicalization exists:
The CRM's numeric cpt/spg/ccc IDs are NOT reliable category identifiers -
the same text label (e.g. "Product Quality") is stored under several
different numeric codes, and the same category is sometimes typed with
different casing/spacing ("After-Sales Service" vs "After-sales service").
We always train and predict on the cleaned TEXT label, never the numeric id.
"""
import re
from pathlib import Path

# ---- paths -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "embedding_cache"
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# If your VOC text / agent notes are in Chinese (or mixed English/Chinese),
# swap to a multilingual model instead - all-MiniLM-L6-v2 is English-centric
# and will underperform on Chinese text. Good options:
#   "paraphrase-multilingual-MiniLM-L12-v2"  (drop-in, still small/fast)
#   "BAAI/bge-m3"                            (stronger, slower)

# ---- CRM column names --------------------------------------------------
COL_VOC = "voc"
COL_NOTES = "agentnotes"
COL_ID = "id"
COL_COMPLAINFOR = "complainfor"
COL_CPT = "cptvalue"
COL_SPG = "spgvalue"
COL_CCC = "cccvalue"

# ---- tone/emotion file column names ------------------------------------
TONE_COL_ID = "id"
TONE_COL_VOC = "voc"
TONE_COL_LABEL = "tone_label"

EMOTION_CLASSES = ["Angry", "Frustrated", "Calm", "Neutral"]

# Some CRMs use "Frustrated/Disappointed" wording - normalize into our
# 4-class scheme. Extend this if you see other spellings show up.
EMOTION_LABEL_MAP = {
    "frustrated": "Frustrated",
    "frustrated/disappointed": "Frustrated",
    "disappointed": "Frustrated",
    "angry": "Angry",
    "calm": "Calm",
    "neutral": "Neutral",
}

# Manually-known wording variants that mean the same category but aren't
# simple case/whitespace differences (case/whitespace dupes are merged
# automatically - see canonicalize_series below). Add to this as you spot
# more in the data.
#
# IMPORTANT: these are scoped PER COLUMN on purpose. A single global alias
# dict previously caused a real bug: complainfor's genuine value "After
# Sales" collided with a CPT-column alias meant for "after sale(s)" ->
# "After-Sales Service", silently corrupting 7600+ rows' top-level label.
# Merging is only safe within the SAME category/column - never share these
# across complainfor/cpt/spg/ccc.
#
# complainfor DOES now have one confirmed alias (checked against the actual
# CRM export): 'After-sales service' is a 1-row typo of 'After Sales'
# (7602 rows), not a real 6th category - without this it gets silently
# dropped as "too rare" every time, and the model can never predict it.
COMPLAINFOR_LABEL_ALIASES = {
    "after-sales service": "After Sales",
}
CPT_LABEL_ALIASES = {
    "sales service": "Sales Services",  # confirmed intentional merge (singular/plural variant)
}
SPG_LABEL_ALIASES = {
    "quality of maintenance": "Maintenance quality",
}

CCC_LABEL_ALIASES = {
    # confirmed same real category, just typed with/without "ed" - verified
    # both spellings share the same parent (complainfor/cpt/spg) in the actual CRM data
    "vehicle delayed": "Vehicle Delay",
    "invoice delayed": "Invoice Delay",
}


# ---- rare-class filtering ------------------------------------------------
# Classes with fewer rows than this are dropped entirely before training -
# too few examples to learn a reliable pattern or evaluate honestly.
# Applied level-by-level in data_prep.load_crm(): complainfor first, then
# cpt, then spg, then ccc (each within whatever rows survived the level above).
MIN_CLASS_SIZE_COMPLAINFOR = 20  
MIN_CLASS_SIZE_CPT = 15           # back to original
MIN_CLASS_SIZE_SPG = 15           # back to original
MIN_CLASS_SIZE_CCC = 10           # back to original
# ---- model selection & confidence -----------------------------------------
# Inside a hierarchy node, classes are split into "big" (>= this many rows)
# and "small". What the node actually does depends on the mix - see
# models.py: _NodeModel.fit():
#   - every class is big              -> plain LinearSVC
#   - every class is small            -> plain nearest-centroid (cosine
#                                         similarity to each class's average
#                                         embedding - degrades far more
#                                         gracefully than an SVM margin when
#                                         a class only has a handful of rows)
#   - a MIX of big and small classes  -> "hybrid": SVC trained only on the
#                                         big classes (so those keep full
#                                         accuracy), plus a centroid over
#                                         ALL classes that can override SVC's
#                                         answer for a genuinely close match
#                                         to a small class (see
#                                         RARE_CLASS_OVERRIDE_THRESHOLD below)
MIN_SAMPLES_FOR_SVC = 20
# In a "hybrid" node (mix of big and small classes - see models.py), a rare
# class is only allowed to override the SVC prediction if the point's
# centroid-similarity to it is at least this high. Keeps the fix from
# firing on weak/borderline matches.
RARE_CLASS_OVERRIDE_THRESHOLD = 0.5
# Predictions (hierarchy path confidence OR emotion confidence) below this
# get flagged NeedsReview=True in predict.py's output, instead of being
# silently trusted. Tune this by looking at the "accuracy at different
# confidence thresholds" table train.py prints (per-level, not just the
# combined full path) - pick the threshold where accuracy on the "trusted"
# side is high enough for your use case. 0.45 is a starting point, not a
# final answer - re-tune once you've looked at your own printed tables,
# since centroid-node confidence is on a raw cosine-similarity scale
# (typically 0.3-0.8) rather than a 0-1 probability.
REVIEW_CONFIDENCE_THRESHOLD = 0.45

# How many ranked alternatives predict.py suggests for CPT/SPG/CCC, in
# addition to its single best guess. Given how little data some parent
# paths have (many have well under 10 rows to choose from - see the
# project notes), a ranked shortlist for a human to pick from is a more
# realistic integration than forcing one hard auto-fill at these levels.
TOPK_SUGGESTIONS = 3


def _clean_text(s: str) -> str:
    """Collapse internal whitespace, strip ends."""
    return re.sub(r"\s+", " ", str(s).strip())


def build_canonical_map(values, alias_map=None):
    """
    Given an iterable of raw label strings, return a dict mapping every
    observed raw value -> a single canonical spelling.

    Step 1: group values that are identical once whitespace-normalized and
    lower-cased (pure formatting differences) - pick the most frequent
    original spelling in each group as the canonical form.
    Step 2: apply `alias_map` on top for known wording variants THAT ARE
    SPECIFIC TO THIS COLUMN. Never pass a shared/global alias map here -
    see the comment above CPT_LABEL_ALIASES for why that's dangerous.
    """
    from collections import Counter

    alias_map = alias_map or {}
    cleaned = [_clean_text(v) for v in values if v is not None and str(v).strip()]
    counts = Counter(cleaned)

    groups = {}
    for v, c in counts.items():
        key = v.lower()
        groups.setdefault(key, []).append((v, c))

    canon_by_key = {}
    for key, variants in groups.items():
        variants.sort(key=lambda vc: -vc[1])
        canon_by_key[key] = variants[0][0]

    raw_to_canon = {}
    for v in cleaned:
        canon = canon_by_key[v.lower()]
        # apply alias on top, keyed by lowercase canonical text
        alias = alias_map.get(canon.lower())
        raw_to_canon[v] = alias if alias else canon

    return raw_to_canon


def canonicalize_series(series, alias_map=None):
    """
    Clean + canonicalize a pandas Series of category labels.

    `alias_map`, if given, must be specific to THIS column (e.g. pass
    config.CPT_LABEL_ALIASES when canonicalizing the cpt column, or
    config.COMPLAINFOR_LABEL_ALIASES for complainfor). Reusing another
    column's alias map here risks exactly the corruption bug this
    parameter was added to prevent - see the comment above
    COMPLAINFOR_LABEL_ALIASES.
    """
    cleaned = series.apply(lambda v: _clean_text(v) if v is not None and str(v).strip() and str(v).lower() != "nan" else None)
    raw_to_canon = build_canonical_map(cleaned.dropna().tolist(), alias_map)
    return cleaned.map(lambda v: raw_to_canon.get(v, v) if v is not None else None)