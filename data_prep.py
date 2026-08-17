"""
Loads the CRM complaint export and the tone-labeled file, cleans them,
and canonicalizes the hierarchy labels (see config.canonicalize_series
for why this is necessary).
"""
import csv
import io
import re

import pandas as pd

import config

_WORDING_SUFFIX_RE = re.compile(r"\btroubles?\b|\bissues?\b|\bproblems?\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _core_form(s: str) -> str:
    """Normalize a label down to its bare concept: lowercase, strip common
    'TROUBLES'/'issues'/'problems' suffix words, collapse punctuation."""
    s = _WORDING_SUFFIX_RE.sub("", s.lower())
    return _NON_ALNUM_RE.sub(" ", s).strip()

def _merge_wording_variants(df: pd.DataFrame, col: str, parent_cols: list) -> pd.Series:
    """
    Some CCC/SPG/CPT values exist in two different data-entry conventions
    for the exact same real category - e.g. 'HEADLAMP TROUBLES' (ALL CAPS,
    high volume) vs 'Headlamp issues' (sentence case, low volume). These
    survive config.canonicalize_series because that function only merges
    pure case/whitespace duplicates, not different WORDING for the same
    concept.

    SCOPED to parent_cols: two spellings only get merged if they occur
    under an identical parent path (e.g. same complainfor+cpt+spg for a
    ccc merge). Without this scoping, two DIFFERENT real categories that
    happen to follow the same "X issues" / "X TROUBLES" spelling pattern
    under different parents would get silently merged into one - checked
    against the real CRM data, this actually happens (e.g. 'Battery
    issues' sits under a different SPG than 'BATTERY TROUBLES').

    Prints what it merges, and under which parent, so it's never silent.
    """
    result = df[col].copy()
    remap_log = []

    for parent_key, group in df.groupby(parent_cols, dropna=False):
        idx = group.index
        counts = df.loc[idx, col].value_counts()
        core_groups = {}
        for val, cnt in counts.items():
            if val is None:
                continue
            core = _core_form(val)
            core_groups.setdefault(core, []).append((val, cnt))
        for variants in core_groups.values():
            if len(variants) < 2:
                continue
            variants.sort(key=lambda vc: -vc[1])
            canonical = variants[0][0]
            for v, cnt in variants[1:]:
                mask = df.loc[idx, col] == v
                target_idx = idx[mask]
                result.loc[target_idx] = canonical
                remap_log.append((parent_key, v, canonical, len(target_idx)))

    if remap_log:
        total_rows = sum(n for *_, n in remap_log)
        print(f"  Merging {len(remap_log)} parent-scoped wording-variant labels "
              f"({total_rows} rows total) into their larger counterpart within the same parent:")
        for parent_key, old, new, n in remap_log:
            print(f"    under {parent_key}: {old!r} -> {new!r}  ({n} rows)")

    return result

def load_crm(path: str) -> pd.DataFrame:
    """
    Load the CRM Excel export and return a cleaned dataframe with:
      id, voc, agentnotes, text (voc + agentnotes combined),
      complainfor, cpt, spg, ccc   (all canonicalized text labels)

    Rows missing voc or any hierarchy label are dropped (can't train/eval on them).
    Rows belonging to a class with too few examples (see config.MIN_CLASS_SIZE_*)
    are also dropped, level-by-level - too few examples to learn a reliable
    pattern or evaluate honestly.
    """
    df = pd.read_excel(path)

    required = [config.COL_ID, config.COL_VOC, config.COL_COMPLAINFOR,
                config.COL_CPT, config.COL_SPG, config.COL_CCC]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CRM file is missing expected columns: {missing_cols}")

    df = df[[config.COL_ID, config.COL_VOC, config.COL_NOTES,
             config.COL_COMPLAINFOR, config.COL_CPT, config.COL_SPG, config.COL_CCC]].copy()
    df.columns = ["id", "voc", "agentnotes", "complainfor", "cpt", "spg", "ccc"]

    df = df.dropna(subset=["voc", "complainfor", "cpt", "spg", "ccc"])
    df["voc"] = df["voc"].astype(str).str.strip()
    df = df[df["voc"] != ""]

    df["agentnotes"] = df["agentnotes"].apply(
        lambda v: str(v).strip() if pd.notna(v) and str(v).strip().lower() != "nan" else ""
    )
    df["text"] = df.apply(
        lambda r: (r["voc"] + " " + r["agentnotes"]).strip() if r["agentnotes"] else r["voc"],
        axis=1,
    )

    # canonicalize category labels (fixes case/wording duplicates, see config.py)
    # NOTE: alias maps are scoped per column - never share these across
    # complainfor/cpt/spg/ccc (see the comment above CPT_LABEL_ALIASES for why).
    df["complainfor"] = config.canonicalize_series(df["complainfor"], config.COMPLAINFOR_LABEL_ALIASES)
    df["cpt"] = config.canonicalize_series(df["cpt"], config.CPT_LABEL_ALIASES)
    df["spg"] = config.canonicalize_series(df["spg"], config.SPG_LABEL_ALIASES)
    df["ccc"] = config.canonicalize_series(df["ccc"], config.CCC_LABEL_ALIASES)

    # separate pass: merge same-category-different-wording-convention labels
    # (e.g. 'HEADLAMP TROUBLES' vs 'Headlamp issues') - see _merge_wording_variants
    # above. Not applied to complainfor - its alias above already covers the
    # one real duplicate there, and its 5 values don't have this TROUBLES/
    # issues-style split.
    df["cpt"] = _merge_wording_variants(df, "cpt", parent_cols=["complainfor"])
    df["spg"] = _merge_wording_variants(df, "spg", parent_cols=["complainfor", "cpt"])
    df["ccc"] = _merge_wording_variants(df, "ccc", parent_cols=["complainfor", "cpt", "spg"])

    df = df.dropna(subset=["complainfor", "cpt", "spg", "ccc"]).reset_index(drop=True)

    # drop rare classes level-by-level (each within whatever rows survived
    # the level above) - too few examples to learn a reliable pattern or
    # evaluate honestly.
    df = _drop_rare_classes(df, "complainfor", config.MIN_CLASS_SIZE_COMPLAINFOR)
    df = _drop_rare_classes(df, "cpt", config.MIN_CLASS_SIZE_CPT)
    df = _drop_rare_classes(df, "spg", config.MIN_CLASS_SIZE_SPG)
    df = _drop_rare_classes(df, "ccc", config.MIN_CLASS_SIZE_CCC)

    return df


def _drop_rare_classes(df: pd.DataFrame, col: str, min_class_size: int) -> pd.DataFrame:
    """Remove rows whose value in `col` occurs fewer than min_class_size times
    in the current dataframe. Prints what got dropped so it's never silent."""
    counts = df[col].value_counts()
    rare = counts[counts < min_class_size]
    if len(rare) > 0:
        print(f"  Dropping {col} classes with fewer than {min_class_size} rows "
              f"({len(rare)} classes, {int(rare.sum())} rows): {list(rare.index)}")
        df = df[~df[col].isin(rare.index)]
    return df.reset_index(drop=True)


def _parse_double_quoted_csv(path: str) -> pd.DataFrame:
    """
    Some tone-label exports come with every row wrapped in an extra layer
    of CSV quoting (each full row is itself one quoted field). This parses
    that format if present; falls back to a normal pd.read_csv otherwise.
    """
    with open(path, newline="", encoding="utf-8") as f:
        outer_rows = list(csv.reader(f))

    is_double_wrapped = all(len(r) == 1 for r in outer_rows[:20])
    if not is_double_wrapped:
        return pd.read_csv(path)

    inner_text = "\n".join(r[0] for r in outer_rows)
    inner_rows = list(csv.reader(io.StringIO(inner_text)))
    return pd.DataFrame(inner_rows[1:], columns=inner_rows[0])


def load_tone_labels(path: str, crm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the tone-labeled file and attach agentnotes from the CRM data
    (by id) so the emotion model can use the same VOC+notes combined text
    as the hierarchy models. Returns: id, text, emotion
    """
    tone_df = _parse_double_quoted_csv(path)

    required = [config.TONE_COL_ID, config.TONE_COL_VOC, config.TONE_COL_LABEL]
    missing_cols = [c for c in required if c not in tone_df.columns]
    if missing_cols:
        raise ValueError(f"Tone file is missing expected columns: {missing_cols}")

    tone_df = tone_df[[config.TONE_COL_ID, config.TONE_COL_VOC, config.TONE_COL_LABEL]].copy()
    tone_df.columns = ["id", "voc", "tone_raw"]
    tone_df["id"] = pd.to_numeric(tone_df["id"], errors="coerce")
    tone_df = tone_df.dropna(subset=["id"])
    tone_df["id"] = tone_df["id"].astype(int)

    tone_df["emotion"] = (
        tone_df["tone_raw"].astype(str).str.strip().str.lower().map(config.EMOTION_LABEL_MAP)
    )
    tone_df = tone_df.dropna(subset=["emotion"])

    notes_lookup = crm_df.set_index("id")["agentnotes"].to_dict()
    voc_lookup = crm_df.set_index("id")["voc"].to_dict()

    def combined_text(row):
        # prefer the CRM's own voc text (source of truth) when we have a match,
        # otherwise fall back to the voc text stored in the tone file itself
        voc = voc_lookup.get(row["id"], row["voc"])
        notes = notes_lookup.get(row["id"], "")
        voc = str(voc).strip()
        notes = str(notes).strip() if notes and str(notes).lower() != "nan" else ""
        return (voc + " " + notes).strip() if notes else voc

    tone_df["text"] = tone_df.apply(combined_text, axis=1)
    tone_df = tone_df[tone_df["text"].str.len() > 0]

    return tone_df[["id", "text", "emotion"]].reset_index(drop=True)