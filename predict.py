"""
Loads the saved hierarchy + emotion models and runs them on a complaint
file, writing a clean Excel with:
  id, voc, AgentNotes,
  ComplaintFor, ComplaintForConfidence,
  CPT, CPTConfidence, CPTAlternatives,
  SPG, SPGConfidence, SPGAlternatives,
  CCC, CCCConfidence, CCCAlternatives,
  HierarchyConfidence, Emotion, EmotionConfidence, NeedsReview

The "Alternatives" columns (CPT/SPG/CCC only - see LEVELS_WITH_SUGGESTIONS)
are a ranked shortlist of the next-best guesses, formatted as
"Label (0.62) | Label (0.55)" - useful at these deeper levels since a
single forced guess is least reliable there; see config.TOPK_SUGGESTIONS.

NeedsReview=True means either the hierarchy path confidence or the emotion
confidence fell below --review-threshold - route these to a human instead
of trusting the auto-fill, same idea as the DMS dropdown but with a safety
net for the cases the model isn't sure about.

Usage:
    python predict.py --input /path/to/complaints.xlsx --output /path/to/predictions.xlsx
    python predict.py --input complaints.xlsx --output predictions.xlsx --review-threshold 0.6
"""
import argparse
import pickle

import pandas as pd

import config
from embeddings import EmbeddingModel

HIERARCHY_LEVELS = ["complainfor", "cpt", "spg", "ccc"]
OUTPUT_COLUMN_NAMES = {
    "complainfor": "ComplaintFor",
    "cpt": "CPT",
    "spg": "SPG",
    "ccc": "CCC",
}
# Levels sparse enough that a ranked shortlist is more honest than one
# forced guess - see the per-level confidence tables train.py prints.
LEVELS_WITH_SUGGESTIONS = ["cpt", "spg", "ccc"]


def _format_topk(topk_list, exclude_first=True) -> str:
    """'Label1 (0.62) | Label2 (0.55)' - skips the first entry since that's
    already the main predicted column; this is just the alternatives."""
    entries = topk_list[1:] if exclude_first else topk_list
    return " | ".join(f"{label} ({score:.2f})" for label, score in entries)


def load_input(path: str) -> pd.DataFrame:
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)

    if config.COL_VOC not in df.columns:
        raise ValueError(f"Input file must have a '{config.COL_VOC}' column.")

    df = df.copy()
    df["voc"] = df[config.COL_VOC].astype(str).str.strip()
    df = df[df["voc"] != ""].reset_index(drop=True)

    notes_col = config.COL_NOTES if config.COL_NOTES in df.columns else None
    if notes_col:
        df["agentnotes"] = df[notes_col].apply(
            lambda v: str(v).strip() if pd.notna(v) and str(v).strip().lower() != "nan" else ""
        )
    else:
        df["agentnotes"] = ""

    df["text"] = df.apply(
        lambda r: (r["voc"] + " " + r["agentnotes"]).strip() if r["agentnotes"] else r["voc"],
        axis=1,
    )

    id_col = config.COL_ID if config.COL_ID in df.columns else None
    df["id"] = df[id_col] if id_col else range(len(df))

    return df[["id", "voc", "agentnotes", "text"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="CRM_.xlsx", help="Complaint file (.xlsx/.csv) with a voc column (default: CRM_.xlsx in this folder)")
    parser.add_argument("--output", default="predictions.xlsx", help="Where to write the combined prediction Excel (default: predictions.xlsx in this folder)")
    parser.add_argument("--review-threshold", type=float, default=config.REVIEW_CONFIDENCE_THRESHOLD,
                         help="Rows with hierarchy path confidence OR emotion confidence below this "
                              "get NeedsReview=True instead of being silently auto-filled. Tune using "
                              "the per-level tables train.py prints - the default is a starting point.")
    args = parser.parse_args()

    with open(config.MODEL_DIR / "hierarchy_model.pkl", "rb") as f:
        hierarchy_model = pickle.load(f)
    with open(config.MODEL_DIR / "emotion_model.pkl", "rb") as f:
        emotion_model = pickle.load(f)

    df = load_input(args.input)
    embed_model = EmbeddingModel()

    # Both hierarchy and emotion use the same combined VOC+AgentNotes text,
    # so one embedding call covers both.
    text_embeddings = embed_model.encode(df["text"].tolist())

    hierarchy_preds, confidences, path_confidence, topk = hierarchy_model.predict(
        text_embeddings, return_topk=True, k=config.TOPK_SUGGESTIONS
    )
    emotion_preds = emotion_model.predict(text_embeddings)
    emotion_conf = emotion_model.confidence(text_embeddings)

    out = pd.DataFrame({
        "id": df["id"],
        "voc": df["voc"],
        "AgentNotes": df["agentnotes"],
    })
    for level in HIERARCHY_LEVELS:
        col = OUTPUT_COLUMN_NAMES[level]
        out[col] = hierarchy_preds[level]
        out[f"{col}Confidence"] = confidences[level].round(3)
        if level in LEVELS_WITH_SUGGESTIONS:
            out[f"{col}Alternatives"] = [_format_topk(row) for row in topk[level]]
    out["HierarchyConfidence"] = path_confidence.round(3)
    out["Emotion"] = emotion_preds
    out["EmotionConfidence"] = emotion_conf.round(3)
    out["NeedsReview"] = (path_confidence < args.review_threshold) | (emotion_conf < args.review_threshold)

    out.to_excel(args.output, index=False)
    n_review = int(out["NeedsReview"].sum())
    print(f"Wrote {len(out)} predictions to {args.output}")
    print(f"  {n_review} rows ({n_review / len(out):.1%}) flagged NeedsReview=True "
          f"(confidence < {args.review_threshold})")


if __name__ == "__main__":
    main()