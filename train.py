"""
Trains the ComplaintFor -> CPT -> SPG -> CCC hierarchy (on the ~10k labeled
CRM rows) and the emotion model (on the tone-labeled rows), reports
accuracy for each, and saves everything to models/.

What changed from the original all-LinearSVC version:
  - Each hierarchy node now picks its own classifier based on how much data
    it actually has (see models.py: constant / svc / centroid / hybrid).
    The "hybrid" case matters most in practice: a node with a mix of
    well-populated and rare classes trains SVC only on the well-populated
    ones (so those don't lose accuracy) and layers a centroid fallback on
    top just for the rare ones, instead of the whole node degrading to
    centroid-only just because one sibling class is small.
  - Both the hierarchy and the emotion model report a confidence score per
    prediction. train.py prints "accuracy at different confidence
    thresholds" - both a combined full-path table and a PER-LEVEL table,
    since complainfor/cpt/spg/ccc have very different numbers of classes
    per node and one global threshold doesn't mean the same thing at every
    level - so you can see how much cleaner the trusted subset is if you
    route low-confidence rows to a human instead of auto-filling them.
  - Now also prints a per-category (precision/recall/F1/support) breakdown
    for every level, not just one overall accuracy number per level - see
    print_per_class_report().

Usage:
    python train.py --crm /path/to/CRM_.xlsx --tone /path/to/emotion-labelled.csv
"""
import argparse
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
import data_prep
from embeddings import EmbeddingModel
from models import EmotionClassifier, HierarchicalClassifier

HIERARCHY_LEVELS = ["complainfor", "cpt", "spg", "ccc"]
# Centroid-node confidence is now raw cosine similarity (typically 0.3-0.8),
# not a 0-1 probability, so the thresholds worth checking are lower/denser
# than a plain accuracy-style scale.
CONFIDENCE_THRESHOLDS_TO_REPORT = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]


def print_per_class_report(y_true, y_pred, level_name):
    """Precision/recall/F1/support for every individual category at this
    level - not just one blended accuracy number. NOTE: categories with
    very few test examples (low 'support') will have noisy, unreliable
    scores here - don't over-read a 100% or 0% on a category that only
    had a handful of test rows."""
    print(f"\n  --- {level_name}: accuracy per category ---")
    print(classification_report(y_true, y_pred, zero_division=0))


def evaluate_hierarchy(df, embed_model: EmbeddingModel):
    print("\n--- Hierarchy accuracy (held-out test split) ---")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["complainfor"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = embed_model.encode(train_df["text"].tolist())
    X_test = embed_model.encode(test_df["text"].tolist())

    clf = HierarchicalClassifier(HIERARCHY_LEVELS).fit(X_train, train_df)
    preds, confidences, path_confidence = clf.predict(X_test, return_confidence=True)

    for level in HIERARCHY_LEVELS:
        acc = accuracy_score(test_df[level], preds[level])
        f1 = f1_score(test_df[level], preds[level], average="macro", zero_division=0)
        print(f"  {level:12s}  accuracy={acc:.3f}   macro-F1={f1:.3f}")

    # per-category breakdown for every level - this is what your lead asked for
    for level in HIERARCHY_LEVELS:
        print_per_class_report(test_df[level], preds[level], level)

    correct = np.array([
        all(preds[level][i] == test_df[level].iloc[i] for level in HIERARCHY_LEVELS)
        for i in range(len(test_df))
    ])
    print(f"  {'full path':12s}  accuracy={correct.mean():.3f}   (all 4 levels correct simultaneously)")

    print("\n  Full-path accuracy at different confidence thresholds:")
    print("  (this tells you how much cleaner predictions get if you route the")
    print("   low-confidence ones to a human instead of trusting every row)")
    for thresh in CONFIDENCE_THRESHOLDS_TO_REPORT:
        mask = path_confidence >= thresh
        coverage = mask.mean()
        acc_at_thresh = correct[mask].mean() if mask.any() else float("nan")
        print(f"    confidence >= {thresh:.2f}:  covers {coverage:.1%} of complaints, "
              f"accuracy {acc_at_thresh:.3f} on those")

    # PER-LEVEL tables, not just the combined full path - the four levels
    # have very different numbers of classes per node (complainfor: ~5,
    # ccc: often 20+), so one global threshold doesn't mean the same thing
    # at every level. Use this to pick separate auto-accept thresholds per
    # level - e.g. auto-fill complainfor/cpt confidently, but always show
    # spg/ccc as a shortlist rather than a forced single guess.
    print("\n  Per-level accuracy at different confidence thresholds:")
    for level in HIERARCHY_LEVELS:
        level_correct = (preds[level] == test_df[level].values)
        level_conf = confidences[level]
        print(f"\n  {level}:")
        for thresh in CONFIDENCE_THRESHOLDS_TO_REPORT:
            mask = level_conf >= thresh
            coverage = mask.mean()
            acc_at_thresh = level_correct[mask].mean() if mask.any() else float("nan")
            print(f"    confidence >= {thresh:.2f}:  covers {coverage:.1%}, accuracy {acc_at_thresh:.3f}")

    # transparency: how many nodes ended up using centroid vs SVC vs constant
    # vs hybrid (informational only - tells you where the size-aware switch
    # kicked in, using the same train-split model already fit above)
    kind_counts = {"constant": 0, "svc": 0, "centroid": 0, "hybrid": 0}
    for level_nodes in clf.nodes:
        for node in level_nodes.values():
            kind_counts[node.kind] += 1
    print(f"\n  Node model types across the hierarchy: {kind_counts}")
    print("  ('centroid' nodes are the ones that would've been unreliable LinearSVC fits before)")

    small_groups = df["ccc"].value_counts()
    small_groups = small_groups[small_groups < 10]
    if len(small_groups):
        print(f"\n  Note: {len(small_groups)} CCC categories still have fewer than 10 examples "
              f"in the whole dataset. The centroid fallback handles these much better than SVC "
              f"did, but for a real accuracy boost on these specifically, augment them with "
              f"paraphrases (see the augmentation suggestion in the project notes).")


def evaluate_emotion(tone_df, embed_model: EmbeddingModel):
    print("\n--- Emotion accuracy (5-fold stratified cross-validation) ---")
    X = embed_model.encode(tone_df["text"].tolist())
    y = tone_df["emotion"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    all_conf, all_correct = [], []
    all_true, all_preds = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = EmotionClassifier().fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        conf = clf.confidence(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))
        all_conf.append(conf)
        all_correct.append(preds == y[test_idx])
        all_true.append(y[test_idx])
        all_preds.append(preds)

    print(f"  overall     accuracy={np.mean(accs):.3f} (+/- {np.std(accs):.3f})   "
          f"macro-F1={np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")

    # per-emotion-class breakdown, built from predictions collected across all 5 folds
    print_per_class_report(np.concatenate(all_true), np.concatenate(all_preds), "emotion")

    all_conf = np.concatenate(all_conf)
    all_correct = np.concatenate(all_correct)
    print("\n  Accuracy at different confidence thresholds:")
    for thresh in CONFIDENCE_THRESHOLDS_TO_REPORT:
        mask = all_conf >= thresh
        coverage = mask.mean()
        acc_at_thresh = all_correct[mask].mean() if mask.any() else float("nan")
        print(f"    confidence >= {thresh:.2f}:  covers {coverage:.1%} of complaints, "
              f"accuracy {acc_at_thresh:.3f} on those")

    print("\n  class distribution:")
    for cls, n in tone_df["emotion"].value_counts().items():
        print(f"    {cls:12s} {n:4d} examples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crm", default="CRM_.xlsx", help="Path to CRM_.xlsx (default: CRM_.xlsx in this folder)")
    parser.add_argument("--tone", default="emotion-labelled.csv", help="Path to the tone-labeled csv (default: emotion-labelled.csv in this folder)")
    args = parser.parse_args()

    print("Loading and cleaning data...")
    crm_df = data_prep.load_crm(args.crm)
    tone_df = data_prep.load_tone_labels(args.tone, crm_df)
    print(f"  CRM rows usable for hierarchy training: {len(crm_df)}")
    print(f"  Tone rows usable for emotion training:  {len(tone_df)}")

    embed_model = EmbeddingModel()

    # --- accuracy reports (held-out / cross-validated, not on training data) ---
    evaluate_hierarchy(crm_df, embed_model)
    evaluate_emotion(tone_df, embed_model)

    # --- fit final models on ALL available labeled data and save them ---
    print("\nFitting final models on full data and saving...")
    X_all = embed_model.encode(crm_df["text"].tolist())
    hierarchy_model = HierarchicalClassifier(HIERARCHY_LEVELS).fit(X_all, crm_df)

    X_tone_all = embed_model.encode(tone_df["text"].tolist())
    emotion_model = EmotionClassifier().fit(X_tone_all, tone_df["emotion"].values)

    with open(config.MODEL_DIR / "hierarchy_model.pkl", "wb") as f:
        pickle.dump(hierarchy_model, f)
    with open(config.MODEL_DIR / "emotion_model.pkl", "wb") as f:
        pickle.dump(emotion_model, f)

    print(f"Saved models to {config.MODEL_DIR}")


if __name__ == "__main__":
    main()