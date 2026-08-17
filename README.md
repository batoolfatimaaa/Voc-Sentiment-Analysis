# VOC Classification Pipeline (Master Changan)

Predicts, from VOC (+ Agent Notes) text: **ComplaintFor -> CPT -> SPG -> CCC**
(chained hierarchy, matching the DMS dropdown behavior) and **Emotion**
(Angry / Frustrated / Calm / Neutral), using `all-MiniLM-L6-v2` sentence
embeddings + `LinearSVC`.

## Files

- `config.py` - paths + category label canonicalization (fixes duplicate
  categories like "After-Sales Service" vs "After-sales service").
- `embeddings.py` - embedding generation with a disk cache so re-running on
  the same complaint text doesn't re-embed it.
- `data_prep.py` - loads/cleans the CRM export and the tone-labeled file.
- `models.py` - the chained hierarchical classifier + the emotion classifier.
- `train.py` - trains everything, prints accuracy, saves models to `models/`.
- `predict.py` - runs the saved models on any complaint file, writes one
  clean combined Excel.

## Setup

```
pip install -r requirements.txt
```

The first run of `train.py` or `predict.py` downloads `all-MiniLM-L6-v2`
from Hugging Face, so you'll need internet access at least once (after
that, the model is cached locally by sentence-transformers, and complaint
text embeddings are cached separately in `embedding_cache/`).

## 1. Train + evaluate

```
python train.py --crm /path/to/CRM_.xlsx --tone /path/to/emotion-labelled.csv
```

This prints:
- accuracy + macro-F1 for ComplaintFor, CPT, SPG, CCC on a held-out 20%
  test split (never seen during training), plus the % of rows where all
  4 levels are predicted correctly at once
- a note on which CCC categories have too few examples to trust
- 5-fold cross-validated accuracy + macro-F1 for the emotion model, plus
  the class distribution (your Calm class only has 13 examples - expect
  that to be the weakest one no matter what)

Then it fits final models on 100% of the labeled data and saves them to
`models/hierarchy_model.pkl` and `models/emotion_model.pkl`.

## 2. Predict on new/full complaint data

```
python predict.py --input /path/to/complaints.xlsx --output /path/to/predictions.xlsx
```

Input file just needs a `voc` column (and `agentnotes` if you want it used
for emotion - optional). Output is one clean Excel:

| id | voc | ComplaintFor | CPT | SPG | CCC | Emotion |

No other columns.

## Re-running on the same data

Embeddings are cached by a hash of the exact text, in
`embedding_cache/all-MiniLM-L6-v2.pkl`. Re-running `train.py` or
`predict.py` on unchanged complaint text will skip re-embedding almost
entirely and just reuse the cache - only genuinely new/edited VOC text
gets embedded.

## Notes on the data as-is

- `complainfor` has two very small classes: **Pre-Sales (12 rows)** and
  **General (11 rows)**. The model will still fit on them but predictions
  for these will not be reliable - worth deciding whether to fold them
  into a nearby category before training seriously.
- Category label cleanup already applied automatically: case/whitespace
  variants are merged by picking whichever spelling is more frequent, and
  a couple of known wording variants (`"Sales Service"` -> `"Sales
  Services"`, `"after sale(s)"` -> `"After-Sales Service"`) are merged via
  `MANUAL_LABEL_ALIASES` in `config.py`. If you spot more duplicate
  categories, add them there.
