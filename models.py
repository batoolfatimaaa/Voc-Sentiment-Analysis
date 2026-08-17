"""
Two model types:

1. HierarchicalClassifier - a chain of node models mirroring the DMS dropdown
   hierarchy: ComplaintFor -> CPT -> SPG -> CCC. Each level is trained only on
   rows belonging to its specific parent path.

   Each node now picks its OWN classifier based on how much data it actually
   has (this is the main accuracy fix over the original all-LinearSVC version):
     - only one class ever seen at this node          -> constant label
     - smallest class has >= MIN_SAMPLES_FOR_SVC rows  -> LinearSVC
     - smallest class has fewer rows than that         -> nearest-centroid
       (cosine similarity to each class's average embedding - this degrades
       far more gracefully than an SVM margin when a class has <10-20 examples,
       since it doesn't need enough points to estimate a boundary, just an
       average)

   Every node also exposes a confidence score per prediction (not a
   calibrated probability, but consistent enough within one node to rank or
   threshold on), so predict.py can flag low-confidence full paths for
   manual review instead of silently guessing.

2. EmotionClassifier - LinearSVC wrapped in CalibratedClassifierCV, so it
   exposes real predict_proba (plain LinearSVC doesn't), used the same way
   to flag low-confidence emotion calls.
"""
from collections import defaultdict

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

import config


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax - turns raw decision_function margins / cosine
    similarities into a comparable 0-1 'confidence' number per node."""
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


class _CentroidModel:
    """
    Nearest-class-mean ("prototype") classifier: for each class, store the
    average L2-normalized embedding of its training examples. Predict by
    cosine similarity to each class centroid, pick the highest.

    This is the approach from the few-shot notes: it needs almost no data
    per class to behave sensibly, unlike LinearSVC which needs enough points
    to even locate a margin.
    """

    def __init__(self):
        self.classes_ = None
        self.centroids_ = None  # (n_classes, dim), L2-normalized

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xn = self._normalize(X)
        classes = np.unique(y)
        centroids = np.vstack([
            self._normalize(Xn[y == c].mean(axis=0, keepdims=True))
            for c in classes
        ])
        self.classes_ = classes
        self.centroids_ = centroids
        return self

    def _similarities(self, X: np.ndarray) -> np.ndarray:
        Xn = self._normalize(X)
        return Xn @ self.centroids_.T  # (n_samples, n_classes) cosine similarity

    def predict(self, X: np.ndarray) -> np.ndarray:
        sims = self._similarities(X)
        return self.classes_[np.argmax(sims, axis=1)]

    def confidence(self, X: np.ndarray) -> np.ndarray:
        # Raw top cosine similarity, NOT softmax. A softmax over many
        # roughly-equal classes crushes every score toward 1/n_classes
        # regardless of whether the top match is actually a good one - with
        # 20+ classes at some CCC nodes this meant confidence almost never
        # exceeded ~0.4 even for a correct, confident prediction, which made
        # any review threshold above ~0.4 useless (0% coverage). Raw
        # similarity doesn't have that problem: it reflects how close the
        # match actually is, independent of how many other classes exist.
        sims = self._similarities(X)
        return sims.max(axis=1)

    def topk(self, X: np.ndarray, k: int = 3):
        """Returns, per row, a list of up to k (label, similarity) pairs
        sorted best-first."""
        sims = self._similarities(X)
        k_eff = min(k, sims.shape[1])
        order = np.argsort(-sims, axis=1)[:, :k_eff]
        return [
            [(self.classes_[j], float(sims[i, j])) for j in order[i]]
            for i in range(len(X))
        ]


class _NodeModel:
    """
    One node in the hierarchy: predicts a child label given an embedding.

    Four possible outcomes when fitting, decided by class sizes:
      - "constant" - only one class ever seen -> always predict it
      - "svc"      - every class has >= MIN_SAMPLES_FOR_SVC rows -> plain LinearSVC
      - "centroid" - every class is small -> plain nearest-centroid
      - "hybrid"   - MIX of big and small classes at the same node. This is the
                     fix for "one rare class drags the whole node down": SVC is
                     trained ONLY on the well-populated classes (so those keep
                     full accuracy, unaffected by the rare one), and a centroid
                     is trained on ALL classes including the rare ones. At
                     prediction time, a rare class only wins if the point's
                     centroid-similarity to it is both reasonably high AND
                     higher than its similarity to whatever class SVC picked -
                     i.e. the point genuinely looks more like the rare class,
                     not just "SVC was unsure." Otherwise SVC's answer stands.
    """

    def __init__(self):
        self.constant_label = None
        self.clf = None
        self.centroid = None
        self.small_classes = None
        self.kind = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 1:
            self.constant_label = unique[0]
            self.kind = "constant"
            return self

        big_mask = counts >= config.MIN_SAMPLES_FOR_SVC
        big_classes = unique[big_mask]
        small_classes = unique[~big_mask]

        if len(small_classes) == 0:
            self.clf = LinearSVC(class_weight="balanced", max_iter=5000)
            self.clf.fit(X, y)
            self.kind = "svc"
        elif len(big_classes) < 2:
            # not enough well-populated classes to bother splitting - same as before
            self.centroid = _CentroidModel().fit(X, y)
            self.kind = "centroid"
        else:
            mask = np.isin(y, big_classes)
            self.clf = LinearSVC(class_weight="balanced", max_iter=5000)
            self.clf.fit(X[mask], y[mask])
            self.centroid = _CentroidModel().fit(X, y)  # all classes, incl. rare
            self.small_classes = set(small_classes.tolist())
            self.kind = "hybrid"
        return self

    def _svc_confidence(self, X: np.ndarray) -> np.ndarray:
        scores = self.clf.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return _softmax(scores).max(axis=1)

    def _hybrid_decide(self, X: np.ndarray):
        """Returns (labels, confidences) for a hybrid node - see class docstring."""
        svc_labels = self.clf.predict(X)
        svc_confs = self._svc_confidence(X)
        sims = self.centroid._similarities(X)
        class_index = {c: i for i, c in enumerate(self.centroid.classes_)}

        out_labels = np.empty(len(X), dtype=object)
        out_confs = np.empty(len(X))
        for i in range(len(X)):
            best_small_label, best_small_sim = None, -1.0
            for lbl in self.small_classes:
                sim = sims[i, class_index[lbl]]
                if sim > best_small_sim:
                    best_small_sim, best_small_label = sim, lbl
            svc_label_sim = sims[i, class_index[svc_labels[i]]]
            if (best_small_label is not None
                    and best_small_sim >= config.RARE_CLASS_OVERRIDE_THRESHOLD
                    and best_small_sim > svc_label_sim):
                out_labels[i] = best_small_label
                out_confs[i] = best_small_sim
            else:
                out_labels[i] = svc_labels[i]
                out_confs[i] = svc_confs[i]
        return out_labels, out_confs

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "constant":
            return np.array([self.constant_label] * len(X))
        if self.kind == "centroid":
            return self.centroid.predict(X)
        if self.kind == "svc":
            return self.clf.predict(X)
        labels, _ = self._hybrid_decide(X)
        return labels

    def confidence(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "constant":
            return np.ones(len(X))
        if self.kind == "centroid":
            return self.centroid.confidence(X)
        if self.kind == "svc":
            return self._svc_confidence(X)
        _, confs = self._hybrid_decide(X)
        return confs

    def topk(self, X: np.ndarray, k: int = 3):
        n = len(X)
        if self.kind == "constant":
            return [[(self.constant_label, 1.0)] for _ in range(n)]
        if self.kind == "centroid":
            return self.centroid.topk(X, k=k)
        if self.kind == "svc":
            scores = self.clf.decision_function(X)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            probs = _softmax(scores)
            classes = self.clf.classes_
            k_eff = min(k, probs.shape[1])
            order = np.argsort(-probs, axis=1)[:, :k_eff]
            return [
                [(classes[j], float(probs[i, j])) for j in order[i]]
                for i in range(n)
            ]
        # hybrid: merge SVC's ranked big-class list with centroid-ranked small classes
        scores = self.clf.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        svc_probs = _softmax(scores)
        svc_classes = self.clf.classes_
        sims = self.centroid._similarities(X)
        class_index = {c: i for i, c in enumerate(self.centroid.classes_)}
        out = []
        for i in range(n):
            big_ranked = list(zip(svc_classes, svc_probs[i]))
            small_ranked = [(lbl, sims[i, class_index[lbl]]) for lbl in self.small_classes]
            combined = sorted(big_ranked + small_ranked, key=lambda t: -t[1])
            out.append([(lbl, float(score)) for lbl, score in combined[:k]])
        return out


class HierarchicalClassifier:
    """
    levels: list of column names in parent->child order,
    e.g. ["complainfor", "cpt", "spg", "ccc"]
    """

    def __init__(self, levels):
        self.levels = levels
        # nodes[level_index][parent_path_tuple] -> _NodeModel
        self.nodes = [dict() for _ in levels]

    def fit(self, X: np.ndarray, df):
        n = len(df)
        parent_paths = [tuple() for _ in range(n)]

        for level_idx, level_name in enumerate(self.levels):
            groups = defaultdict(list)
            for i in range(n):
                groups[parent_paths[i]].append(i)

            for path, idxs in groups.items():
                idxs = np.array(idxs)
                node = _NodeModel().fit(X[idxs], df[level_name].values[idxs])
                self.nodes[level_idx][path] = node

            # extend each row's parent path with its true label for the next level
            labels = df[level_name].values
            parent_paths = [parent_paths[i] + (labels[i],) for i in range(n)]

        return self

    def predict(self, X: np.ndarray, return_confidence: bool = False,
                return_topk: bool = False, k: int = 3):
        """
        Returns {level_name: np.array of predictions}, chained on predicted
        parents (top-1 prediction at each level is what feeds the next level,
        regardless of return_topk).

        If return_confidence=True, also returns:
          - {level_name: np.array of per-level confidences}
          - path_confidence: np.array = min confidence across all levels for
            that row (the weakest link in the chain - use this to decide
            which rows need a human to check them).

        If return_topk=True (implies return_confidence), also returns:
          - {level_name: list of per-row [(label, score), ...] top-k lists}
            Most useful for cpt/spg/ccc, where a single hard guess is least
            reliable - a ranked shortlist is a more realistic thing to show
            a human than one forced prediction.
        """
        n = len(X)
        parent_paths = [tuple() for _ in range(n)]
        predictions = {level: [None] * n for level in self.levels}
        confidences = {level: np.ones(n) for level in self.levels}
        topk_out = {level: [None] * n for level in self.levels}

        for level_idx, level_name in enumerate(self.levels):
            groups = defaultdict(list)
            for i in range(n):
                groups[parent_paths[i]].append(i)

            level_preds = [None] * n
            level_conf = np.ones(n)
            for path, idxs in groups.items():
                idxs = np.array(idxs)
                node = self.nodes[level_idx].get(path)
                if node is None:
                    # parent path never seen in training - fall back to the
                    # most common child at this level, and flag it as
                    # low-confidence since we're genuinely guessing here
                    fallback = self._most_common_label(level_idx)
                    preds = np.array([fallback] * len(idxs))
                    conf = np.zeros(len(idxs))
                    topk_lists = [[(fallback, 0.0)]] * len(idxs)
                else:
                    preds = node.predict(X[idxs])
                    conf = node.confidence(X[idxs])
                    topk_lists = node.topk(X[idxs], k=k) if return_topk else None
                for j, i in zip(idxs, range(len(idxs))):
                    level_preds[j] = preds[i]
                    level_conf[j] = conf[i]
                    if return_topk:
                        topk_out[level_name][j] = topk_lists[i]

            predictions[level_name] = np.array(level_preds)
            confidences[level_name] = level_conf
            parent_paths = [parent_paths[i] + (level_preds[i],) for i in range(n)]

        if not return_confidence and not return_topk:
            return predictions

        path_confidence = np.min(np.vstack([confidences[l] for l in self.levels]), axis=0)

        if return_topk:
            return predictions, confidences, path_confidence, topk_out
        return predictions, confidences, path_confidence

    def _most_common_label(self, level_idx):
        from collections import Counter
        counts = Counter()
        for node in self.nodes[level_idx].values():
            if node.kind == "constant":
                counts[node.constant_label] += 1
            elif node.clf is not None:
                for c in node.clf.classes_:
                    counts[c] += 1
        return counts.most_common(1)[0][0] if counts else None


class EmotionClassifier:
    """
    LinearSVC wrapped in CalibratedClassifierCV so we get real predict_proba
    (plain LinearSVC doesn't expose probabilities) - needed to flag
    low-confidence emotion calls for review, same as the hierarchy.
    """

    def __init__(self):
        self.clf = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000), cv=3
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def confidence(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X).max(axis=1)