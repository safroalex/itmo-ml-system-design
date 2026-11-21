"""Признаки пары, групповая валидация и метрики трёх классов."""

import itertools
import re
from collections import Counter

import numpy as np
from hw3_data import LABELS, SEED
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.preprocessing import normalize

STOP = set("а в во за и из или к на не о от по под при с со у".split())


def words(text):
    return set(re.findall(r"[а-яёa-z0-9]+", text.casefold())) - STOP


def parts(row):
    return " ".join(row["criteria"]), (row["name"] + " " + row["description"]).strip()


class PairFeatures:
    def __init__(self, min_df, word_max, char_max):
        self.settings = [
            dict(analyzer="word", ngram_range=(1, word_max)),
            dict(analyzer="char_wb", ngram_range=(3, char_max)),
        ]
        self.min_df = min_df

    def fit(self, rows):
        texts = [value for row in rows for value in parts(row)]
        self.vectors = []
        for settings in self.settings:
            v = TfidfVectorizer(min_df=self.min_df, sublinear_tf=True, **settings)
            try:
                v.fit(texts)
            except ValueError as exc:
                if not any(
                    msg in str(exc)
                    for msg in ("empty vocabulary", "no terms remain", "max_df corresponds")
                ):
                    raise
                continue
            self.vectors.append(v)
        return self

    def transform(self, rows):
        criteria, companies = zip(*(parts(r) for r in rows), strict=True)
        blocks = []
        if self.vectors:
            a = normalize(hstack([v.transform(criteria) for v in self.vectors], format="csr"))
            b = normalize(hstack([v.transform(companies) for v in self.vectors], format="csr"))
            product = a.multiply(b)
            blocks.extend((product, abs(a - b), csr_matrix(product.sum(axis=1))))
        scalar = []
        for r, text in zip(rows, companies, strict=True):
            tokens = words(text)
            n = sum(bool(words(term) & tokens) for term in r["criteria"])
            scalar.append([np.log1p(len(r["description"])), n, n / len(r["criteria"])])
        return hstack([*blocks, csr_matrix(scalar)], format="csr")


def scores(truth, pred):
    errors = "technical_error"
    labels = [*LABELS, errors]
    matrix = confusion_matrix(truth, pred, labels=labels)[:3]
    out = matrix_scores(matrix)
    return {**out, "labels": list(LABELS), "prediction_columns": labels, "matrix": matrix.tolist()}


def matrix_scores(matrix):
    support = matrix.sum(axis=1)
    columns = matrix[:, :3].sum(axis=0)
    tp = np.diag(matrix[:, :3])
    precision = tp / np.maximum(columns, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * tp / np.maximum(support + columns, 1)
    return dict(
        n=int(support.sum()),
        macro_f1=float(f1.mean()),
        by_class={
            k: dict(precision=float(p), recall=float(r), f1=float(f), n=int(n))
            for k, p, r, f, n in zip(LABELS, precision, recall, f1, support, strict=True)
        },
        relevant_loss=float((matrix[0, 1] + matrix[0, 3]) / support[0]) if support[0] else None,
        coverage=float(matrix[:, :3].sum() / max(support.sum(), 1)),
    )


def fold_rows(rows):
    groups = {}
    for r in rows:
        if r["fold"] not in range(5) or groups.setdefault(r["group"], r["fold"]) != r["fold"]:
            raise ValueError("Invalid fold assignment or group leakage.")
    if set(groups.values()) != set(range(5)):
        raise ValueError("Expected five nonempty validation folds.")
    for f in range(5):
        train = [r for r in rows if r["fold"] != f]
        valid = [r for r in rows if r["fold"] == f]
        if {r["label"] for r in train} != set(LABELS):
            raise ValueError(f"Fold {f} lacks a training class. Collect more labeled groups.")
        yield train, valid


def classifier(c):
    return LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=SEED)


def predictions(clf, features, rows):
    pred = clf.predict(features).tolist()
    return [y if parts(r)[1] else "insufficient_data" for r, y in zip(rows, pred, strict=True)]


def fit_baseline(rows):
    configs = list(itertools.product((1, 2), ((1, 4), (2, 5)), (0.1, 1, 10)))
    pred = {cfg: {} for cfg in configs}
    logs = {cfg: 0.0 for cfg in configs}
    majority = {}
    for train, valid in fold_rows(rows):
        y = [r["label"] for r in train]
        yv = [r["label"] for r in valid]
        majority.update({r["id"]: Counter(y).most_common(1)[0][0] for r in valid})
        for min_df, (ww, cw) in itertools.product((1, 2), ((1, 4), (2, 5))):
            vec = PairFeatures(min_df, ww, cw).fit(train)
            x, v = vec.transform(train), vec.transform(valid)
            for c in (0.1, 1, 10):
                cfg = (min_df, (ww, cw), c)
                model = classifier(c).fit(x, y)
                pred[cfg].update(
                    zip((r["id"] for r in valid), predictions(model, v, valid), strict=True)
                )
                logs[cfg] += log_loss(
                    yv, model.predict_proba(v), labels=model.classes_, normalize=False
                )
    truth = [r["label"] for r in rows]

    def aligned(mapping):
        return [mapping[r["id"]] for r in rows]

    metrics = {cfg: scores(truth, aligned(pred[cfg])) for cfg in configs}
    best = max(configs, key=lambda cfg: metrics[cfg]["macro_f1"])
    min_df, (ww, cw), c = best
    vec = PairFeatures(min_df, ww, cw).fit(rows)
    model = classifier(c).fit(vec.transform(rows), truth)
    report = dict(
        selected=dict(min_df=min_df, word_max=ww, char_max=cw, C=c),
        oof=metrics[best],
        majority_oof=scores(truth, aligned(majority)),
        grid=[
            dict(
                min_df=n,
                word_max=w,
                char_max=h,
                C=c,
                macro_f1=metrics[(n, (w, h), c)]["macro_f1"],
                log_loss=logs[(n, (w, h), c)] / len(rows),
            )
            for n, (w, h), c in configs
        ],
    )
    return (vec, model, Counter(truth).most_common(1)[0][0]), report, aligned(pred[best])


def paired_ci(rows, baseline, llm, iterations=2000):
    groups = {}
    for row, base_label, llm_label in zip(rows, baseline, llm, strict=True):
        groups.setdefault(row["group"], []).append((row["label"], base_label, llm_label))
    matrices = []
    for group in groups.values():
        truth, base_labels, llm_labels = zip(*group, strict=True)
        matrices.append([scores(truth, base_labels)["matrix"], scores(truth, llm_labels)["matrix"]])
    matrices = np.asarray(matrices)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(iterations):
        base_matrix, llm_matrix = matrices[rng.integers(0, len(groups), len(groups))].sum(axis=0)
        base_score = matrix_scores(base_matrix)["macro_f1"]
        llm_score = matrix_scores(llm_matrix)["macro_f1"]
        samples.append((base_score, llm_score, llm_score - base_score))
    intervals = np.quantile(samples, [0.025, 0.975], axis=0).T.tolist()
    return dict(zip(("baseline", "llm", "llm_minus_baseline"), intervals, strict=True))
