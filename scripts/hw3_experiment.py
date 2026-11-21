"""ДЗ №3. Запуск моделей на разработке и сравнение на закрытом test."""

import argparse
import hashlib
import json
import math
import os
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import hw3_llm as llm
import joblib
from hw3_baseline import fit_baseline, fold_rows, paired_ci, predictions, scores
from hw3_data import LABELS, index, json_rows, write_jsonl


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(dev):
    scripts = Path(__file__).resolve().parent
    return dict(
        dev_sha256=sha(dev),
        seed=42,
        python=sys.version.split()[0],
        packages={p: version(p) for p in ("scikit-learn", "scipy", "numpy", "joblib")},
        code={
            name: sha(scripts / name)
            for name in ("hw3_data.py", "hw3_baseline.py", "hw3_llm.py", "hw3_experiment.py")
        },
    )


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text())


def summary(rows, pred):
    index(rows, "id")
    if len(rows) != len(pred):
        raise ValueError("Different numbers of labels and predictions.")
    if any(r["label"] not in LABELS for r in rows) or any(
        p not in (*LABELS, "technical_error") for p in pred
    ):
        raise ValueError("Unknown class label.")
    return scores([r["label"] for r in rows], pred)


def fresh(out, action):
    if out.exists():
        raise ValueError(f"{out} exists. Keep the previous result and choose a new directory.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as stage:
        result = Path(stage) / "result"
        result.mkdir()
        action(result)
        result.rename(out)


def baseline(args):
    rows = json_rows(args.dev)
    index(rows, "id")

    def calculate(out):
        model, report, oof = fit_baseline(rows)
        joblib.dump(model, out / "baseline.joblib")
        save(out / "report.json", dict(stage="development", **meta(args.dev), **report))
        write_jsonl(
            out / "oof.jsonl",
            [dict(id=r["id"], prediction=p) for r, p in zip(rows, oof, strict=True)],
        )

    fresh(args.out, calculate)


def resource_budget(rows, rate, cap):
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("Specify a nonnegative hourly local compute rate (RUB).")
    # Upper estimate at the specified batch deadline. A local run uses no external API.
    cost = sum(math.ceil(len(fold) / 30) for fold in rows) * 60 * rate / 3600
    if cost > cap:
        raise ValueError(f"Estimated maximum {cost:.2f} RUB exceeds the {cap} RUB stage budget.")


def develop_llm(args):
    rows = json_rows(args.dev)
    index(rows, "id")
    folds = list(fold_rows(rows))
    names = args.models
    if len(set(names)) != len(names) or not 1 <= len(names) <= 2:
        raise ValueError("Fix one or two distinct model tags before the run.")
    ids = llm.model_ids(names)
    if len(set(ids.values())) != len(ids):
        raise ValueError("Both tags point to the same model.")
    spec = dict(
        **meta(args.dev), models=ids, order=names, prompt=sha(llm.PROMPT), rub_hour=args.rub_hour
    )
    resource_budget([valid for _, valid in folds] * len(names), args.rub_hour, 600)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    config = out / "config.json"
    if config.exists() and read(config) != spec:
        raise ValueError("Saved configuration differs. Use a new output directory.")
    save(config, spec)
    candidates = []
    for m, model in enumerate(names):
        oof = {}
        batches = []
        for f, (train, valid) in enumerate(folds):
            cache = out / f"model-{m}-fold-{f}.json"
            if not cache.exists():
                answers, stats = llm.run(model, train, valid, args.rub_hour)
                save(cache, dict(answers=answers, batches=stats))
            item = read(cache)
            actual = index(item["answers"], "id")
            if set(actual) != {r["id"] for r in valid}:
                raise ValueError("Prediction IDs do not match the saved fold.")
            oof.update(actual)
            batches.extend(item["batches"])
        labels = [oof[r["id"]]["label"] for r in rows]
        metrics = summary(rows, labels)
        res = llm.resources(batches)
        eligible = (
            res["full_batches"] > 0 and res["within_limit"] >= 0.95 and res["max_cost_rub"] <= 10
        )
        candidates.append(dict(model=model, metrics=metrics, resources=res, eligible=eligible))
    best = max(
        (c for c in candidates if c["eligible"]),
        key=lambda c: (c["metrics"]["macro_f1"], -c["resources"]["cost_rub"]),
        default=None,
    )
    save(
        out / "report.json",
        dict(
            stage="development",
            **spec,
            candidates=candidates,
            selected=best["model"] if best else None,
        ),
    )


def holdout(args):
    dev, test = json_rows(args.dev), json_rows(args.test)
    expected = meta(args.dev)
    bspec = read(args.baseline / "report.json")
    lspec = read(args.llm / "report.json")
    if any(s.get(k) != v for s in (bspec, lspec) for k, v in expected.items()):
        raise ValueError("Changed development data, code or environment.")
    model = lspec["selected"]
    if not model:
        raise ValueError("No LLM candidate met the development resource limits.")
    if (
        llm.model_ids([model])[model] != lspec["models"][model]
        or sha(llm.PROMPT) != lspec["prompt"]
    ):
        raise ValueError("Changed model or prompt.")
    index(dev, "id")
    index(test, "id")
    if any("fold" in r for r in test) or any(
        {r[k] for r in dev} & {r[k] for r in test} for k in ("id", "group", "company", "query")
    ):
        raise ValueError("Test overlaps development.")
    if {r["label"] for r in test} != set(LABELS):
        raise ValueError("The holdout must contain all three classes.")
    resource_budget([test], lspec["rub_hour"], 600)

    def calculate(out):
        vec, clf, majority = joblib.load(args.baseline / "baseline.joblib")
        b = predictions(clf, vec.transform(test), test)
        answers, batches = llm.run(model, dev, test, lspec["rub_hour"])
        pred = [r["label"] for r in answers]
        for row, result, base in zip(test, answers, b, strict=True):
            if row["id"] != result["id"]:
                raise ValueError("Prediction order changed.")
            result["baseline"] = base
        write_jsonl(out / "predictions.jsonl", answers)
        save(
            out / "report.json",
            dict(
                stage="test",
                **expected,
                test_sha256=sha(args.test),
                model=model,
                model_digest=lspec["models"][model],
                prompt=lspec["prompt"],
                majority=summary(test, [majority] * len(test)),
                baseline=summary(test, b),
                llm=summary(test, pred),
                macro_f1_ci95=paired_ci(test, b, pred),
                resources=llm.resources(batches),
                by_group={
                    g: summary(
                        [r for r in test if r["group"] == g],
                        [y for r, y in zip(test, pred, strict=True) if r["group"] == g],
                    )
                    for g in sorted({r["group"] for r in test})
                },
            ),
        )

    fresh(args.out, calculate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="cmd", required=True)
    for name, func in [("baseline", baseline), ("llm-dev", develop_llm), ("test", holdout)]:
        c = cmds.add_parser(name)
        c.set_defaults(func=func)
        c.add_argument("--dev", type=Path, required=True)
        c.add_argument("--out", type=Path, required=True)
        if name == "llm-dev":
            c.add_argument("--models", nargs="+", required=True)
            c.add_argument("--rub-hour", type=float, required=True)
        if name == "test":
            for arg in ("test", "baseline", "llm"):
                c.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
