"""Подготовка снимка ДЗ №2, слепой разметки и согласованного эталона."""

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GroupKFold

SEED = 42
LABELS = ("relevant", "irrelevant", "insufficient_data")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.I)
URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+[a-zа-яё]{2,}(?:/\S*)?", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){7,14}\d(?!\w)")
EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]")
ANNOTATION = ["id", "criteria", "name", "description", "label", "reason"]


def clean(value):
    value = unicodedata.normalize("NFKC", value or "")
    value = URL.sub("<url>", EMAIL.sub("<email>", value))
    value = PHONE.sub("<phone>", value)
    return " ".join(re.sub(r"([,;:!?._=*-])\1+", r"\1", EMOJI.sub(" ", value)).split())


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def json_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def index(rows, key):
    mapping = {r[key]: r for r in rows}
    if not rows or "" in mapping or len(mapping) != len(rows):
        raise ValueError(f"Empty input, blank or duplicate {key}.")
    return mapping


def digest(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:20]


def criteria_sheet(snapshot, dialogues, out):
    queries = index(csv_rows(snapshot / "queries.csv"), "query_id")
    ids = {r["query_id"] for r in csv_rows(snapshot / "pairs.csv")}
    if not ids or ids - queries.keys():
        raise ValueError("pairs.csv and queries.csv do not match.")
    by_url = defaultdict(set)
    if dialogues:
        for i, dialog in enumerate(json_rows(dialogues)):
            group = digest("dialog", str(dialog.get("id", i)))
            for turn in dialog["turns"]:
                url = (turn.get("response") or {}).get("search_url")
                if url:
                    by_url[url].add(group)
    rows = []
    for q in queries.values():
        if q["query_id"] in ids:
            rows.append(
                dict(
                    query_id=q["query_id"],
                    query_text=clean(q["query_text"]),
                    dialogue_ids="|".join(sorted(by_url[q["dealrocket_url"]])),
                    related_group="",
                    criteria="",
                    status="",
                )
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows, list(rows[0]))


def build_rows(snapshot, criteria):
    pairs = csv_rows(snapshot / "pairs.csv")
    forms = index(csv_rows(criteria), "query_id")
    queries = index(csv_rows(snapshot / "queries.csv"), "query_id")
    if set(forms) != {p["query_id"] for p in pairs} or set(forms) - queries.keys():
        raise ValueError("Criteria must cover exactly the queries with results.")
    rows = []
    for pair in pairs:
        q, c = pair["query_id"], pair["company_id"]
        form = forms[q]
        if form["status"] == "exclude":
            continue
        if form["status"] != "confirmed" or not form["dialogue_ids"].strip():
            raise ValueError(f"{q}: confirm criteria and dialogue group, or set status=exclude.")
        terms = [clean(v) for v in form["criteria"].split("|") if clean(v)]
        if not terms or not c.strip():
            raise ValueError(f"{q}: missing criteria or company ID.")
        desc = clean(pair["description"])
        rows.append(
            dict(
                id=digest(q, c),
                query=q,
                company=c,
                criteria=terms,
                name=clean(pair["company_name"]),
                description=desc[:1500],
                truncated=len(desc) > 1500,
            )
        )
    index(rows, "id")
    # Union queries sharing a company, conversation, confirmed criteria or manual cluster.
    parent = {}

    def root(q):
        parent.setdefault(q, q)
        while q != parent[q]:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    seen = {}
    for row in rows:
        q = row["query"]
        form = forms[q]
        keys = [
            ("company", row["company"]),
            ("criteria", tuple(sorted(t.casefold() for t in row["criteria"]))),
        ]
        keys += [("dialogue", v.strip()) for v in form["dialogue_ids"].split("|") if v.strip()]
        if form["related_group"].strip():
            keys.append(("related", form["related_group"].strip()))
        for key in keys:
            parent[root(q)] = root(seen.setdefault(key, q))
    members = defaultdict(set)
    for row in rows:
        members[root(row["query"])].add(row["query"])
    for row in rows:
        row["group"] = digest(*sorted(members[root(row["query"])]))
    return sorted(rows, key=lambda r: r["id"])


def allocate(rows):
    groups = defaultdict(set)
    for r in rows:
        groups[r["group"]].add(r["query"])
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    # Subset sum by query count, independent of labels and ranking.
    choices = {0: ()}
    for key in keys:
        for size, previous in list(choices.items()):
            choices.setdefault(size + len(groups[key]), previous + (key,))
    total = sum(map(len, groups.values()))
    valid = [n for n, subset in choices.items() if n and len(keys) - len(subset) >= 5]
    if not valid:
        raise ValueError("Need a separate test group and at least five independent dev groups.")
    held = set(choices[min(valid, key=lambda n: (abs(2 * n - total), n))])
    dev = [r for r in rows if r["group"] not in held]
    test = [r for r in rows if r["group"] in held]
    for fold, (_, valid_idx) in enumerate(
        GroupKFold(5).split(dev, groups=[r["group"] for r in dev])
    ):
        for i in valid_idx:
            dev[i]["fold"] = fold
    return dev, test


def split(snapshot, criteria, out):
    dev, test = allocate(build_rows(snapshot, criteria))
    rng = random.Random(SEED)
    double = rng.sample(dev, min(120, len(dev))) + test

    def blind(items):
        items = list(items)
        rng.shuffle(items)
        return [
            dict(
                id=r["id"],
                criteria=" | ".join(r["criteria"]),
                name=r["name"],
                description=r["description"],
                label="",
                reason="",
            )
            for r in items
        ]

    if out.exists():
        raise ValueError("Split directory exists. Use the saved split or choose a new directory.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / "split"
        stage.mkdir()
        for part, items in (("dev", dev), ("test", test)):
            write_jsonl(stage / f"{part}.jsonl", items)
        write_csv(stage / "author.csv", blind(dev + test), ANNOTATION)
        write_csv(stage / "second.csv", blind(double), ANNOTATION)
        (stage / "double.json").write_text(json.dumps(sorted(r["id"] for r in double)))
        stage.rename(out)
    print(f"dev={len(dev)}, test={len(test)}, double={len(double)}")


def assemble_gold(folder, part, decisions, out):
    rows = json_rows(folder / f"{part}.jsonl")
    author = index(csv_rows(folder / "author.csv"), "id")
    second = index(csv_rows(folder / "second.csv"), "id")
    expected_double = set(json.loads((folder / "double.json").read_text()))
    if set(second) != expected_double:
        raise ValueError("The second annotator's original assignment has changed.")
    resolutions = index(csv_rows(decisions), "id") if decisions else {}
    left, right, conflicts = [], [], []
    for row in rows:
        k = row["id"]
        first = author[k]
        final = first
        raters = [first, second[k]] if k in expected_double else [first]
        if any(r["label"] not in LABELS or not r["reason"].strip() for r in raters):
            raise ValueError(f"{k}: missing human label or reason in {part}.")
        if len(raters) == 2:
            left.append(first["label"])
            right.append(second[k]["label"])
            if left[-1] != right[-1]:
                final = resolutions.get(k, {})
                if final.get("label") not in LABELS or not final.get("reason", "").strip():
                    conflicts.append(k)
        row["label"] = final.get("label")
    if left:
        kappa = float(cohen_kappa_score(left, right, labels=LABELS))
        print(
            json.dumps(
                dict(
                    n=len(left),
                    labels=LABELS,
                    agreement=float(np.mean(np.array(left) == right)),
                    kappa=kappa if np.isfinite(kappa) else None,
                    matrix=confusion_matrix(left, right, labels=LABELS).tolist(),
                )
            )
        )
        if not np.isfinite(kappa) or kappa < 0.6:
            raise ValueError(
                f"{part}: insufficient agreement. Review the rules or collect a fresh test set."
            )
    if conflicts:
        raise ValueError("Resolve disagreements (id,label,reason): " + ", ".join(conflicts))
    write_jsonl(out, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="cmd", required=True)
    form = cmds.add_parser("criteria")
    form.add_argument("--snapshot", type=Path, required=True)
    form.add_argument("--dialogues", type=Path)
    form.add_argument("--out", type=Path, required=True)
    grouping = cmds.add_parser("split")
    grouping.add_argument("--snapshot", type=Path, required=True)
    grouping.add_argument("--criteria", type=Path, required=True)
    grouping.add_argument("--out", type=Path, required=True)
    gold = cmds.add_parser("gold")
    gold.add_argument("--split", type=Path, required=True)
    gold.add_argument("--part", choices=("dev", "test"), required=True)
    gold.add_argument("--decisions", type=Path)
    gold.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "criteria":
        criteria_sheet(args.snapshot, args.dialogues, args.out)
    elif args.cmd == "split":
        split(args.snapshot, args.criteria, args.out)
    else:
        assemble_gold(args.split, args.part, args.decisions, args.out)


if __name__ == "__main__":
    main()
