"""Изолированный тест контракта. Все строки вымышлены, ответы Ollama имитируются."""

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hw3_data as data
import hw3_experiment as exp
import hw3_llm as llm
from hw3_baseline import scores


class HomeworkTest(unittest.TestCase):
    def test_mask_and_errors(self):
        masked = data.clean("а@пример.рф https://example.org/a +7 (912) 333-44-55 😊")
        self.assertEqual(masked, "<email> <url> <phone>")
        result = scores(list(data.LABELS), ["technical_error", "irrelevant", "insufficient_data"])
        self.assertEqual(result["by_class"]["relevant"]["recall"], 0)
        self.assertEqual(result["relevant_loss"], 1)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        with patch.object(llm, "request", side_effect=ValueError("invalid JSON")) as call:
            ans = llm.classify(
                "fake",
                "rules",
                [],
                dict(id="a", criteria=["стройка"], name="А", description="Б"),
                float("inf"),
            )
            self.assertEqual((ans["label"], call.call_count), ("technical_error", 2))
            ans = llm.classify(
                "fake",
                "rules",
                [],
                dict(id="b", criteria=["стройка"], name="", description=""),
                float("inf"),
            )
            self.assertEqual((ans["label"], call.call_count), ("insufficient_data", 2))

    def test_pipeline(self):
        with TemporaryDirectory() as tmp:
            self.pipeline(Path(tmp))

    def pipeline(self, root):
        queries, pairs, dialogs, labels = [], [], [], {}
        for q in range(24):
            terms = f"отрасль {q}"
            url = f"https://example.org/selection/{q}"
            queries.append(dict(query_id=str(q), query_text=terms, dealrocket_url=url))
            dialogs.append(
                dict(id=q // 2 if q < 2 else q, turns=[dict(response=dict(search_url=url))])
            )
            for k in range(15):
                company = "shared" if k == 0 and q in (2, 3) else f"{q}_{k}"
                name = ("строит", "торгует", "неизвестно")[k % 3]
                pairs.append(
                    dict(query_id=str(q), company_id=company, company_name=name, description=terms)
                )
                labels[data.digest(str(q), company)] = data.LABELS[k % 3]
        data.write_csv(root / "queries.csv", queries, list(queries[0]))
        data.write_csv(root / "pairs.csv", pairs, list(pairs[0]))
        data.write_jsonl(root / "dialogs.jsonl", dialogs)
        form = root / "criteria.csv"
        data.criteria_sheet(root, root / "dialogs.jsonl", form)
        criteria = data.csv_rows(form)
        for r in criteria:
            r.update(criteria=r["query_text"], status="confirmed")
        data.write_csv(root / "confirmed.csv", criteria, list(criteria[0]))
        split = root / "split"
        data.split(root, root / "confirmed.csv", split)
        dev, test = [data.json_rows(split / f"{p}.jsonl") for p in ("dev", "test")]
        self.assertFalse({r["company"] for r in dev} & {r["company"] for r in test})
        assignments = {r["query"]: (r["group"], r.get("fold")) for r in dev + test}
        for a, b in [("0", "1"), ("2", "3")]:
            self.assertEqual(assignments[a], assignments[b])
        for file in ("author.csv", "second.csv"):
            path = split / file
            rows = data.csv_rows(path)
            self.assertNotIn("position", rows[0])
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                for row in rows:
                    writer.writerow({**row, "label": labels[row["id"]], "reason": "учебный пример"})
        for part in ("dev", "test"):
            data.assemble_gold(split, part, None, root / f"{part}-gold.jsonl")
        dev_file = root / "dev-gold.jsonl"
        exp.baseline(SimpleNamespace(dev=dev_file, out=root / "baseline"))
        args = SimpleNamespace(dev=dev_file, out=root / "llm", models=["fixture:1"], rub_hour=1)

        def fake(route, body=None, timeout=8):
            if route == "tags":
                return {"models": [dict(name="fixture:1", digest="fixture-digest")]}
            if route == "show":
                return {"model_info": {"general.architecture": "fixture"}}
            company = json.loads(body["messages"][1]["content"])["company"]
            self.assertEqual(set(company), {"criteria", "name", "description"})
            label = {
                "строит": "relevant",
                "торгует": "irrelevant",
                "неизвестно": "insufficient_data",
            }[company["name"]]
            return {"message": {"content": json.dumps(dict(label=label, reason="имитация"))}}

        with patch.object(llm, "request", side_effect=fake):
            exp.develop_llm(args)
            exp.holdout(
                SimpleNamespace(
                    dev=dev_file,
                    test=root / "test-gold.jsonl",
                    baseline=root / "baseline",
                    llm=root / "llm",
                    out=root / "test-run",
                )
            )
        report = exp.read(root / "test-run/report.json")
        self.assertEqual(report["llm"]["n"], len(test))
        self.assertEqual(report["llm"]["macro_f1"], 1.0)
        self.assertTrue((root / "baseline/baseline.joblib").is_file())


if __name__ == "__main__":
    unittest.main()
