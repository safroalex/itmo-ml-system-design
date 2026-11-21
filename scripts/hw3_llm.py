"""Оценка готовой instruction-модели через локальную Ollama."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from hw3_data import LABELS

PROMPT = Path(__file__).resolve().parents[1] / "prompts/relevance-v1.txt"
API = "http://127.0.0.1:11434/api/"
SCHEMA = dict(
    type="object",
    properties=dict(
        label=dict(type="string", enum=list(LABELS)), reason=dict(type="string", minLength=1)
    ),
    required=["label", "reason"],
    additionalProperties=False,
)


def request(route, body=None, timeout=8):
    req = Request(
        API + route,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with build_opener(ProxyHandler({})).open(req, timeout=timeout) as result:
        return json.load(result)


def model_ids(names):
    models = {m["name"]: m["digest"] for m in request("tags")["models"]}
    unknown = set(names) - models.keys()
    if unknown:
        raise ValueError(f"Model tags are missing in local Ollama: {sorted(unknown)}")
    for name in names:
        info = request("show", {"model": name})
        if (
            "cloud" in name.casefold()
            or info.get("remote_model")
            or info.get("remote_host")
            or not info.get("model_info")
        ):
            raise ValueError(f"{name}: expected a local model with installed weights.")
    return {name: models[name] for name in names}


def payload(row):
    return {k: row[k] for k in ("criteria", "name", "description")}


def examples(train):
    selected = []
    for label in LABELS:
        candidates = [r for r in train if r["label"] == label and (r["description"] or r["name"])]
        if candidates:
            selected.append(min(candidates, key=lambda r: r["id"]))
    return [dict(input=payload(r), label=r["label"]) for r in selected]


def classify(model, prompt, shots, row, deadline):
    start = time.monotonic()
    out = dict(
        id=row["id"],
        label="technical_error",
        reason="",
        attempts=0,
        prompt_tokens=0,
        output_tokens=0,
    )
    if not row["name"] and not row["description"]:
        return {**out, "label": "insufficient_data", "reason": "Пустая карточка.", "seconds": 0.0}
    system = prompt
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            out["error"] = "batch_timeout"
            break
        out["attempts"] += 1
        try:
            response = request(
                "chat",
                dict(
                    model=model,
                    stream=False,
                    format=SCHEMA,
                    messages=[
                        dict(role="system", content=system),
                        dict(
                            role="user",
                            content=json.dumps(
                                dict(examples=shots, company=payload(row)), ensure_ascii=False
                            ),
                        ),
                    ],
                    options=dict(temperature=0, seed=42),
                ),
                timeout=min(8, remaining),
            )
            out["prompt_tokens"] += response.get("prompt_eval_count", 0)
            out["output_tokens"] += response.get("eval_count", 0)
            answer = json.loads(response["message"]["content"])
            if (
                not isinstance(answer, dict)
                or answer.get("label") not in LABELS
                or not isinstance(answer.get("reason"), str)
                or not answer["reason"].strip()
            ):
                raise ValueError("Invalid answer schema")
            if time.monotonic() > deadline:
                raise TimeoutError("batch_timeout")
            out.update(label=answer["label"], reason=answer["reason"].strip())
            out.pop("error", None)
            break
        except (ValueError, TypeError, KeyError, TimeoutError, URLError, OSError) as exc:
            out["error"] = type(exc).__name__
            if attempt == 0 and isinstance(exc, (ValueError, TypeError, KeyError)):
                system = (
                    prompt + "\nПредыдущий ответ нарушил схему. Верни только JSON с label и reason."
                )
    out["seconds"] = round(time.monotonic() - start, 4)
    return out


def run(model, train, rows, rub_hour):
    shots = examples(train)
    prompt = PROMPT.read_text()
    output, batches = [], []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for i in range(0, len(rows), 30):
            chunk = rows[i : i + 30]
            start = time.monotonic()
            futures = [
                pool.submit(classify, model, prompt, shots, row, start + 60) for row in chunk
            ]
            answers = [f.result() for f in futures]
            elapsed = time.monotonic() - start
            output.extend(answers)
            batches.append(
                dict(
                    n=len(chunk),
                    seconds=elapsed,
                    cost_rub=elapsed * rub_hour / 3600,
                    complete=all(r["label"] != "technical_error" for r in answers),
                )
            )
    return output, batches


def resources(batches):
    full = [b for b in batches if b["n"] == 30]
    successful = [b["seconds"] for b in full if b["complete"]]
    return dict(
        full_batches=len(full),
        within_limit=float(np.mean([b["complete"] and b["seconds"] <= 60 for b in full]))
        if full
        else None,
        max_cost_rub=max((b["cost_rub"] for b in full), default=None),
        cost_rub=sum(b["cost_rub"] for b in batches),
        p50_seconds=float(np.median(successful)) if successful else None,
        p95_seconds=float(np.quantile(successful, 0.95)) if successful else None,
    )
