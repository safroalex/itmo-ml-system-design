"""Replay DealRocket searches and save the top companies of each search.

Queries come from assistant dialogs exported from the admin panel (``--source admin``) or from
manager links in the request workbook (``--source excel``). The token is read from
DEALROCKET_TOKEN and is never written to disk. Employee records returned by the API are dropped;
only company fields are kept. Results are appended after every query, so a run can be resumed.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://server.dealrocket.ru/api"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://dealrocket.ru",
    "referer": "https://dealrocket.ru/",
}
# Filters that describe the company itself; a query needs at least one to judge relevance.
COMPANY_FILTERS = {"includeAny", "includeAll", "industries", "customIndustries", "okved", "maps"}
# Filters that point to a fixed list of companies: relevance is defined by the list, not text.
LIST_FILTERS = {"companyNameList", "companyIds", "domainsList", "innsList"}
QUERY_FIELDS = [
    "query_id",
    "source",
    "created_at",
    "query_text",
    "context",
    "segment",
    "dealrocket_url",
    "filters_current_labels",
]
PAIR_FIELDS = [
    "query_id",
    "position",
    "company_id",
    "company_name",
    "description",
    "industries",
    "city",
    "website",
    "linked_pages",
    "ai_status",
    "ai_reason",
    "fetched_at",
]


def post(endpoint, body, token, timeout=180):
    request = Request(
        f"{API}/{endpoint}",
        data=json.dumps(body).encode(),
        headers={**HEADERS, "authorization": f"Bearer {token}"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    if payload.get("hasError"):
        raise RuntimeError(f"{endpoint}: {payload.get('errorMessage')}")
    return payload["response"]


def url_filters(url):
    parts = urlsplit(url)
    filters = {"searchId": [parts.path.rsplit("/", 1)[-1]]}
    for key, value in parse_qsl(parts.query):
        if key != "type":
            filters.setdefault(key, []).append(value)
    # Company mode returns companies in ranking order; people mode orders them by employees.
    filters["type"] = ["companies"]
    return filters


def describe_filters(url, dictionaries):
    """Return current labels of the link's filters and the number of IDs missing from them."""
    labels, missing = [], 0
    for key, values in url_filters(url).items():
        if key in ("searchId", "type", "companyIdsExclude"):
            continue
        known = dictionaries.get(key)
        for value in values:
            if known is None:
                labels.append(f"{key}={value}")
            elif value in known:
                labels.append(f"{key}={known[value]}")
            else:
                missing += 1
    return "; ".join(labels), missing


def is_eligible(url, missing):
    keys = set(url_filters(url))
    return bool(keys & COMPANY_FILTERS) and not keys & LIST_FILTERS and missing == 0


def load_admin(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        dialog = json.loads(line)
        previous = []
        for turn in dialog["turns"]:
            url = (turn.get("response") or {}).get("search_url")
            if url and turn["state"] == "completed":
                rows.append(
                    {
                        "query_id": f"a{hashlib.sha1(url.encode()).hexdigest()[:10]}",
                        "source": "admin",
                        "created_at": datetime.fromtimestamp(turn["created_at"], UTC).isoformat(
                            timespec="seconds"
                        ),
                        "query_text": turn["question"].strip(),
                        "context": "\n".join(previous),
                        "segment": turn.get("scenario") or "",
                        "dealrocket_url": url,
                    }
                )
            previous.append(turn["question"].strip())
    return pd.DataFrame(rows)


def load_excel(path):
    classes = pd.read_excel(path, sheet_name="Классификация", header=3)
    samples = pd.read_excel(path, sheet_name="Выборки", header=3)
    if not (classes["Ссылка DealRocket"].values == samples["Ссылка на выборку"].values).all():
        raise ValueError("Sheets 'Классификация' and 'Выборки' are not row-aligned")
    frame = pd.DataFrame(
        {
            "query_id": "e" + classes["№"].astype(str),
            "source": "excel",
            "created_at": classes["Дата (МСК)"].astype(str),
            "query_text": samples["Описание выборки менеджером"].astype(str).str.strip(),
            "context": "",
            "segment": classes["Сегменты компаний"]
            .astype(str)
            .str.strip("|")
            .str.split("|")
            .str[0],
            "dealrocket_url": classes["Ссылка DealRocket"],
            "quality": samples["Качество извлечения"],
        }
    )
    return frame[frame.quality != "Низкое"].drop(columns="quality")


def select_queries(frame, dictionaries, size, seed):
    described = frame["dealrocket_url"].map(lambda url: describe_filters(url, dictionaries))
    frame = frame.assign(
        filters_current_labels=described.str[0],
        eligible=[
            is_eligible(u, m) for u, (_, m) in zip(frame.dealrocket_url, described, strict=True)
        ],
    )
    normalized = frame["query_text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    candidates = (
        frame[frame.eligible]
        .assign(normalized=normalized)
        .drop_duplicates("dealrocket_url")
        .drop_duplicates("normalized")
    )
    shuffled = candidates.sample(frac=1, random_state=seed)
    # Round-robin over segments so that frequent cases do not crowd out rare ones.
    shuffled["rank"] = shuffled.groupby("segment").cumcount()
    selected = shuffled.sort_values(["rank", "segment"], kind="stable").head(size)
    return candidates, selected[QUERY_FIELDS].sort_values("created_at")


def company_row(query_id, position, company, fetched_at):
    assessment = company.get("aiAssessment") or {}
    linked = [
        " — ".join(part for part in (child.get("name"), child.get("description")) if part)
        for child in company.get("childrens") or []
    ]
    return {
        "query_id": query_id,
        "position": position,
        "company_id": company["companyId"],
        "company_name": company.get("name") or "",
        "description": company.get("description") or "",
        "industries": "; ".join(item["name"] for item in company.get("industries") or []),
        "city": company.get("city") or "",
        "website": company.get("website") or "",
        "linked_pages": "\n".join(linked),
        "ai_status": assessment.get("status") or "",
        "ai_reason": assessment.get("reason") or "",
        "fetched_at": fetched_at,
    }


def append_csv(path, rows, fields):
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def fetch_companies(url, token, pause):
    body = [{"filterId": k, "values": v} for k, v in url_filters(url).items()]
    error = None
    for _attempt in range(2):
        try:
            return post("getResultsV2", body, token)["companies"], None
        except Exception as exc:  # noqa: BLE001 - retry once, then report the failure
            error = re.sub(r"\s+", " ", str(exc))
            time.sleep(pause * 5)
    return None, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["admin", "excel"], default="admin")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/dealrocket_admin/conversations.jsonl",
        help="conversations.jsonl for admin, the request workbook for excel",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/dealrocket")
    parser.add_argument("--queries", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between API calls")
    args = parser.parse_args()
    token = os.environ["DEALROCKET_TOKEN"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    filters = post("getFilters", [{"filterId": "searchId", "values": ["s"]}], token)
    (args.output_dir / "filters.json").write_text(
        json.dumps(filters, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dictionaries = {
        item["filterId"]: {value["id"]: value["value"] for value in item["values"]}
        for item in filters
        if item["values"]
    }
    frame = load_admin(args.input) if args.source == "admin" else load_excel(args.input)
    candidates, selected = select_queries(frame, dictionaries, args.queries, args.seed)
    queries_path = args.output_dir / "queries.csv"
    pairs_path = args.output_dir / "pairs.csv"
    failures_path = args.output_dir / "failures.csv"
    selected.to_csv(queries_path, index=False)

    counts_path = args.output_dir / "results_count.csv"
    done = set(pd.read_csv(counts_path)["query_id"]) if counts_path.exists() else set()
    for query in selected.itertuples():
        if query.query_id in done:
            continue
        time.sleep(args.pause)
        fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
        companies, error = fetch_companies(query.dealrocket_url, token, args.pause)
        if companies is None:
            append_csv(
                failures_path, [{"query_id": query.query_id, "error": error}], ["query_id", "error"]
            )
            print(f"{query.query_id}: failed ({error})", flush=True)
            continue
        rows = [
            company_row(query.query_id, position, company, fetched_at)
            for position, company in enumerate(companies[: args.top_k], start=1)
        ]
        append_csv(pairs_path, rows, PAIR_FIELDS)
        append_csv(
            args.output_dir / "results_count.csv",
            [{"query_id": query.query_id, "results_returned": len(companies)}],
            ["query_id", "results_returned"],
        )
        print(f"{query.query_id}: {len(companies)} companies", flush=True)

    pairs = pd.read_csv(pairs_path) if pairs_path.exists() else pd.DataFrame(columns=PAIR_FIELDS)
    manifest = {
        "source": args.source,
        "api": f"{API}/getResultsV2 (type=companies)",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "eligible_queries": len(candidates),
        "queries_selected": len(selected),
        "queries_fetched": int(pairs["query_id"].nunique()),
        "pairs": len(pairs),
        "top_k": args.top_k,
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
