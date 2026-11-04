"""Exploratory analysis of DealRocket queries and search results.

Reads local exports from data/raw/ (not committed) and writes only aggregates:
figures to reports/figures/ and numbers to reports/eda_summary.json.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
LEGAL_FORM = re.compile(r"\b(?:ООО|ОАО|ЗАО|ПАО|АО|ИП|ГУП|МУП|ФГУП|НКО|АНО)\b")
NON_RUSSIAN = re.compile(r"[іїєґў]", re.IGNORECASE)
WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
STOP = set(
    "и в во на по для с со к ко от до из у о об а но или не нужны нужен нужна найди найти "
    "компании компаний компания все всех которые который которая это их как что".split()
)
ROLE = re.compile(
    r"директор|руковод|владел|собственник|лпр|менеджер|специалист|бухгалтер|логист|сметчик|"
    r"маркетолог|hr|рекрутер|инженер|закупщик|снабж|ceo|cmo|cfo",
    re.IGNORECASE,
)


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 10,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def stems(text):
    return {w[:5] for w in WORD.findall(str(text).lower()) if w not in STOP and len(w) > 2}


def query_groups(pairs):
    """Connected components of queries linked by a shared company."""
    parent = {query: query for query in pairs.query_id.unique()}

    def root(query):
        while parent[query] != query:
            parent[query] = parent[parent[query]]
            query = parent[query]
        return query

    for queries in pairs.groupby("company_id").query_id.unique():
        for query in queries[1:]:
            parent[root(query)] = root(queries[0])
    return pd.Series({query: root(query) for query in parent})


def barh(ax, labels, values, fmt="{:.0f}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=SERIES[0], height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, value in zip(y, values, strict=True):
        ax.text(value, yi, " " + fmt.format(value), va="center", color=INK, fontsize=9)
    ax.set_xlim(0, max(values) * 1.15)


def save(fig, name, out):
    fig.savefig(out / name)
    plt.close(fig)


def dialog_turns(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        dialog = json.loads(line)
        for turn in dialog["turns"]:
            rows.append(
                {
                    "state": turn["state"],
                    "error_code": turn.get("error_code") or "",
                    "scenario": turn.get("scenario") or "",
                    "duration_s": (turn.get("duration_ms") or np.nan) / 1000,
                    "question": turn["question"],
                    "search_url": (turn.get("response") or {}).get("search_url"),
                }
            )
    return pd.DataFrame(rows)


def analyse_queries(turns, queries, out, summary):
    completed = turns[turns.state == "completed"]
    with_url = completed[completed.search_url.notna()]
    funnel = {
        "Ходы ассистента": len(turns),
        "Завершены без ошибки": len(completed),
        "Построен поиск": len(with_url),
        "Уникальные поиски": with_url.search_url.nunique(),
        "Пригодны для оценки": summary["eligible_queries"],
        "В выборке с результатами": len(queries),
    }
    summary["query_funnel"] = funnel
    fig, ax = plt.subplots(figsize=(7, 3.2))
    barh(ax, list(funnel), list(funnel.values()))
    ax.set_title("От обращения к ассистенту до запроса в выборке")
    ax.set_xlabel("Число ходов или запросов")
    save(fig, "01_query_funnel.png", out)

    states = turns.state.value_counts()
    errors = turns[turns.state == "failed"].error_code.replace("", "без кода").value_counts()
    durations = completed.duration_s.dropna()
    summary["assistant"] = {
        "turn_states": states.to_dict(),
        "failed_share": round(float((turns.state == "failed").mean()), 3),
        "top_errors": errors.head(5).to_dict(),
        "duration_s_p50": round(float(durations.median()), 1),
        "duration_s_p95": round(float(durations.quantile(0.95)), 1),
    }

    params = Counter()
    for url in with_url.search_url.drop_duplicates():
        params.update({key for key, _ in parse_qsl(urlsplit(url).query)} - {"type"})
    top = params.most_common(12)
    summary["filters_in_unique_searches"] = dict(top)
    keyword = sum(
        bool({key for key, _ in parse_qsl(urlsplit(url).query)} & {"includeAny", "includeAll"})
        for url in with_url.search_url.drop_duplicates()
    )
    summary["keyword_search_share"] = round(keyword / with_url.search_url.nunique(), 3)
    fig, ax = plt.subplots(figsize=(7, 4))
    total = with_url.search_url.nunique()
    barh(ax, [k for k, _ in top], [100 * v / total for _, v in top], fmt="{:.0f}%")
    ax.set_title("Фильтры, которые ассистент ставит в поиск")
    ax.set_xlabel("Доля уникальных поисков, %")
    save(fig, "02_filters.png", out)

    words = queries.query_text.str.split().str.len()
    role = queries.query_text.str.contains(ROLE)
    summary["queries"] = {
        "n": len(queries),
        "words_median": float(words.median()),
        "words_p90": float(words.quantile(0.9)),
        "share_with_role": round(float(role.mean()), 3),
        "share_with_context": round(float((queries.context.fillna("") != "").mean()), 3),
        "refinements": int(
            (
                queries.query_text.str.contains("Уточнение:|Хочу уточнить")
                | (queries.context.fillna("") != "")
            ).sum()
        ),
    }
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(words, bins=range(1, int(words.max()) + 2), color=SERIES[0], edgecolor=SURFACE)
    ax.set_title("Длина запроса пользователя")
    ax.set_xlabel("Слов в запросе")
    ax.set_ylabel("Запросов")
    save(fig, "03_query_length.png", out)


def analyse_pairs(pairs, queries, out, summary):
    text = (pairs.company_name.fillna("") + " " + pairs.description.fillna("")).str.strip()
    desc = pairs.description.fillna("")
    length = desc.str.len()
    merged = pairs.merge(queries[["query_id", "query_text"]], on="query_id")
    overlap = [
        bool(stems(q) & stems(f"{n} {d}"))
        for q, n, d in zip(
            merged.query_text,
            merged.company_name.fillna(""),
            merged.description.fillna(""),
            strict=True,
        )
    ]
    repeats = pairs.groupby("company_id").query_id.nunique()
    groups = query_groups(pairs)
    sources = pairs.linked_pages.fillna("")
    legal = pairs.company_name.fillna("").str.contains(LEGAL_FORM)
    summary["pairs"] = {
        "n": len(pairs),
        "queries": int(pairs.query_id.nunique()),
        "per_query_min": int(pairs.groupby("query_id").size().min()),
        "unique_companies": int(pairs.company_id.nunique()),
        "companies_in_2plus_queries": int((repeats > 1).sum()),
        "pairs_with_repeated_company": int(pairs.company_id.map(repeats).gt(1).sum()),
        "query_groups": int(groups.nunique()),
        "largest_group_queries": int(groups.value_counts().max()),
        "empty_description_share": round(float((length == 0).mean()), 3),
        "short_description_share": round(float(((length > 0) & (length < 50)).mean()), 3),
        "description_chars_median": float(length[length > 0].median()),
        "description_chars_p95": float(length.quantile(0.95)),
        "website_share": round(float(pairs.website.notna().mean()), 3),
        "city_share": round(float(pairs.city.notna().mean()), 3),
        # The company card merges several sources (social page, legal entity, map card).
        "sources_per_company_median": float(sources.str.count("\n").add(1).median()),
        "empty_description_with_source_text_share": round(
            float(sources[length == 0].str.contains(" — ").mean()), 3
        ),
        "empty_description_share_legal_form": round(float((length[legal] == 0).mean()), 3),
        "empty_description_share_other": round(float((length[~legal] == 0).mean()), 3),
        "legal_form_share_of_empty": round(float(legal[length == 0].mean()), 3),
        "legal_form_in_name_share": round(
            float(pairs.company_name.fillna("").str.contains(LEGAL_FORM).mean()), 3
        ),
        "non_russian_markers_share": round(float(text.str.contains(NON_RUSSIAN).mean()), 3),
        "url_in_description_share": round(
            float(desc.str.contains(r"https?://|www\.|t\.me/").mean()), 3
        ),
        "contact_in_description_share": round(
            float(
                desc.str.contains(
                    r"\+7[\s(-]*\d|8\s?\(?9\d\d|@[a-z0-9-]+\.[a-z]", case=False
                ).mean()
            ),
            3,
        ),
        "duplicate_name_within_query_share": round(
            float(
                pairs.assign(name=pairs.company_name.fillna("").str.lower().str.strip())
                .duplicated(["query_id", "name"], keep=False)
                .mean()
            ),
            3,
        ),
        "lexical_overlap_share": round(float(np.mean(overlap)), 3),
        "ai_status_filled": int(pairs.ai_status.notna().sum()),
        "text_chars_mean": round(float(text.str.len().mean()), 1),
    }

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bins = [0, 1, 50, 100, 200, 400, 800, 1600, max(1601, int(length.max()) + 1)]
    counts = pd.cut(length, bins=bins, right=False).value_counts(sort=False)
    labels = ["пусто", "1–49", "50–99", "100–199", "200–399", "400–799", "800–1599", "1600+"]
    ax.bar(
        labels, 100 * counts.values / len(pairs), color=SERIES[0], edgecolor=SURFACE, linewidth=2
    )
    ax.set_title("Длина описания компании")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Символов в описании")
    ax.set_ylabel("Доля пар, %")
    ax.grid(axis="x", visible=False)
    save(fig, "04_description_length.png", out)

    by_position = pd.DataFrame(
        {"position": pairs.position, "empty": length == 0, "overlap": overlap}
    ).groupby("position")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(
        by_position.empty.mean().index,
        100 * by_position.empty.mean(),
        color=SERIES[0],
        lw=2,
        marker="o",
        ms=4,
        label="пустое описание",
    )
    ax.plot(
        by_position.overlap.mean().index,
        100 * by_position.overlap.mean(),
        color=SERIES[1],
        lw=2,
        marker="o",
        ms=4,
        label="есть слово запроса",
    )
    ax.set_title("Описания по позициям выдачи")
    ax.set_xlabel("Позиция в выдаче")
    ax.set_ylabel("Доля пар, %")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(1, 16))
    ax.legend(frameon=False, loc="center right")
    save(fig, "05_by_position.png", out)

    per_query = pd.DataFrame(
        {"query_id": pairs.query_id, "is_empty": length == 0, "overlap": overlap}
    )
    agg = per_query.groupby("query_id").mean()
    summary["per_query"] = {
        "queries_with_half_empty": int((agg.is_empty >= 0.5).sum()),
        "queries_overlap_below_half": int((agg.overlap < 0.5).sum()),
    }
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(100 * agg.overlap, bins=np.arange(0, 101, 10), color=SERIES[0], edgecolor=SURFACE)
    ax.set_title("Лексическое совпадение по запросам")
    ax.set_xlabel("Доля компаний запроса со словом из запроса, %")
    ax.set_ylabel("Запросов")
    save(fig, "06_overlap_per_query.png", out)

    industries = Counter(
        item.strip()
        for value in pairs.industries.dropna()
        for item in value.split(";")
        if item.strip()
    )
    summary["top_industry_tags"] = dict(industries.most_common(10))


def analyse_excel(excel, dictionaries, out, summary):
    classes = pd.read_excel(excel, sheet_name="Классификация", header=3)
    id_filters = {k for k in dictionaries if k not in ("type",)}
    rows = []
    for url, date in zip(classes["Ссылка DealRocket"], classes["Дата (МСК)"], strict=True):
        pairs = [(k, v) for k, v in parse_qsl(urlsplit(url).query) if k in id_filters]
        if pairs:
            rows.append(
                {"month": str(date)[:7], "missing": any(v not in dictionaries[k] for k, v in pairs)}
            )
    drift = pd.DataFrame(rows).groupby("month").missing.agg(["mean", "size"])
    summary["excel"] = {
        "links": len(classes),
        "links_with_id_filters": len(rows),
        "share_with_missing_ids": round(float(np.mean([r["missing"] for r in rows])), 3),
        "by_month": {
            m: {"share": round(r["mean"], 3), "n": int(r["size"])} for m, r in drift.iterrows()
        },
        "segments_top": classes["Сегменты компаний"]
        .astype(str)
        .str.strip("|")
        .str.split("|")
        .str[0]
        .value_counts()
        .head(8)
        .to_dict(),
    }
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(drift.index, 100 * drift["mean"], color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for x, (share, n) in enumerate(zip(drift["mean"], drift["size"], strict=True)):
        ax.text(
            x,
            100 * share,
            f"{100 * share:.0f}%\nn={n}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    ax.set_title("Ссылки менеджеров с ID, которых нет в текущем справочнике")
    ax.set_xlabel("Месяц создания ссылки")
    ax.set_ylabel("Доля ссылок, %")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    save(fig, "07_excel_id_drift.png", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dialogs", type=Path, default=ROOT / "data/raw/dealrocket_admin/conversations.jsonl"
    )
    parser.add_argument("--results", type=Path, default=ROOT / "data/raw/dealrocket")
    parser.add_argument("--filters", type=Path, default=ROOT / "data/raw/dealrocket/filters.json")
    parser.add_argument("--excel", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()

    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    queries = pd.read_csv(args.results / "queries.csv")
    pairs = pd.read_csv(args.results / "pairs.csv")
    counts_path = args.results / "results_count.csv"
    if counts_path.exists():
        counts = pd.read_csv(counts_path).results_returned
        summary_counts = {
            "zero": int((counts == 0).sum()),
            "below_15": int(((counts > 0) & (counts < 15)).sum()),
            "at_api_cap_300": int((counts >= 300).sum()),
            "n": len(counts),
        }
    else:
        summary_counts = None
    queries = queries[queries.query_id.isin(pairs.query_id)]
    summary = {"snapshot": manifest, "eligible_queries": manifest["eligible_queries"]}
    summary["results_returned"] = summary_counts
    analyse_queries(dialog_turns(args.dialogs), queries, figures, summary)
    analyse_pairs(pairs, queries, figures, summary)
    if args.excel:
        dictionaries = {
            item["filterId"]: {v["id"] for v in item["values"]}
            for item in json.loads(args.filters.read_text(encoding="utf-8"))
            if item["values"]
        }
        analyse_excel(args.excel, dictionaries, figures, summary)
    (args.output / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
