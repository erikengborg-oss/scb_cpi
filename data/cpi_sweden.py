import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import plotly.graph_objects as go
import requests


# API annual January weights for 1980-2026
API_URL = "https://api.scb.se/OV0104/v1/doris/en/ssd/START/PR/PR0101/PR0101A/KPI2020COICOP2M"
QUERY = {
    "query": [
        {"code": "ContentsCode", "selection": {"filter": "item", "values": ["0000080F"]}},
        {"code": "VaruTjanstegrupp", "selection": {"filter": "item", "values": [f"{i:02d}" for i in range(14)]}},
        {"code": "Tid", "selection": {"filter": "item", "values": [f"{y}M01" for y in range(1980, 2027)]}},
    ],
    "response": {"format": "json-stat2"},
}

COICOP_LABELS_EN = {
    "01": "Food and non-alcoholic beverages",
    "02": "Alcoholic beverages and tobacco",
    "03": "Clothing and footwear",
    "04": "Housing",
    "05": "Furnishings and household goods",
    "06": "Health",
    "07": "Transport",
    "08": "Communication",
    "09": "Recreation and culture",
    "10": "Education",
    "11": "Restaurants and hotels",
    "12": "Miscellaneous goods and services",
    "13": "Other",
}


# Convert json-stat2 payload into a flat long table.
def json_stat2_to_df(payload: dict) -> pd.DataFrame:
    dims = payload["id"]
    dim_values: list[list[str]] = []
    dim_labels: dict[str, dict[str, str]] = {}
    for dim in dims:
        category = payload["dimension"][dim]["category"]
        idx = category["index"]
        labels = category.get("label", {})
        dim_labels[dim] = labels if isinstance(labels, dict) else {}
        ordered = [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        dim_values.append(ordered)

    combos = itertools.product(*dim_values)
    rows = []
    for combo, v in zip(combos, payload["value"]):
        rec = {dims[i]: combo[i] for i in range(len(dims))}
        for i, dim in enumerate(dims):
            rec[f"{dim}_label"] = dim_labels.get(dim, {}).get(combo[i], combo[i])
        rec["value"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


# Build a yearly wide table by COICOP code.
def build_wide_table(df: pd.DataFrame, from_year: int, to_year: int | None) -> pd.DataFrame:
    rename_map = {"VaruTjanstegrupp": "code", "Tid": "period"}
    if "VaruTjanstegrupp_label" in df.columns:
        rename_map["VaruTjanstegrupp_label"] = "label_raw"
    out = df.rename(columns=rename_map).copy()

    out["year"] = out["period"].str.extract(r"(\d{4})", expand=False).astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["code"] = out["code"].astype(str).str.extract(r"(\d{1,2})", expand=False).str.zfill(2)

    out = out[out["code"].notna()]
    out = out[out["code"] != "00"]
    out = out[out["year"] >= from_year]
    if to_year is not None:
        out = out[out["year"] <= to_year]

    out = out.dropna(subset=["value"]).sort_values(["code", "year"])
    out = out.groupby(["code", "year"], as_index=False).tail(1)

    labels = (
        out[["code", "label_raw"]]
        .dropna()
        .drop_duplicates("code")
        .set_index("code")["label_raw"]
        .to_dict()
        if "label_raw" in out.columns
        else {}
    )

    wide = out.pivot(index="code", columns="year", values="value").sort_index().reset_index()
    wide["label"] = wide["code"].map(labels).fillna(wide["code"].map(COICOP_LABELS_EN)).fillna(wide["code"])

    year_cols = sorted([c for c in wide.columns if isinstance(c, int)])
    return wide[["code", "label"] + year_cols]


# Save interactive stacked share chart as HTML.
def save_stacked_share_html(wide: pd.DataFrame, out_html: Path) -> None:
    year_cols = [c for c in wide.columns if isinstance(c, int)]
    share_pct = wide[year_cols].div(wide[year_cols].sum(axis=0), axis=1) * 100.0
    colors = plt.cm.tab20(range(len(wide)))

    fig = go.Figure()
    for idx, row in wide.iterrows():
        label = f"{row['code']} {row['label']}"
        vals = share_pct.loc[idx, year_cols].fillna(0.0).values
        color_hex = mcolors.to_hex(colors[idx])
        fig.add_trace(
            go.Bar(
                x=year_cols,
                y=vals,
                name=label,
                marker=dict(color=color_hex),
                hovertemplate=f"Year: %{{x}}<br>Category: {label}<br>Share: %{{y:.2f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Share of the CPI-basket over time",
        barmode="stack",
        xaxis_title="Year",
        yaxis_title="Share (%)",
        yaxis=dict(range=[0, 100]),
        hovermode="closest",
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


# Run pipeline: fetch -> transform -> export CSV + HTML.
def run(out_csv: Path, out_html: Path, from_year: int, to_year: int | None, timeout: int) -> None:
    r = requests.post(API_URL, json=QUERY, timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    raw = json_stat2_to_df(payload)
    wide = build_wide_table(raw, from_year=from_year, to_year=to_year)
    if wide.empty:
        raise ValueError("The table is empty after filtering.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_csv, index=False)
    save_stacked_share_html(wide, out_html)
    print(f"CSV saved: {out_csv}")
    print(f"HTML saved: {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch CPI weights from SCB API and build a yearly wide table.")
    base_dir = Path.home() / "python" / "kpi_scb"
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=base_dir / "data" / "share_cpi_wide.csv",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=base_dir / "figures" / "index.html",
    )
    parser.add_argument("--from-year", type=int, default=1980)
    parser.add_argument("--to-year", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    run(
        out_csv=args.out_csv,
        out_html=args.out_html,
        from_year=args.from_year,
        to_year=args.to_year,
        timeout=args.timeout,
    )
