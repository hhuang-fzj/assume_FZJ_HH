"""
WP9 (minimal): collect the WP8 runs into one results table and three figures.

Run first (from the repository root, .venv active):
    python examples/inputs/LocalRetailer_test/course/wp8_run_assume.py
    python examples/inputs/LocalRetailer_test/course/wp8_run_assume.py --s-hat-feedback
    python examples/inputs/LocalRetailer_test/course/wp8_run_assume.py --alpha 0
    python examples/inputs/LocalRetailer_test/course/wp8_run_assume.py --alpha 0.5
    python examples/inputs/LocalRetailer_test/course/wp8_run_assume.py --alpha 1
Then:
    python examples/inputs/LocalRetailer_test/course/wp9_results.py

Reads:   course/results/<run>_seed1/summary.csv and settlement.csv   (WP8)
Writes:  course/results/wp9_results_table.csv
         course/results/wp9_fig1_total_cost.png
         course/results/wp9_fig2_cost_components.png
         course/results/wp9_fig3_one_day.png

Nothing new is computed here except sums, shares and differences of WP8 outputs.
Units: volumes MWh per week (+ buy / - sell), costs EUR per week (+ pay / - earn).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COURSE_PATH = Path(__file__).resolve().parent
RESULTS = COURSE_PATH / "results"
SEED = 1

# run folder -> label in table and figures (order = order in table and figures)
RUNS = {
    "fixed_alpha_0_00": "fixed α = 0 (all WM)",
    "fixed_alpha_0_50": "fixed α = 0.5",
    "fixed_alpha_1_00": "fixed α = 1 (all LEM)",
    "optimised_option1": "optimised α, option 1",
    "optimised_option2": "optimised α, option 2",
}
DAY = "2016-06-21"  # day shown in figure 3 (any day of the week works)

# categorical colours: reference palette of the dataviz guideline, slots 1-3 (light)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"


def load_runs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Summary rows and per-period settlements of all runs."""
    summaries, settlements = [], {}
    for folder, label in RUNS.items():
        path = RESULTS / f"{folder}_seed{SEED}"
        if not (path / "summary.csv").exists():
            raise FileNotFoundError(f"{path} missing: run wp8_run_assume.py for it first")
        row = pd.read_csv(path / "summary.csv").iloc[0]
        row["label"] = label
        summaries.append(row)
        settlements[label] = pd.read_csv(path / "settlement.csv", index_col=0,
                                         parse_dates=True)
    return pd.DataFrame(summaries).set_index("label"), settlements


def results_table(summary: pd.DataFrame, settlements: dict) -> pd.DataFrame:
    """One row per run: volumes, costs, and two derived shares."""
    table = summary[["q_lem_bid", "q_lem_accepted", "q_wm_accepted", "imbalance",
                     "cost_lem", "cost_wm", "cost_imbalance", "cost_total"]].copy()
    table["expected_cost_total"] = summary.get("expected_cost_total")
    # LEM acceptance: accepted / offered, sell periods only (buy: no LEM supply at night)
    shares = {}
    for label, s in settlements.items():
        sell = s[s.q_lem_bid < 0]
        shares[label] = sell.q_lem_accepted.sum() / sell.q_lem_bid.sum() if len(sell) else None
    table["lem_sell_accepted_share"] = pd.Series(shares)
    baseline = table.loc[RUNS["fixed_alpha_0_00"], "cost_total"]
    table["saving_vs_all_wm"] = baseline - table.cost_total
    return table.round(4)


def style(ax) -> None:
    """Recessive axes: no top/right spines, light grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="x" if ax.get_ylabel() == "" else "y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_total_cost(table: pd.DataFrame) -> None:
    """Figure 1: weekly total cost per run, one colour, value at each bar."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = range(len(table))
    ax.barh(list(y), table.cost_total, color=BLUE, height=0.55)
    ax.axvline(0, color=MUTED, linewidth=1)
    for i, value in zip(y, table.cost_total):
        ax.text(value + (0.4 if value >= 0 else -0.4), i, f"{value:.2f}",
                va="center", ha="left" if value >= 0 else "right", color=INK, fontsize=9)
    ax.set_yticks(list(y), table.index)
    ax.invert_yaxis()
    ax.set_xlabel("total cost C = C_LEM + C_WM + C_reBAP  [EUR / week]")
    ax.set_title("Portfolio cost, week 2016-06-20 to 06-26 (lower is better)",
                 loc="left", fontsize=10, color=INK)
    ax.set_xlim(table.cost_total.min() - 4, table.cost_total.max() + 4)
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "wp9_fig1_total_cost.png", dpi=200)
    plt.close(fig)


def fig_cost_components(table: pd.DataFrame) -> None:
    """Figure 2: the three cost parts per run, grouped bars around zero."""
    parts = {"cost_lem": ("C_LEM", BLUE), "cost_wm": ("C_WM", ORANGE),
             "cost_imbalance": ("C_reBAP", AQUA)}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.26
    for k, (column, (name, colour)) in enumerate(parts.items()):
        x = [i + (k - 1) * width for i in range(len(table))]
        ax.bar(x, table[column], width=width - 0.03, color=colour, label=name)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(table)), [label.replace(", ", ",\n").replace(" (", "\n(")
                                      for label in table.index], fontsize=8)
    ax.set_ylabel("EUR / week  (+ pay, − earn)")
    ax.set_title("Cost components per run", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower right",
              bbox_to_anchor=(1.0, 1.0))  # above the plot, never on a bar
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "wp9_fig2_cost_components.png", dpi=200)
    plt.close(fig)


def fig_one_day(settlements: dict) -> None:
    """Figure 3: one day of the optimised run: residual load, LEM offer and acceptance."""
    s = settlements[RUNS["optimised_option1"]].loc[DAY]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    kwargs = {"where": "post", "linewidth": 2}
    ax.step(s.index, s.residual_load, color=MUTED, label="residual load L (realised)",
            **kwargs)
    ax.step(s.index, s.q_lem_bid, color=BLUE, label="LEM offer α·L̂ (α = 1 when selling, so it covers L)", **kwargs)
    ax.step(s.index, s.q_lem_accepted, color=ORANGE, label="LEM accepted A",
            linestyle="--", **kwargs)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylabel("MWh per 15 min  (+ buy, − sell)")
    ax.set_title(f"Optimised α (option 1), {DAY}: the LEM takes only part of the offer",
                 loc="left", fontsize=10, color=INK)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "wp9_fig3_one_day.png", dpi=200)
    plt.close(fig)


def main() -> None:
    summary, settlements = load_runs()
    table = results_table(summary, settlements)
    table.to_csv(RESULTS / "wp9_results_table.csv")
    print(table.T.to_string())
    fig_total_cost(table)
    fig_cost_components(table)
    fig_one_day(settlements)
    print(f"\nfigures written to {RESULTS}")


if __name__ == "__main__":
    main()
