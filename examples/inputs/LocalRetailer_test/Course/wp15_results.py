"""
WP15: results table and figures of the round-2 runs (same style as wp9_results.py).

Run first (from the repository root, .venv active):
    python examples/inputs/LocalRetailer_test/Course/wp15_run_round2.py
Then:
    python examples/inputs/LocalRetailer_test/Course/wp15_results.py

Reads:   Course/results/wp15_summary.csv                         (wp15_run_round2.py)
         Course/results/round2_<run>/settlement.csv               (aggregator, per period)
         Course/results/round2_<run>/assume_output/*/market_meta.csv  (prices, fig. 3)
Writes:  Course/results/wp15_results_table.csv
         Course/results/wp15_fig1_aggregator_cost.png
         Course/results/wp15_fig2_cost_components.png
         Course/results/wp15_fig3_one_day.png
         Course/results/wp15_fig4_other_imbalance.png

Nothing new is computed here except sums, shares and differences of the WP15 outputs.
Units: volumes MWh per week (+ buy / - sell), costs EUR per week (+ pay / - earn).
Colours follow the market in every figure: LEM blue, WM orange, imbalance (reBAP) aqua.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COURSE_PATH = Path(__file__).resolve().parent
RESULTS = COURSE_PATH / "results"
SUMMARY_FILE = RESULTS / "wp15_summary.csv"

BETAS = (0.0, 0.5, 1.0)  # as wp15_run_round2.py
S_HAT_LABELS = {"full": "Ŝ = |L̂| (full acceptance)", "feedback": "Ŝ from feedback (7 days)"}
TIE_LABELS = {"lem": "tie → LEM", "wm": "tie → WM"}
# participant groups of settlement_participants.py
GROUPS = {"battery": "4 batteries", "community_PV": "community PV",
          "independent_demand": "9 independent nodes"}
# run shown in figure 3 and day shown in figure 3 (UTC)
DAY_RUN = "round2_beta_0_50_tie_lem_shat_feedback"
DAY = "2016-06-21"

# categorical colours: reference palette of the dataviz guideline, slots 1-3 (light),
# validated with validate_palette.js (all checks pass; aqua below 3:1 contrast, so
# every figure has a legend and the numbers are in wp15_results_table.csv)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def run_folder(run: str) -> Path:
    """Result folder of one run (wp15_run_round2.py, run_id())."""
    return RESULTS / run


def load_summary() -> pd.DataFrame:
    """wp15_summary.csv, one row per run, index = run name."""
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"{SUMMARY_FILE} missing: run wp15_run_round2.py first")
    return pd.read_csv(SUMMARY_FILE).set_index("run")


def results_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    One row per run with the aggregator totals and two derived numbers.

    lem_accepted_share: accepted / offered LEM volume of the aggregator over the week
        (absolute values, buy and sell together), from settlement.csv.
    saving_vs_all_wm: cost of the all-WM runs (tie -> WM; the aggregator never bids in
        the LEM, alpha = 0) minus the cost of this run; > 0 = cheaper than all-WM.
    """
    table = summary[[
        "beta", "tie_rule", "s_hat_rule", "agg_q_lem_accepted", "agg_q_wm_accepted",
        "agg_imbalance", "agg_cost_lem", "agg_cost_wm", "agg_cost_imbalance",
        "agg_cost_total", "agg_expected_cost_total", "agg_periods_alpha_gt_0",
    ]].copy()
    shares = {}
    for run in table.index:
        s = pd.read_csv(run_folder(run) / "settlement.csv", index_col=0)
        offered = s.q_lem_bid.abs().sum()
        shares[run] = s.q_lem_accepted.abs().sum() / offered if offered > 1e-12 else None
    table["lem_accepted_share"] = pd.Series(shares)
    all_wm = table.loc[table.tie_rule == "wm", "agg_cost_total"]
    # all tie -> WM runs give the same cost (the aggregator only uses the WM)
    assert (all_wm - all_wm.iloc[0]).abs().max() < 1e-6, "tie -> WM runs differ"
    table["saving_vs_all_wm"] = all_wm.iloc[0] - table.agg_cost_total
    return table.round(4)


def style(ax, grid_axis: str = "y") -> None:
    """Recessive axes: no top/right spines, light grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_aggregator_cost(table: pd.DataFrame) -> None:
    """
    Figure 1: weekly total cost of the aggregator. One panel per S_hat rule; per beta
    one bar for each tie rule; dashed line = all-WM cost.
    """
    all_wm = table.loc[table.tie_rule == "wm", "agg_cost_total"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    width = 0.36
    for ax, (s_hat, s_label) in zip(axes, S_HAT_LABELS.items()):
        for k, (tie, colour) in enumerate((("lem", BLUE), ("wm", ORANGE))):
            rows = table[(table.s_hat_rule == s_hat) & (table.tie_rule == tie)]
            rows = rows.set_index("beta").loc[list(BETAS)]
            x = [i + (k - 0.5) * width for i in range(len(BETAS))]
            ax.bar(x, rows.agg_cost_total, width=width - 0.04, color=colour,
                   label=TIE_LABELS[tie])
            for xi, value in zip(x, rows.agg_cost_total):
                # value inside the bar top, so it never collides with the dashed line
                ax.text(xi, value - 0.4, f"{value:.2f}", ha="center", va="top",
                        color="white", fontsize=7)
        ax.axhline(all_wm, color=MUTED, linewidth=1, linestyle="--",
                   label=f"all WM (α = 0): {all_wm:.2f}")
        ax.set_xticks(range(len(BETAS)), [f"β = {b:g}" for b in BETAS])
        ax.set_title(s_label, loc="left", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("aggregator cost  [EUR / week]")
    axes[0].set_ylim(0, table.agg_cost_total.max() + 3)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right",
                   bbox_to_anchor=(1.0, 1.08), ncol=3)
    fig.suptitle("Round 2: aggregator cost, week 2016-06-20 to 06-26 (lower is better)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "wp15_fig1_aggregator_cost.png", dpi=200)
    plt.close(fig)


def fig_cost_components(table: pd.DataFrame) -> None:
    """Figure 2: C_LEM, C_WM, C_reBAP of the aggregator for the runs with tie -> LEM."""
    rows = table[table.tie_rule == "lem"].sort_values(["s_hat_rule", "beta"],
                                                      ascending=[False, True])
    parts = {"agg_cost_lem": ("C_LEM", BLUE), "agg_cost_wm": ("C_WM", ORANGE),
             "agg_cost_imbalance": ("C_reBAP", AQUA)}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.26
    for k, (column, (name, colour)) in enumerate(parts.items()):
        x = [i + (k - 1) * width for i in range(len(rows))]
        ax.bar(x, rows[column], width=width - 0.03, color=colour, label=name)
    ax.axhline(0, color=MUTED, linewidth=1)
    labels = [f"β = {r.beta:g}\n{'Ŝ = |L̂|' if r.s_hat_rule == 'full' else 'Ŝ feedback'}"
              for r in rows.itertuples()]
    ax.set_xticks(range(len(rows)), labels, fontsize=8)
    ax.set_ylabel("EUR / week  (+ pay, − earn)")
    ax.set_title("Round 2, tie → LEM: cost components of the aggregator", loc="left",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower right",
              bbox_to_anchor=(1.0, 1.0))  # above the plot, never on a bar
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "wp15_fig2_cost_components.png", dpi=200)
    plt.close(fig)


def clearing_price(run: str, market_id: str) -> pd.Series:
    """
    Clearing price of one market per period from market_meta.csv; NaN where nothing
    cleared (ASSUME writes price 0 then, clearing_algorithms/simple.py, lines 162-163).
    Read from the market, not from settlement.csv: there price_wm is 0 in periods in
    which the aggregator had no WM order.
    """
    meta_file = next((run_folder(run) / "assume_output").glob("*/market_meta.csv"))
    meta = pd.read_csv(meta_file, parse_dates=["product_start"])
    market = meta[meta.market_id == market_id].set_index("product_start")
    return market.price.where(market.supply_volume.abs() > 1e-12)


def fig_one_day() -> None:
    """
    Figure 3: one day of one run, two panels with the same time axis (no second
    y-axis): volumes of the aggregator (top) and prices (bottom).
    """
    s = pd.read_csv(run_folder(DAY_RUN) / "settlement.csv", index_col=0,
                    parse_dates=True).loc[DAY]
    p_lem = clearing_price(DAY_RUN, "LEM_DA").loc[DAY]
    p_wm = clearing_price(DAY_RUN, "WM_DA").loc[DAY]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True,
                                      height_ratios=[3, 2])
    kwargs = {"where": "post", "linewidth": 2}
    top.step(s.index, s.residual_load, color=MUTED, label="residual load L (realised)",
             **kwargs)
    top.step(s.index, s.q_lem_bid, color=BLUE, linestyle="--", label="LEM offer α·L̂",
             **kwargs)
    top.step(s.index, s.q_lem_accepted, color=BLUE, label="LEM accepted A", **kwargs)
    top.step(s.index, s.q_wm_accepted, color=ORANGE, label="WM accepted W", **kwargs)
    top.axhline(0, color=MUTED, linewidth=1)
    top.set_ylabel("MWh per 15 min\n(+ buy, − sell)")
    top.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    top.set_title(f"{DAY_RUN}, {DAY} (UTC)", loc="left", fontsize=10, color=INK)
    style(top)

    bottom.step(p_lem.index, p_lem, color=BLUE, label="LEM clearing price", **kwargs)
    bottom.step(p_wm.index, p_wm, color=ORANGE, label="WM clearing price", **kwargs)
    bottom.set_ylabel("EUR/MWh")
    bottom.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    bottom.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    style(bottom)
    fig.tight_layout()
    fig.savefig(RESULTS / "wp15_fig3_one_day.png", dpi=200)
    plt.close(fig)


def fig_other_imbalance(summary: pd.DataFrame) -> None:
    """
    Figure 4: imbalance D of the other participants per beta (tie -> LEM, S_hat = |L_hat|;
    their bids do not depend on the aggregator's rules). One panel per group.
    """
    rows = summary[(summary.tie_rule == "lem") & (summary.s_hat_rule == "full")]
    rows = rows.set_index("beta").loc[list(BETAS)]
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(8, 2.8), sharey=True)
    for ax, (key, label) in zip(axes, GROUPS.items()):
        values = rows[f"{key}_D"].where(rows[f"{key}_D"].abs() >= 0.005, 0.0)  # no "-0.00"
        ax.bar(range(len(BETAS)), values, width=0.55, color=AQUA)
        for i, value in enumerate(values):
            ax.text(i, value + (0.05 if value >= 0 else -0.05), f"{value:.2f}",
                    ha="center", va="bottom" if value >= 0 else "top", color=INK,
                    fontsize=7)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(BETAS)), [f"β = {b:g}" for b in BETAS], fontsize=8)
        ax.set_title(label, loc="left", fontsize=9, color=INK)
        style(ax)
    low = min(0.0, min(rows[f"{key}_D"].min() for key in GROUPS))
    high = max(rows[f"{key}_D"].max() for key in GROUPS)
    axes[0].set_ylim(low - 0.4, high + 0.4)  # room for the value labels
    axes[0].set_ylabel("imbalance D  [MWh / week]\n(+ bought, − sold at reBAP)")
    fig.suptitle("Round 2: imbalance of the other participants (settled at reBAP)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "wp15_fig4_other_imbalance.png", dpi=200)
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    table = results_table(summary)
    table.to_csv(RESULTS / "wp15_results_table.csv")
    print(table.T.to_string())
    fig_aggregator_cost(table)
    fig_cost_components(table)
    fig_one_day()
    fig_other_imbalance(summary)
    print(f"\nfigures written to {RESULTS}")


if __name__ == "__main__":
    main()
