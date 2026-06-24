"""2D comparison of per-marker correlations: (observed vs EMAmarker) vs (observed vs GrimageV2).

Aliases (see also data/EMAmarker_Grimage_mapping.csv):
  observed  : data/observed_data_for_grimagev2.csv
  EMAmarker : data/oike_dcEMA_on_grimagev2_markers.csv
  GrimageV2 : output/grimage_v2_wide.csv

For each marker this script computes two Pearson correlations against the
observed value -- one for the EMAmarker prediction, one for the GrimageV2
prediction -- and draws a single 2D scatter where each point is one marker:
    x = corr(observed, EMAmarker)
    y = corr(observed, GrimageV2)
A summary CSV with the underlying numbers is also written.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# Some observed columns contain Japanese (e.g. "CRP定量", "HbA1c(NGSP)").
# Prefer an installed CJK-capable font so labels do not render as tofu.
for _font in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "Noto Sans CJK JP", "IPAexGothic"):
    if any(_font == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _font
        break
matplotlib.rcParams["axes.unicode_minus"] = False

ID_COL = "sample_id"

OBSERVED_PATH = Path("data/observed_data_for_grimagev2.csv")
EMAMARKER_PATH = Path("data/oike_dcEMA_on_grimagev2_markers.csv")
GRIMAGE_PATH = Path("output/grimage_v2_wide.csv")

OUTPUT_DIR = Path("output")

# marker label -> (observed column, EMAmarker column, GrimageV2 column)
MARKER_MAP: dict[str, tuple[str, str, str]] = {
    "PACKYRS": ("pack_years", "cigarettes_pack_year", "DNAmPACKYRS"),
    "ADM": ("OID43339_ADM_observed", "OID43339_ADM", "DNAmADM"),
    "B2M": ("OID45441_B2M_observed", "OID45441_B2M", "DNAmB2M"),
    "CystatinC": ("OID45345_CST3_observed", "OID45345_CST3", "DNAmCystatinC"),
    "GDF15": ("OID45131_GDF15_observed", "OID45131_GDF15", "DNAmGDF15"),
    "Leptin": ("OID44746_LEP_observed", "OID44746_LEP", "DNAmLeptin"),
    "CRP": ("CRP定量", "crp", "DNAmlogCRP"),
    "A1C": ("HbA1c(NGSP)", "hba1c_ngsp", "DNAmlogA1C"),
    "PAI1": ("OID45256_SERPINE1_observed", "OID45256_SERPINE1", "DNAmPAI1"),
    "TIMP1": ("OID45425_TIMP1_observed", "OID45425_TIMP1", "DNAmTIMP1"),
}


def load_data() -> pd.DataFrame:
    """Load the three sources and merge them on sample_id (inner join)."""
    df_obs = pd.read_csv(OBSERVED_PATH)
    df_ema = pd.read_csv(EMAMARKER_PATH)
    df_grim = pd.read_csv(GRIMAGE_PATH)

    obs_cols = [ID_COL] + [c[0] for c in MARKER_MAP.values()]
    ema_cols = [ID_COL] + [c[1] for c in MARKER_MAP.values()]
    grim_cols = [ID_COL] + [c[2] for c in MARKER_MAP.values()]

    merged = (
        df_obs[obs_cols]
        .merge(df_ema[ema_cols], on=ID_COL, how="inner")
        .merge(df_grim[grim_cols], on=ID_COL, how="inner")
    )
    return merged


def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Pearson r, p-value and n over the rows where both values are present."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n >= 2 and x[mask].std() > 0 and y[mask].std() > 0:
        r, p = pearsonr(x[mask], y[mask])
        return float(r), float(p), n
    return float("nan"), float("nan"), n


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for marker, (obs_col, ema_col, grim_col) in MARKER_MAP.items():
        r_ema, p_ema, n_ema = _pearson(df[obs_col], df[ema_col])
        r_grim, p_grim, n_grim = _pearson(df[obs_col], df[grim_col])
        rows.append(
            {
                "marker": marker,
                "observed_col": obs_col,
                "ema_col": ema_col,
                "grimage_col": grim_col,
                "r_ema": r_ema,
                "p_ema": p_ema,
                "n_ema": n_ema,
                "r_grimage": r_grim,
                "p_grimage": p_grim,
                "n_grimage": n_grim,
            }
        )
    return pd.DataFrame(rows)


def plot_2d(summary: pd.DataFrame, out_path: Path) -> None:
    """Per-marker 2D scatter: x = corr(observed, EMAmarker), y = corr(observed, GrimageV2)."""
    x = summary["r_ema"]
    y = summary["r_grimage"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=55, alpha=0.85, color="#2ca02c", edgecolor="white", linewidth=0.6, zorder=3)
    for _, row in summary.iterrows():
        ax.annotate(
            row["marker"],
            (row["r_ema"], row["r_grimage"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    lim_lo = float(min(x.min(), y.min(), 0)) - 0.05
    lim_hi = float(max(x.max(), y.max(), 1)) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("corr(observed, EMAmarker)  Pearson r")
    ax.set_ylabel("corr(observed, GrimageV2)  Pearson r")
    ax.set_title("Per-marker correlation with observed: EMAmarker vs GrimageV2")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"merged samples: {len(df)}")

    summary = compute_summary(df)

    summary_path = OUTPUT_DIR / "correlation_comparison.csv"
    plot_path = OUTPUT_DIR / "correlation_comparison.png"
    summary.to_csv(summary_path, index=False)
    plot_2d(summary, plot_path)

    print(summary[["marker", "r_ema", "r_grimage", "n_ema", "n_grimage"]].to_string(index=False))
    print(f"\n2D plot -> {plot_path}")
    print(f"csv     -> {summary_path}")


if __name__ == "__main__":
    main()
