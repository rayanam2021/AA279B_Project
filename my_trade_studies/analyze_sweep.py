from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

filenames = [
    "magnitude_delay_angular_results_retry.csv",
    "parallel_resume_results.csv",
]

tables = []
for filename in filenames:
    table = pd.read_csv(RESULTS / filename)
    table["source_file"] = filename
    tables.append(table)
    print(f"{filename}: {len(table):,} rows")

combined = pd.concat(tables, ignore_index=True)

# These four inputs identify a sweep case.
keys = [
    "theta_deg",
    "phi_deg",
    "perturbation_dv",
    "delay_orbits",
]

# Prefer a successful, finite result when a case appears more than once.
numeric_columns = [
    "scp_total_dv",
    "baseline_total_dv",
    "min_separation",
]
for column in numeric_columns:
    combined[column] = pd.to_numeric(combined[column], errors="coerce")

finite = (
    combined[numeric_columns].notna().all(axis=1)
    & ~combined[numeric_columns].isin([float("inf"), -float("inf")]).any(axis=1)
)
combined["_returned_ok"] = combined["status"].eq("success") & finite

combined = (
    combined.sort_values("_returned_ok", kind="stable")
    .drop_duplicates(subset=keys, keep="last")
    .drop(columns="_returned_ok")
)

output = RESULTS / "combined_sweep_results.csv"
combined.to_csv(output, index=False)

print(f"\nUnique sweep cases: {len(combined):,}")
print(f"Expected full grid: {4 * 5 * 19 * 36:,}")
print("\nStatus counts:")
print(combined["status"].value_counts(dropna=False).to_string())

if "converged" in combined.columns:
    print("\nConvergence values:")
    print(combined["converged"].value_counts(dropna=False).to_string())

print(f"\nSaved: {output}")

# Normalize convergence values, including missing records.
convergence = (
    combined["converged"]
    .astype("string")
    .str.strip()
    .str.lower()
)

combined["convergence_group"] = "unknown"
combined.loc[
    convergence.eq("true").fillna(False), "convergence_group"
] = "converged"
combined.loc[
    convergence.eq("false").fillna(False), "convergence_group"
] = "not_converged"

for column in ["slack_max", "fuel_savings_percent"]:
    combined[column] = pd.to_numeric(combined[column], errors="coerce")

print("\nQuality checks by convergence group:")
for group, rows in combined.groupby("convergence_group"):
    print(f"\n{group}: {len(rows):,} cases")

    separation = rows["min_separation"]
    print(f"  Missing separation: {separation.isna().sum():,}")
    print(f"  Separation below 200 m: {(separation < 200).sum():,}")
    print(f"  Separation below 210 m: {(separation < 210).sum():,}")

    print("  Slack statistics:")
    print(
        rows["slack_max"]
        .describe()
        .to_string()
    )

    print("  Fuel savings (%) statistics:")
    print(
        rows["fuel_savings_percent"]
        .describe()
        .to_string()
    )

combined.to_csv(output, index=False)

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

all_results = combined.copy()
all_results["fuel_savings_percent"] = pd.to_numeric(
    all_results["fuel_savings_percent"], errors="coerce"
)

# Summary across all records.
summary = (
    all_results.groupby(["perturbation_dv", "delay_orbits"])
    .agg(
        cases=("theta_deg", "size"),
        valid_savings=("fuel_savings_percent", "count"),
        median_savings_percent=("fuel_savings_percent", "median"),
        mean_savings_percent=("fuel_savings_percent", "mean"),
    )
    .reset_index()
)

summary.to_csv(RESULTS / "all_records_summary.csv", index=False)
print("\nSummary — all records:")
print(summary.to_string(index=False))

# Median savings versus delay.
fig, ax = plt.subplots(figsize=(9, 5))

for magnitude, rows in summary.groupby("perturbation_dv"):
    rows = rows.sort_values("delay_orbits")
    ax.plot(
        rows["delay_orbits"],
        rows["median_savings_percent"],
        marker="o",
        label=f"Disturbance Δv = {magnitude:g} m/s",
    )

ax.axhline(0, color="gray", linewidth=1)
ax.set_xlabel("Recovery delay (orbits)")
ax.set_ylabel("Median fuel savings relative to baseline (%)")
ax.set_title("Fuel savings — all records, no quality filtering")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "all_records_median_savings_vs_delay.png", dpi=200)

# Direction heatmaps for all four magnitudes.
theta_values = np.arange(0, 360, 10)
phi_values = np.arange(-90, 91, 10)
delays = sorted(all_results["delay_orbits"].unique())

cmap = plt.get_cmap("RdBu").copy()
cmap.set_bad("lightgray")
norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)

for magnitude in sorted(all_results["perturbation_dv"].unique()):
    fig, axes = plt.subplots(
        1, len(delays),
        figsize=(19, 5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for ax, delay in zip(axes, delays):
        rows = all_results.loc[
            np.isclose(all_results["perturbation_dv"], magnitude)
            & np.isclose(all_results["delay_orbits"], delay)
        ]

        grid = (
            rows.pivot(
                index="phi_deg",
                columns="theta_deg",
                values="fuel_savings_percent",
            )
            .reindex(index=phi_values, columns=theta_values)
        )

        im = ax.imshow(
            np.ma.masked_invalid(grid.to_numpy(dtype=float)),
            origin="lower",
            extent=(-5, 355, -95, 95),
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )

        ax.set_title(f"Delay: {delay:g} orbits\n{len(rows)} cases")
        ax.set_xlabel("θ (degrees)")
        ax.set_xticks([0, 90, 180, 270])
        ax.set_yticks([-90, -45, 0, 45, 90])

    axes[0].set_ylabel("φ (degrees)")
    fig.suptitle(
        f"Fuel savings — all records — Δv = {magnitude:g} m/s"
    )
    fig.colorbar(
        im,
        ax=list(axes),
        label="Fuel savings (%) — colors saturate below −100%",
        shrink=0.8,
    )

    path = RESULTS / f"all_records_direction_heatmaps_{magnitude:.2f}.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")

plt.show()