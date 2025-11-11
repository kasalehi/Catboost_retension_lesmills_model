import pandas as pd
from pathlib import Path
import traceback

# --- file lists (fix: 2023-11-15 -> 2024-11-15 to match your churn/snapshot dates) ---
list_churned = [
    'churned_2023-07-23.csv',
    'churned_2025-02-02.csv',
    'churned_2024-11-15.csv'
]
list_snaps = [
    'snapshot_52wks_2023-07-23.csv',
    'snapshot_52wks_2025-02-02.csv',
    'snapshot_52wks_2024-11-15.csv'
]
list_final = [
    'final_2023-07-23.csv',
    'final_2025-02-02.csv',
    'final_2024-11-15.csv'
]

# --- ensure output dir exists (final/) ---
Path('final').mkdir(parents=True, exist_ok=True)

for i in range(len(list_churned)):
    try:
        # 1) churn: weekly rows -> one row per member (bucket & churn_flag)
        dfc = pd.read_csv(Path('monthlychurned') / list_churned[i])
        dfc["end_month_bucket"] = pd.to_numeric(dfc["end_month_bucket"], errors="coerce")  # keep NaN

        churn_one = (
            dfc.sort_values(["member_id", "week"])
               .groupby("member_id", as_index=False)
               .agg(end_month_bucket=("end_month_bucket", "first"),
                    churn_flag=("churn_flag", "max"))
        )

        # 2) snapshots: latest risk_band per member
        dfs = pd.read_csv(Path('snapshots') / list_snaps[i], parse_dates=["week"])
        snap_latest = (
            dfs.sort_values(["member_id", "week"])
               .drop_duplicates(subset=["member_id"], keep="last")[["member_id", "risk_band"]]
        )
        snap_latest["risk_band"] = (
            snap_latest["risk_band"].astype(str).str.strip().str.capitalize()
            .replace({"Lowe": "Low", "Meduim": "Medium"})
        )

        # 3) LEFT JOIN snapshots → churn
        merged = snap_latest.merge(churn_one, on="member_id", how="left")
        merged["churn_flag"] = merged["churn_flag"].fillna(0).astype(int)

        order_rows = ["Low", "Medium", "High", "Critical"]
        month_cols = list(range(1, 13))

        # 4) counts by month bucket (1..12)
        counts = (
            merged[merged["end_month_bucket"].isin(month_cols)]
            .pivot_table(index="risk_band",
                         columns="end_month_bucket",
                         values="member_id",
                         aggfunc="nunique",
                         fill_value=0)
            .reindex(index=order_rows, fill_value=0)
            .reindex(columns=month_cols, fill_value=0)
        )

        # 5) null bucket (no end within 12m or not in churn)
        null_counts = (
            merged[merged["end_month_bucket"].isna()]
            .groupby("risk_band")["member_id"].nunique()
            .reindex(order_rows, fill_value=0)
            .rename("null")
        )

        # 6) totals per risk band (universe = snapshots)
        total_members = (
            merged.groupby("risk_band")["member_id"].nunique()
            .reindex(order_rows, fill_value=0)
        )
        total_churned = counts.sum(axis=1)
        total_not_churned = total_members - total_churned

        totals_df = pd.DataFrame({
            "Total_Members": total_members.astype(int),
            "Total_Churned": total_churned.astype(int),
            "Total_Not_Churned": total_not_churned.astype(int),
            "Total_Churned_And_Not": total_members.astype(int),  # = Total_Members
        })
        totals_df["%_Churned"] = (totals_df["Total_Churned"] / totals_df["Total_Members"]).fillna(0)

        # 7) per-month % columns using Total_Members denominator
        pct = counts.div(total_members.replace(0, pd.NA), axis=0).fillna(0)
        pct = pct.rename(columns={m: (f"%churned_{m}month" if m == 1 else f"%churned_{m}months")
                                  for m in month_cols})

        # 8) assemble final
        final = pd.concat([counts, null_counts, pct, totals_df], axis=1)

        # round % columns
        pct_cols = [c for c in final.columns if isinstance(c, str) and c.startswith("%churned_")] + ["%_Churned"]
        final[pct_cols] = final[pct_cols].round(4)

        final.to_csv(Path('final') / list_final[i], index=True)
        print(f"Saved final/{list_final[i]}")

    except Exception as e:
        print(f"Error on set {i+1}: churn='{list_churned[i]}', snapshot='{list_snaps[i]}'")
        traceback.print_exc()
