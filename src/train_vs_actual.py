import pandas as pd
df=pd.read_csv("snapshot__52wks_2024-11-15.csv")
dg=pd.read_csv("attendance_52wks_2024-11-15.csv")
 
# Standardize churn_flag to 0/1
dg["churn_flag"] = (
    dg["churn_flag"].astype(str).str.strip().str.lower()
      .map({"1":1,"true":1,"yes":1,"y":1,
            "0":0,"false":0,"no":0,"n":0})
      .fillna(0).astype(int)
)
 
# Order risk bands
band_order = ["Low", "Medium", "High", "Critical"]
cat = pd.api.types.CategoricalDtype(band_order, ordered=True)
df["risk_band"] = df["risk_band"].astype(str).str.title().astype(cat)
 
# Collapse to member-level (worst band per member)
pred_member = df.groupby("member_id", as_index=False).agg(
    risk_band=("risk_band", "max")
)
 
# One row per member_id with prediction + actual churn
final = pred_member.merge(
    dg[["member_id", "churn_flag"]].drop_duplicates("member_id"),
    on="member_id", how="inner"
)
 
pd.crosstab(final["risk_band"], final["churn_flag"])
# --- CLEANING ---
band_order = ["Low", "Medium", "High", "Critical"]
cat = pd.api.types.CategoricalDtype(band_order, ordered=True)
df["risk_band"] = df["risk_band"].astype(str).str.strip().str.title().astype(cat)
 
# Normalize churn_flag to 0/1
dg["churn_flag"] = (
    dg["churn_flag"].astype(str).str.strip().str.lower()
      .map({"1":1,"true":1,"yes":1,"y":1,"0":0,"false":0,"no":0,"n":0})
      .fillna(0).astype(int)
)
 
# --- COLLAPSE df TO MEMBER-LEVEL (worst band per member) ---
pred_member = (
    df.groupby("member_id", as_index=False)
      .agg(risk_band=("risk_band", "max"))   # Critical > High > Medium > Low
)
 
# --- DISTINCT ACTUALS ---
actuals = dg.drop_duplicates(subset=["member_id"])[["member_id","churn_flag"]]
 
# --- MERGE ---
final = pred_member.merge(actuals, on="member_id", how="inner")
 
# --- INSIGHT TABLE (like the one you posted) ---
ct = pd.crosstab(final["risk_band"], final["churn_flag"]).reindex(band_order, fill_value=0)
ct.columns = ["not_churned(0)", "churned(1)"] if list(ct.columns)==[0,1] else ct.columns
table = ct.copy()
table["Total"] = table.sum(axis=1)
# % Churned = churned / total
churn_col = "churned(1)" if "churned(1)" in table.columns else 1
table["% Churned"] = (table[churn_col] / table["Total"]).round(3)
 
print("Risk-band insight table (member-level):")
print(table)
 
# --- QUICK PRODUCTION METRICS (treat High/Critical as 'predicted leave') ---
final["predicted_leave"] = final["risk_band"].isin(["High","Critical"]).astype(int)
tp = ((final["predicted_leave"]==1) & (final["churn_flag"]==1)).sum()
fp = ((final["predicted_leave"]==1) & (final["churn_flag"]==0)).sum()
tn = ((final["predicted_leave"]==0) & (final["churn_flag"]==0)).sum()
fn = ((final["predicted_leave"]==0) & (final["churn_flag"]==1)).sum()
 
precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
recall    = tp / (tp + fn) if (tp+fn)>0 else 0.0
f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
 
print("\nProduction metrics (Predicted leave = High/Critical):")
print({
    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    "precision": round(precision, 3),
    "recall":    round(recall, 3),
    "f1":        round(f1, 3),
})
table.to_csv('evaluate_52wks_2024-11-15.csv')