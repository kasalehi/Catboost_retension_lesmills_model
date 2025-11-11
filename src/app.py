from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

# ----------- Minimal feature engineering (same logic you used) -----------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()

def _consecutive_zeros(s: pd.Series) -> pd.Series:
    run=0; out=[]
    for v in s.fillna(0).astype(int).tolist():
        run = run+1 if v==0 else 0
        out.append(run)
    return pd.Series(out, index=s.index)

def _consecutive_zeros_excluding_paused(y: pd.Series, p: pd.Series) -> pd.Series:
    run=0; out=[]
    y=y.fillna(0).astype(int); p=p.fillna(0).astype(int)
    for v,q in zip(y.tolist(), p.tolist()):
        if q==1: run=0
        elif v==0: run+=1
        else: run=0
        out.append(run)
    return pd.Series(out, index=y.index)

def _standardize_by_group(df: pd.DataFrame, keys: List[str], cols: List[str]) -> pd.DataFrame:
    eps=1e-6
    stats=df.groupby(keys)[cols].agg(['mean','std']); stats.columns=['_'.join(c) for c in stats.columns]
    df=df.join(stats, on=keys)
    for c in cols:
        mu=f"{c}_mean"; sd=f"{c}_std"
        df[c+'_z']=(df[c]-df[mu])/(df[sd].replace(0, np.nan)+eps)
        df.drop([mu,sd], axis=1, inplace=True)
    return df

def compute_member_features(df: pd.DataFrame, ema_span:int=4, seasonality:bool=True)->pd.DataFrame:
    df=df.copy().sort_values(["member_id","week"]).reset_index(drop=True)
    if not np.issubdtype(df["week"].dtype, np.datetime64): df["week"]=pd.to_datetime(df["week"])
    if "paused" not in df.columns: df["paused"]=0
    parts=[]
    for mid,g in df.groupby("member_id", sort=False):
        g=g.sort_values("week")
        y=g["engagement"].astype(float); p=g["paused"].fillna(0).astype(int)
        ema=_ema(y, span=ema_span); momentum=ema.diff().fillna(0.0)
        vol=_ema((y-ema).abs(), span=ema_span)
        drought=_consecutive_zeros(y); drought_active=_consecutive_zeros_excluding_paused(y,p)
        # simple within-self percentile (kept linear-time by using rank on full group once)
        pct = g["engagement"].rank(pct=True).astype(float)
        tenure=(g["week"]-g["week"].min()).dt.days//7
        out=g.copy()
        out["ema"]=ema; out["momentum"]=momentum; out["volatility"]=vol
        out["drought"]=drought; out["drought_active"]=drought_active
        out["pct_self"]=pct; out["tenure_weeks"]=tenure
        parts.append(out)
    out=pd.concat(parts, ignore_index=True)
    if seasonality:
        woy=out["week"].dt.isocalendar().week.astype(int)
        out["sin_woy"]=np.sin(2*np.pi*woy/52.0); out["cos_woy"]=np.cos(2*np.pi*woy/52.0)
    out=_standardize_by_group(out, ["member_id"], ["engagement","ema","momentum","volatility"])
    for flag in ["payment_failed","discount_active","price_change","negative_feedback","campaign_exposed"]:
        if flag not in out.columns: out[flag]=0
    return out

# Reason codes (same thresholds as before)
def _reason_codes(feat: pd.DataFrame, k_weeks:int=6)->List[str]:
    dcol="drought_active" if "drought_active" in feat.columns else "drought"
    out=[]
    for _,r in feat.iterrows():
        if int(r.get("paused",0))==1:
            out.append("paused_state"); continue
        tags=[]
        if r.get("momentum_z",0)<-0.5: tags.append("negative_momentum")
        if r.get(dcol,0)>=max(2, k_weeks//2): tags.append("drought_streak")
        if r.get("volatility_z",0)>0.75: tags.append("erratic_usage")
        if r.get("payment_failed",0)==1: tags.append("payment_issue")
        if r.get("price_change",0)==1 and r.get("discount_active",1)==0: tags.append("price_sensitivity")
        out.append("|".join(tags) if tags else "none")
    return out








# ----------- CatBoost training / scoring -----------

from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

def _build_dataset(df: pd.DataFrame, k_weeks:int=6):
    """Future-k label from 'churn_this_week' proxy built from drought≥8 or canceled==1 (if present)."""
    feat = compute_member_features(df, ema_span=4, seasonality=True).copy()
    # Build a simple hazard proxy: churn event if long drought or explicit cancel flag present.
    dcol="drought_active" if "drought_active" in feat.columns else "drought"
    hazard = np.zeros(len(feat), dtype=int)
    if "canceled" in df.columns:
        merged = feat.merge(df[["member_id","week","canceled"]], on=["member_id","week"], how="left")
        hazard = np.where(merged["canceled"].fillna(0).astype(int)==1, 1, 0)
    drought_hit = (feat[dcol] >= 8).astype(int).to_numpy()
    churn_this_week = np.where(hazard==1, 1, drought_hit)

    # future-k label: any event in next k weeks within each member
    y_nextk = []
    for _, g in pd.DataFrame({"m":feat["member_id"], "y":churn_this_week}).groupby("m", sort=False):
        y = g["y"].to_numpy()
        out = np.zeros_like(y)
        for i in range(len(y)):
            j2 = min(i + k_weeks, len(y))
            out[i] = 1 if y[i:j2].sum() > 0 else 0
        y_nextk.extend(out.tolist())

    cols = [c for c in [
        "engagement_z","ema_z","momentum_z","volatility_z",
        dcol, "pct_self","tenure_weeks","paused",
        "payment_failed","discount_active","price_change","negative_feedback","campaign_exposed",
        "sin_woy","cos_woy"
    ] if c in feat.columns]

    X = feat[cols].astype(float).values
    y = np.array(y_nextk, dtype=int)
    groups = feat["member_id"].to_numpy()
    meta = feat[["member_id","week","engagement","paused"]].copy()
    return meta, feat, X, y, cols, groups

def train_catboost(df: pd.DataFrame, k_weeks:int=6, random_state:int=42):
    meta, feat, X, y, cols, groups = _build_dataset(df, k_weeks)
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=random_state)
    tr, te = next(gss.split(X, y, groups))
    pos_w = max(1.0, (len(y[tr]) - y[tr].sum()) / (y[tr].sum() + 1e-6))
    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="AUC",
        class_weights=[1.0, pos_w],
        random_seed=random_state, verbose=False
    )
    model.fit(X[tr], y[tr], eval_set=(X[te], y[te]))
    proba = model.predict_proba(X[te])[:,1]
    print(f"[CatBoost] Holdout AUC={roc_auc_score(y[te], proba):.3f}  AUPRC={average_precision_score(y[te], proba):.3f}")
    return model, cols, k_weeks

def score_catboost(model: CatBoostClassifier, cols: List[str], k_weeks:int, df: pd.DataFrame) -> pd.DataFrame:
    feat = compute_member_features(df, ema_span=4, seasonality=True)
    X = feat[[c for c in cols if c in feat.columns]].astype(float).values
    ews = model.predict_proba(X)[:,1]
    reasons = _reason_codes(feat, k_weeks=k_weeks)
    out = feat[["member_id","week"]].copy()
    out[f"prob_churn_{k_weeks}w"] = ews
    out["EWS"] = ews
    out["reasons"] = reasons
    return out

# ----------- Outreach helpers (unchanged semantics) -----------

def _pick_primary_reason(reasons: str) -> str:
    severity = ["drought_streak", "negative_momentum", "erratic_usage", "paused_state", "none"]
    parts = [x.strip() for x in str(reasons).split("|") if x and x.strip()]
    if not parts:
        return "none"
    for r in severity:
        if r in parts:
            return r
    return parts[0]

def _risk_band(pct: float) -> str:
    if pct >= 0.97: return "Critical"
    if pct >= 0.85: return "High"
    if pct >= 0.65: return "Medium"
    return "Low"

def _severity_rank(reason: str) -> int:
    order = ["drought_streak", "payment_issue", "negative_momentum", "erratic_usage", "price_sensitivity", "none", "paused_state"]
    try:
        return order.index(reason)
    except ValueError:
        return 99

def _latest_row_preferring_nonpaused(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("week")
    g_np = g[g["paused"] == 0]
    if not g_np.empty:
        return g_np.tail(1)
    return g.tail(1)

def _attach_playbook(df: pd.DataFrame) -> pd.DataFrame:
    playbook = {
        "negative_momentum": {
            "tag": "Routine nudge",
            "subject": "We saved your spot this week",
            "body": "You’ve been active recently—nice! To keep the rhythm, here’s an easy next step for this week. Book now → {quick_link}"
        },
        "erratic_usage": {
            "tag": "Consistency plan",
            "subject": "A simple weekly rhythm just for you",
            "body": "Let’s make it easier to stay consistent. Pick one slot each week (we suggest {suggested_slot}) and we’ll remind you."
        },
        "drought_streak": {
            "tag": "Reactivation",
            "subject": "We miss you—ready for a fresh start?",
            "body": "We haven’t seen you lately. We’ve added a small credit and a friction-free next step. Reactivate in 1 click → {winback_link}"
        },
        "paused_state": {
            "tag": "Paused — check-in",
            "subject": "We’ll be ready when you are",
            "body": "Your pause is active. Want to pick your return week now? It takes 10 seconds → {resume_link}"
        },
        "none": {"tag": "No outreach","subject": "","body": ""}
    }

    res = df.copy()
    # map each reason to its playbook dict (no join → no column name clashes)
    mapped = res["main_reason"].map(lambda r: playbook.get(r, playbook["none"]))
    res["action_tag"] = mapped.map(lambda d: d["tag"])
    res["subject"]    = mapped.map(lambda d: d["subject"])
    res["message"]    = mapped.map(lambda d: d["body"])
    return res


# ----------- Core builders -----------

def build_snapshots(scores: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    s = scores.merge(weekly_df[["member_id","week","engagement","paused"]], on=["member_id","week"], how="left")
    snap = (s.groupby("member_id", group_keys=False)
              .apply(_latest_row_preferring_nonpaused)
              .reset_index(drop=True))
    snap["ews_pct"] = snap["EWS"].rank(pct=True)
    snap["risk_band"] = snap["ews_pct"].apply(_risk_band)
    snap["main_reason"] = snap["reasons"].apply(_pick_primary_reason)
    return snap

def build_outreach_from_scores(scores: pd.DataFrame,
                               weekly_df: pd.DataFrame,
                               capacity: int = 500,
                               ews_threshold: Optional[float] = None,
                               recent_contacts: Optional[pd.DataFrame] = None,
                               cooldown_days: int = 21) -> Tuple[pd.DataFrame, pd.DataFrame]:
    snap = build_snapshots(scores, weekly_df)

    active = snap[snap["paused"] == 0].copy()
    paused = snap[snap["paused"] == 1].copy()

    if recent_contacts is not None and len(recent_contacts) > 0:
        rc = recent_contacts.copy()
        rc["last_contact_date"] = pd.to_datetime(rc["last_contact_date"], errors="coerce")
        active = active.merge(rc, on="member_id", how="left")
        active = active[(active["last_contact_date"].isna()) | ((active["week"] - active["last_contact_date"]).dt.days >= cooldown_days)].copy()

    active = _attach_playbook(active)
    paused = _attach_playbook(paused)

    active["sev_rank"] = active["main_reason"].apply(_severity_rank)
    active = active.sort_values(["sev_rank","EWS"], ascending=[True, False])

    if ews_threshold is not None:
        active_q = active[active["EWS"] >= ews_threshold].copy()
    else:
        active_q = active.head(capacity).copy()

    paused_q = paused.sort_values(["week"], ascending=[False]).copy()

    cols = ["member_id","week","engagement","paused","EWS","risk_band","main_reason","reasons","action_tag","subject","message"]
    return active_q[cols], paused_q[cols]

def build_outreach(
    df: pd.DataFrame,
    model: Optional[CatBoostClassifier] = None,
    feature_cols: Optional[List[str]] = None,
    k_weeks: int = 6,
    capacity: int = 500,
    ews_threshold: Optional[float] = None,
    recent_contacts: Optional[pd.DataFrame] = None,
    cooldown_days: int = 21
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CatBoostClassifier, List[str]]:
    if model is None or feature_cols is None:
        model, feature_cols, k_weeks = train_catboost(df, k_weeks=k_weeks, random_state=42)

    scores = score_catboost(model, feature_cols, k_weeks, df)

    active_q, paused_q = build_outreach_from_scores(
        scores, df,
        capacity=capacity,
        ews_threshold=ews_threshold,
        recent_contacts=recent_contacts,
        cooldown_days=cooldown_days
    )

    return active_q, paused_q, scores, model, feature_cols

