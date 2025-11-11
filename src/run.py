from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import app
from datetime import datetime
 
# In run.py
df = pd.read_csv(
    "attendance_52wks_2024-11-15.csv",
    parse_dates=["week"],
    dtype={         # adjust names to your file
        "MembershipID": "string",
        "paused": "int16"
    },
    low_memory=False
)
 
active_q, paused_q, scores, model, feature_cols = app.build_outreach(
    df,
    k_weeks=6,
    capacity=500
)
scores = app.score_catboost(model, feature_cols, k_weeks=6, df=df)
snapshots = app.build_snapshots(scores, df)
snapshots["main_reason"] =snapshots["main_reason"].replace({
    "erratic_usage": "Irregular_Movement",
    "drought_streak": "Continously_Inactive"
})
snapshots["reasons"] =snapshots["reasons"].replace({
    "erratic_usage": "Irregular_Movement",
    "drought_streak": "Continously_Inactive"
})
snapshots.to_csv("snapshot__52wks_2024-11-15.csv", index=False)