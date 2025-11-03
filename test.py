import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os, warnings
warnings.filterwarnings("ignore")

# 1) Generate synthetic dataset
def generate_synthetic_dataset(num_nodes=10, timesteps=100, seed=42):
    np.random.seed(seed)
    rows = []
    for node in range(num_nodes):
        base = 30 + node * 2.5
        amp = 6 + (node % 4) * 2
        phase = (node / num_nodes) * 2 * np.pi
        for t in range(timesteps):
            daily = amp * np.sin(2 * np.pi * (t % 24) / 24.0 + phase)
            noise = np.random.normal(0, 3)
            spike = 0
            if np.random.rand() < 0.01:
                spike = np.random.uniform(10, 35)
            load = max(0.0, base + daily + noise + spike)
            active_sessions = max(1, int(round(load / 3.0 + np.random.normal(0,1))))
            cpu = min(100, max(0, load + np.random.normal(0,4)))
            mem = min(100, max(0, cpu * 0.6 + np.random.normal(0,3)))
            latency = max(1, 20 + (node * 0.6) + np.sin(t/12.0)*2 + np.random.normal(0,1))
            rows.append({
                "time": t,
                "node": int(node),
                "load": load,
                "active_sessions": active_sessions,
                "cpu": cpu,
                "mem": mem,
                "latency": latency
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["node", "time"]).reset_index(drop=True)
    df["lag1"] = df.groupby("node")["load"].shift(1)
    df["lag2"] = df.groupby("node")["load"].shift(2)
    df["hour"] = df["time"] % 24
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df = df.fillna(method="bfill")
    return df

# 2) Save lightweight dataset PDF (summary + sample rows)
def save_dataset_pdf_light(df, out_pdf_path="/mnt/data/edge_dataset.pdf", sample_rows=30):
    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    pp = PdfPages(out_pdf_path)

    # Page 1: Summary
    fig1 = plt.figure(figsize=(8.27, 11.69))
    ax1 = fig1.add_subplot(111)
    ax1.axis('off')
    summary = df.groupby("node")["load"].describe().round(2)
    text = "Edge Dataset Summary (per node)\n\n" + summary.to_string()
    ax1.text(0.01, 0.99, text, va='top', ha='left', fontsize=8, family='monospace')
    pp.savefig(fig1, bbox_inches='tight')
    plt.close(fig1)

    # Page 2: Sample rows
    sub = df.head(sample_rows)
    fig2, ax2 = plt.subplots(figsize=(11, 8.5))
    ax2.axis('off')
    table = ax2.table(cellText=sub.values, colLabels=sub.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    ax2.set_title(f"Edge dataset sample (first {sample_rows} rows)", fontsize=10)
    pp.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)

    pp.close()
    return out_pdf_path

# 3) Train initial models (ARIMA + RF + GB) and compute MSE for weighting
def train_initial_models(df, num_nodes=10, train_end_time=60):
    arima_models, rf_models, gb_models = {}, {}, {}
    mse_stats = {}
    features = ["hour_sin","hour_cos","lag1","lag2","active_sessions","cpu","mem","latency"]

    for node in range(num_nodes):
        node_df = df[(df["node"]==node) & (df["time"] < train_end_time)].copy().reset_index(drop=True)
        y = node_df["load"].values
        split = int(len(y)*0.8) if len(y)>5 else int(len(y)-1)
        if split < 1: split = 1

        train_y, val_y = y[:split].tolist(), y[split:].tolist()

        # --- ARIMA ---
        try:
            history = train_y.copy()
            preds = []
            for obs in val_y:
                model = ARIMA(history, order=(2,0,2)).fit()
                preds.append(float(model.forecast(steps=1)[0]))
                history.append(obs)
            mse_arima = float(mean_squared_error(val_y, preds)) if len(preds)>0 else 1e-6
            arima_full = ARIMA(y, order=(2,0,2)).fit()
        except Exception:
            mse_arima = 1e-6
            arima_full = None

        # --- Random Forest ---
        X = node_df[features].values
        yvec = node_df["load"].values
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = yvec[:split], yvec[split:]
        rf = RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42)
        try:
            rf.fit(X_train, y_train)
            preds_rf = rf.predict(X_val)
            mse_rf = float(mean_squared_error(y_val, preds_rf)) if len(preds_rf)>0 else 1e-6
        except Exception:
            mse_rf = mse_arima
            rf = None

        # --- Gradient Boosting ---
        gb = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42)
        try:
            gb.fit(X_train, y_train)
            preds_gb = gb.predict(X_val)
            mse_gb = float(mean_squared_error(y_val, preds_gb)) if len(preds_gb)>0 else 1e-6
        except Exception:
            mse_gb = mse_rf
            gb = None

        arima_models[node] = arima_full
        rf_models[node] = rf
        gb_models[node] = gb
        mse_stats[node] = {
            "mse_arima": max(mse_arima, 1e-6),
            "mse_rf": max(mse_rf, 1e-6),
            "mse_gb": max(mse_gb, 1e-6)
        }

    return arima_models, rf_models, gb_models, mse_stats

# 4) Predict load with all three models and blend
def predict_blend_and_select(df, arima_models, rf_models, gb_models, mse_stats, current_time, num_nodes=10, eps=1e-6):
    preds = {}
    features = ["hour_sin","hour_cos","lag1","lag2","active_sessions","cpu","mem","latency"]

    for node in range(num_nodes):
        row = df[(df["node"]==node) & (df["time"]==current_time)].copy()
        if row.empty: row = df[(df["node"]==node)].iloc[[-1]]
        feat = row[features].iloc[0].values.reshape(1, -1)

        # Random Forest
        rf = rf_models.get(node)
        try: rf_pred = float(rf.predict(feat)) if rf else float(row["lag1"].values[0])
        except: rf_pred = float(row["lag1"].values[0])

        # Gradient Boosting
        gb = gb_models.get(node)
        try: gb_pred = float(gb.predict(feat)) if gb else float(row["lag1"].values[0])
        except: gb_pred = float(row["lag1"].values[0])

        # ARIMA
        arima_model = arima_models.get(node)
        try: arima_pred = float(arima_model.forecast(steps=1)[0]) if arima_model else float(row["lag1"].values[0])
        except: arima_pred = float(row["lag1"].values[0])

        # Blended weight = inverse MSE
        mse_ar = mse_stats[node]["mse_arima"] + eps
        mse_rf_ = mse_stats[node]["mse_rf"] + eps
        mse_gb_ = mse_stats[node]["mse_gb"] + eps
        w_ar, w_rf, w_gb = 1/mse_ar, 1/mse_rf_, 1/mse_gb_
        blended = (w_ar*arima_pred + w_rf*rf_pred + w_gb*gb_pred) / (w_ar + w_rf + w_gb)

        preds[node] = {"arima": arima_pred, "rf": rf_pred, "gb": gb_pred, "blend": blended}

    selected_node = min(preds.keys(), key=lambda x: preds[x]["blend"])
    return selected_node, preds

# 5) Assign job to node
def assign_job_to_node(df, node, time_step, job_load):
    mask = (df["node"]==node) & (df["time"]==time_step)
    if mask.sum() == 0: return df
    df.loc[mask, "load"] += job_load
    df.loc[mask, "active_sessions"] = (df.loc[mask, "active_sessions"] + int(round(job_load/3))).astype(int)
    df.loc[mask, "cpu"] = (df.loc[mask, "cpu"] + job_load*0.6).clip(0,100)
    df.loc[mask, "mem"] = (df.loc[mask, "mem"] + job_load*0.4).clip(0,100)
    return df

# 6) Orchestrator
def run_simulation(num_nodes=10, timesteps=100, initial_train_end=60, verbose_steps=8):
    df = generate_synthetic_dataset(num_nodes, timesteps)
    df.to_csv("/mnt/data/edge_dataset.csv", index=False)
    save_dataset_pdf_light(df, "/mnt/data/edge_dataset.pdf", 30)

    arima_models, rf_models, gb_models, mse_stats = train_initial_models(df, num_nodes, initial_train_end)

    assignment_log = []
    for t in range(initial_train_end, timesteps):
        selected_node, preds = predict_blend_and_select(df, arima_models, rf_models, gb_models, mse_stats, t, num_nodes)
        job_load = float(np.random.uniform(4.0, 15.0))
        df = assign_job_to_node(df, selected_node, t, job_load)

        entry = {
            "time": t,
            "selected_node": int(selected_node),
            "predicted_blend": float(preds[selected_node]["blend"]),
            "pred_arima": float(preds[selected_node]["arima"]),
            "pred_rf": float(preds[selected_node]["rf"]),
            "pred_gb": float(preds[selected_node]["gb"]),
            "job_load": round(job_load,2),
            "actual_load_after_assign": float(df[(df.node==selected_node) & (df.time==t)].iloc[0]["load"])
        }
        assignment_log.append(entry)

        if len(assignment_log) <= verbose_steps:
            print(f"[t={t}] Node {selected_node} | blend={entry['predicted_blend']:.2f}, arima={entry['pred_arima']:.2f}, rf={entry['pred_rf']:.2f}, gb={entry['pred_gb']:.2f} | job={entry['job_load']} | new_load={entry['actual_load_after_assign']:.2f}")

    assign_df = pd.DataFrame(assignment_log)
    df.to_csv("/mnt/data/edge_dataset_after_assignments.csv", index=False)
    assign_df.to_csv("/mnt/data/assignment_log.csv", index=False)
    return df, assign_df