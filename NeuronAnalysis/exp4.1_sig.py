
import os
import glob
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
MODEL_NAME       = "Qwen2.5-32B-Instruct"
ROLES            = ["china_doctor", "med_student", "resident"]
SELECTION_METHOD = "role_diff"
ABLATION         = "hard"   # or "soft"
EXP4_DIR = "results/exp4_Sig"
OUT_DIR  = os.path.join(EXP4_DIR, "significance")
os.makedirs(OUT_DIR, exist_ok=True)

def load_role_data(model, role, method, ablation):
    pattern = f"{model}_{role}_{method}_{ablation}.csv"
    path = os.path.join(EXP4_DIR, pattern)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")
    return pd.read_csv(path)

def compute_drop_series(df):
    out = {}
    for ds in df["dataset"].unique():
        df_ds = df[df["dataset"]==ds]
        before = df_ds[df_ds["ablated"]==0].set_index("question_idx")["correct"]
        after  = df_ds[df_ds["ablated"]==1].set_index("question_idx")["correct"]
        both   = pd.concat([before, after], axis=1, keys=["before","after"]).dropna()
        drop   = ((both["before"]==1) & (both["after"]==0)).astype(int)
        out[ds] = drop
    return out

def run_significance(model, roles, method, ablation):
    role_drops = {}  # role -> {dataset: Series}
    for role in roles:
        df = load_role_data(model, role, method, ablation)
        role_drops[role] = compute_drop_series(df)

    for ds, _ in next(iter(role_drops.values())).items():
        print(f"\n=== Dataset: {ds} ===")
        drops = []
        for role in roles:
            drops.append(role_drops[role][ds])
        df_drop = pd.concat(drops, axis=1, keys=roles).dropna()
        mat = df_drop.values  # shape (N, k)
        qres = cochrans_q(mat)
        print(f"Cochran's Q = {qres.statistic:.3f}, p = {qres.pvalue:.4f}")
        omnibus_sig = qres.pvalue < 0.05

        results = []
        if omnibus_sig:
            p_raw = []
            pairs = []
            stats = []
            k = len(roles)
            for i in range(k):
                for j in range(i+1, k):
                    a = int(((mat[:,i]==1)&(mat[:,j]==1)).sum())
                    b = int(((mat[:,i]==1)&(mat[:,j]==0)).sum())
                    c = int(((mat[:,i]==0)&(mat[:,j]==1)).sum())
                    d = int(((mat[:,i]==0)&(mat[:,j]==0)).sum())
                    table = [[a,b],[c,d]]
                    mres = mcnemar(table, exact=False, correction=False)
                    p = mres.pvalue
                    p_raw.append(p)
                    pairs.append((roles[i], roles[j]))
                    stats.append(mres.statistic)

            reject, p_adj, *_ = multipletests(p_raw, alpha=0.05, method="holm")
            print("Pairwise McNemar + Holm:")
            for idx, ((r1,r2), stat, pr, pa, sig) in enumerate(zip(pairs, stats, p_raw, p_adj, reject)):
                mark = "Yes" if sig else "No"
                print(f"  {r1:15s} vs {r2:15s}: stat={stat:.3f}, raw_p={pr:.4f}, adj_p={pa:.4f}, sig={mark}")
                results.append({
                    "role1": r1,
                    "role2": r2,
                    "stat": stat,
                    "p_raw": pr,
                    "p_adj": pa,
                    "significant": sig
                })
        else:
            print("Omnibus not significant; skipping pairwise tests.")
        df_res = pd.DataFrame(results)
        out_csv = os.path.join(
            OUT_DIR,
            f"exp4_sig_{model}_{method}_{ablation}_{ds}.csv"
        )
        df_res.to_csv(out_csv, index=False)
        print(f"Saved significance results → {out_csv}")

if __name__ == "__main__":
    run_significance(MODEL_NAME, ROLES, SELECTION_METHOD, ABLATION)
