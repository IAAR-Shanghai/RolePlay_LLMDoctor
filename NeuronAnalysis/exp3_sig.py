# exp3_silhouette_significance.py

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

RAW_DIR  = "results/exp3_Sig/silhouette/raw"
OUT_DIR  = "results/exp3_Sig/silhouette/significance"
os.makedirs(OUT_DIR, exist_ok=True)

ROLES = None  # later infer from files
N_BOOT = 1000
ALPHA  = 0.05

def bootstrap_ci(x, n_boot=N_BOOT):
    means = [np.random.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5,97.5])
    return lo, hi

def main():
    files = glob.glob(os.path.join(RAW_DIR, "*_silhouette_samples.csv"))
    if not files:
        raise RuntimeError("No silhouette sample files found.")
    data = {}
    for fp in files:
        fname = os.path.basename(fp)[:-4]
        dataset, role, _ = fname.split("_",2)
        df = pd.read_csv(fp)
        arr = df["silhouette"].values
        data.setdefault(dataset, {})[role] = arr
    for ds, role_dict in data.items():
        print(f"\n=== Dataset: {ds} ===")
        roles = sorted(role_dict.keys())
        samples = [role_dict[r] for r in roles]

        cis = {r: bootstrap_ci(role_dict[r]) for r in roles}
        print("Bootstrap 95% CI of mean silhouette:")
        for r in roles:
            lo,hi = cis[r]
            print(f"  {r:20s}: [{lo:.4f}, {hi:.4f}]")
        stat, p_f = friedmanchisquare(*samples)
        print(f"\nFriedman test:      chi2={stat:.3f}, p={p_f:.3g}")
        posthoc = []
        if p_f < ALPHA:
            print("Post-hoc Wilcoxon + Holm:")
            p_raw = []
            pairs = []
            for i in range(len(roles)):
                for j in range(i+1, len(roles)):
                    r1, r2 = roles[i], roles[j]
                    stat_w, p_w = wilcoxon(role_dict[r1], role_dict[r2])
                    p_raw.append(p_w)
                    pairs.append((r1,r2,stat_w))
            rej, p_adj, _, _ = multipletests(p_raw, alpha=ALPHA, method="holm")
            for k,(r1,r2,stat_w) in enumerate(pairs):
                ph = {"role1":r1, "role2":r2,
                      "stat":stat_w,
                      "p_raw":p_raw[k],
                      "p_adj":p_adj[k],
                      "significant": bool(rej[k])}
                posthoc.append(ph)
                flag = "YES" if rej[k] else " no"
                print(f"  {r1:<15} vs {r2:<15}: stat={stat_w:.3f}, p_raw={p_raw[k]:.3g}, p_adj={p_adj[k]:.3g} → {flag}")
        else:
            print("No significant difference by Friedman (p>=0.05).")

        out = {
            "dataset": ds,
            "friedman_stat": stat,
            "friedman_p": p_f,
            "bootstrap_CI": cis,
            "posthoc": posthoc
        }
        with open(os.path.join(OUT_DIR, f"{ds}_silhouette_sig.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {OUT_DIR}/{ds}_silhouette_sig.json")

if __name__ == "__main__":
    main()
