# exp5_significance_cross_role.py

import os, glob, re
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# ———— 目录配置 ————
RAW_DIR = os.path.join("results/exp5_sig", "raw")
OUT_DIR = os.path.join("results/exp5_sig", "significance")
os.makedirs(OUT_DIR, exist_ok=True)

def load_drop_series(fp):
    df = pd.read_csv(fp).set_index("question_idx")
    return df["drop"].sort_index()

def pairwise_mcnemar(a, b):
    df = pd.concat([a, b], axis=1, keys=["a","b"]).dropna().astype(int)
    a1b1 = int(((df.a==1)&(df.b==1)).sum())
    a1b0 = int(((df.a==1)&(df.b==0)).sum())
    a0b1 = int(((df.a==0)&(df.b==1)).sum())
    res = mcnemar([[a1b1, a1b0], [a0b1, 0]], exact=False, correction=False)
    return res.statistic, res.pvalue

def main():
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {RAW_DIR!r}")
    regex = re.compile(
    r"^(.+?)_"                                  # 1. model
    r"(resident2|med_student3|china_doctor)_"   # 2. src_role
    r"(resident2|med_student3|china_doctor)_"   # 3. tgt_role
    r"(role_diff|random)_"                      # 4. method
    r"(hard|soft)_"                             # 5. ablation
    r"(medmcqa|medqa|mmlu)\.csv$"               # 6. dataset
)

    files_map = {}
    for fp in files:
        fn = os.path.basename(fp)
        m = regex.match(fn)
        if not m:
            print("⚠ skip unrecognized file:", fn)
            continue
        model, src, tgt, method, abl, ds = m.groups()
        key = (model, method, abl, ds, src, tgt)
        files_map[key] = fp
    processed = set()
    for key, fp1 in files_map.items():
        model, method, abl, ds, src, tgt = key
        rev_key = (model, method, abl, ds, tgt, src)

        print(rev_key)
        if rev_key not in files_map:
            print(f"↔ missing reverse for {src}→{tgt} on {ds}, skip.")
            continue
        if (model,src, tgt, ds) in processed or (model,tgt, src, ds) in processed:
            continue

        fp2 = files_map[rev_key]
        drop1 = load_drop_series(fp1)
        drop2 = load_drop_series(fp2)
        stat, p_raw = pairwise_mcnemar(drop1, drop2)
        p_adj = p_raw   # 单次检验，无需多重校正
        sig   = (p_raw < 0.05)

        print(f"\n=== {model} | {method} | {abl} | {ds} ===")
        print(f" {src}→{tgt}  vs  {tgt}→{src}")
        print(f"  McNemar stat={stat:.3f}, p={p_raw:.4f}, significant={sig}")

        out_df = pd.DataFrame([{
            "model":       model,
            "method":      method,
            "ablation":    abl,
            "dataset":     ds,
            "direction1":  f"{src}→{tgt}",
            "direction2":  f"{tgt}→{src}",
            "stat":        stat,
            "p_raw":       p_raw,
            "p_adj":       p_adj,
            "significant": sig
        }])
        out_name = f"{model}_{method}_{abl}_{ds}_{src}_vs_{tgt}_sig.csv"
        out_path = os.path.join(OUT_DIR, out_name)
        out_df.to_csv(out_path, index=False)
        print(" → saved:", out_path)

        processed.add((src, tgt, ds))

if __name__ == "__main__":
    main()
