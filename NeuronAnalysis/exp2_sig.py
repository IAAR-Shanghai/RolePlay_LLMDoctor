# exp2_significance_analysis.py

import os
import glob
import re
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

RAW_SIG_DIR = os.path.join("results", "exp2_Sig", "raw")
SIG_RESULTS_DIR = os.path.join("results", "exp2_Sig", "significance")
os.makedirs(SIG_RESULTS_DIR, exist_ok=True)

def paired_test(a, b, method="ttest"):
    if method == "wilcoxon":
        stat, p = wilcoxon(a, b)
    else:
        stat, p = ttest_rel(a, b)
    return stat, p

def analyze_pair(baseline_df, df_other, method="ttest"):
    # 合并
    merged = pd.merge(
        baseline_df, df_other,
        on=["dataset","question_id","layer"],
        suffixes=("_base","_other")
    )
    layers = sorted(merged["layer"].unique())
    records = []
    # 每层做配对检验
    for layer in layers:
        grp = merged[merged["layer"] == layer]
        stat, p_raw = paired_test(
            grp["js_value_base"], grp["js_value_other"], method=method
        )
        records.append({"layer": layer, "stat": stat, "p_raw": p_raw})
    # Holm 校正
    p_vals = [r["p_raw"] for r in records]
    reject, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method="holm")
    for i, rec in enumerate(records):
        rec["p_adj"]      = p_adj[i]
        rec["significant"]= bool(reject[i])
    return pd.DataFrame.from_records(records)

def main():
    pattern = os.path.join(RAW_SIG_DIR, "exp2_*_*_raw.csv")
    all_files = glob.glob(pattern)
    if not all_files:
        raise RuntimeError(f"No files found with pattern {pattern}")

    # 找到 baseline 文件
    baseline_file = next(
        f for f in all_files
        if os.path.basename(f).startswith("exp2_baseline_") and f.endswith("_raw.csv")
    )
    df_base = (
        pd.read_csv(baseline_file)
          .rename(columns={"js_value": "js_value_base"})
    )
    for fp in sorted(all_files):
        if fp == baseline_file:
            continue
        fname = os.path.basename(fp)
        m = re.match(r"exp2_(.+)_.+_raw\.csv", fname)
        if not m:
            print(f"跳过不匹配的文件: {fname}")
            continue
        prompt_key = m.group(1)

        df_other = (
            pd.read_csv(fp)
              .rename(columns={"js_value": "js_value_other"})
        )

        # 配对 t 检验 + Holm
        df_sig = analyze_pair(df_base, df_other, method="ttest")

        sig_count    = df_sig["significant"].sum()
        total_layers = len(df_sig)
        print(f"\n=== '{prompt_key}' 显著层: {sig_count}/{total_layers} ===")
        if sig_count:
            print(
                df_sig[df_sig["significant"]]
                  .loc[:,["layer","stat","p_raw","p_adj"]]
                  .to_string(index=False)
            )
        else:
            print("（无任何层显著）")

        # 保存结果
        out_csv = os.path.join(SIG_RESULTS_DIR, f"exp2_significance_{prompt_key}.csv")
        df_sig.to_csv(out_csv, index=False)
        print(f"✔ 已保存: {out_csv}")

if __name__ == "__main__":
    main()
