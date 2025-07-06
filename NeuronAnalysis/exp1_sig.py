import os
import glob
import pandas as pd
import itertools
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
import os, glob
import pandas as pd

def load_data(folder_path=None,
              models=None,
              datasets=None,
              roles=None,
              pattern="Sig1_*_*_*_details.xlsx"):
    if folder_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.normpath(
            os.path.join(script_dir, '..', 'results', 'exp1_Sig')
        )
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"Data folder not found: {folder_path}")

    records = []
    pattern_path = os.path.join(folder_path, pattern)
    for filepath in glob.glob(pattern_path):
        fname = os.path.basename(filepath)
        name = fname[:-5]  # 去掉 ".xlsx"
        parts = name.split('_')
        # 要求：Sig1, MODEL, DATASET, 任意多个 ROLE 片段, details
        if len(parts) < 5 or parts[0] != 'Sig1' or parts[-1] != 'details':
            continue

        model = parts[1]
        dataset = parts[2]
        # 把 parts[3:-1] （可能是 ['china','doctor'] 或 ['emergency','doctor']）拼回一个字符串
        raw_role = '_'.join(parts[3:-1])
        # 如果文件名里角色实际用空格，也可以替换：
        role = raw_role.replace(' ', '_')

        # 应用过滤
        if models and model not in models:
            continue
        if datasets and dataset not in datasets:
            continue
        if roles and role not in roles:
            continue

        df = pd.read_excel(filepath)
        if 'correct' not in df.columns:
            raise ValueError(f"'correct' column missing in {filepath}")
        if 'question_id' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'question_id'})

        # 规范化question_id列
        if not df['question_id'].astype(str).str.match(f'^{dataset}_\d+$').all():
            # 如果不是"数据集_数字"格式，则重新编号
            df['question_id'] = [f"{dataset}_{i+1}" for i in range(len(df))]
        
        df = df[['question_id', 'correct']].copy()
        df['model'] = model
        df['dataset'] = dataset
        df['role'] = role
        records.append(df)

    if not records:
        raise RuntimeError("No matching files found after applying filters.")
    return pd.concat(records, ignore_index=True)
def prepare_pivot(grp_df):
    pivot = grp_df.pivot(
        index='question_id',
        columns='role',
        values='correct'
    )
    # Keep only questions present under all roles
    pivot = pivot.dropna(axis=0, how='any').astype(int)
    return pivot


def omnibus_test(pivot):
    res = cochrans_q(pivot.values)
    return res.statistic, res.pvalue


def pairwise_mcnemar_holm(pivot, alpha=0.05):
    roles = pivot.columns.tolist()
    pairs = list(itertools.combinations(roles, 2))
    raw_pvals = []
    stats = []
    bc_list = []
    for i, j in pairs:
        # b: 1->0; c: 0->1
        b = int(((pivot[i] == 1) & (pivot[j] == 0)).sum())
        c = int(((pivot[i] == 0) & (pivot[j] == 1)).sum())
        table = [[None, b], [c, None]]
        # a,d not needed for McNemar
        test = mcnemar(table, exact=False, correction=False)
        raw_pvals.append(test.pvalue)
        stats.append(test.statistic)
        bc_list.append((b, c))

    reject, p_holm, _, _ = multipletests(raw_pvals, alpha=alpha, method='holm')

    results = []
    for idx, (i, j) in enumerate(pairs):
        b, c = bc_list[idx]
        results.append({
            'pair': f"{i} vs {j}",
            'b': b,
            'c': c,
            'chi2': stats[idx],
            'p_raw': raw_pvals[idx],
            'p_adj': p_holm[idx],
            'significant': bool(reject[idx])
        })
    return results


def run_significance_tests(data, alpha=0.05):
    """
    Loop over each (model, dataset) group and perform:
      1. Cochran's Q omnibus test across roles
      2. If omnibus significant, pairwise McNemar + Holm correction
    Returns summary DataFrame.
    """
    summary = []
    for (model, dataset), grp in data.groupby(['model', 'dataset']):
        print(f"\n=== Model: {model} | Dataset: {dataset} ===")
        pivot = prepare_pivot(grp)
        if pivot.empty:
            print("  Skipping: no complete question set across roles.")
            continue

        # 1. Omnibus test
        q_stat, q_p = omnibus_test(pivot)
        print(f"Cochran's Q = {q_stat:.3f}, p = {q_p:.4f}")
        summary.append({
            'model': model,
            'dataset': dataset,
            'test': 'CochranQ',
            'roles': list(pivot.columns),
            'statistic': q_stat,
            'p_raw': q_p,
            'p_adj': q_p,
            'significant': (q_p < alpha)
        })

        # 2. Pairwise if omnibus significant
        if q_p < alpha:
            print("  -> omnibus significant: running pairwise McNemar + Holm")
            pair_results = pairwise_mcnemar_holm(pivot, alpha)
            for pr in pair_results:
                print(f"    {pr['pair']}: raw p={pr['p_raw']:.4f}, "
                      f"Holm p={pr['p_adj']:.4f}, sig={pr['significant']}")
                summary.append({
                    'model': model,
                    'dataset': dataset,
                    'test': pr['pair'],
                    'roles': pr['pair'].split(' vs '),
                    'statistic': pr['chi2'],
                    'p_raw': pr['p_raw'],
                    'p_adj': pr['p_adj'],
                    'significant': pr['significant']
                })
        else:
            print("  -> omnibus not significant: skip pairwise tests.")

    return pd.DataFrame(summary)


def main():
    data = load_data(
        models=None,       
        datasets=None,
        roles=None         
    )
    results_df = run_significance_tests(data)
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'results', 'exp1_Sig',
        'sig1_summary_results.csv'
    )
    results_df.to_csv(out_path, index=False)
    print(f"\nSummary saved to {out_path}")


if __name__ == '__main__':
    main()
