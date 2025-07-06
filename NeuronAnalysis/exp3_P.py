

import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
PCA_DIR = "results/exp3_32/pca"
OUT_DIR = "results/exp3_Sig/silhouette/raw"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    files = glob.glob(os.path.join(PCA_DIR, "*_hidden.npy"))
    if not files:
        raise RuntimeError(f"No hidden npy files under {PCA_DIR}")

    for fp in files:
        name = os.path.basename(fp)[:-4]  # 去掉 .npy
        assert name.endswith("_hidden"), name
        dataset, role = name.split("_",1)
        role = role[:-7]  # 去掉 "_hidden"

        print(f"→ Processing {dataset} / {role}")
        arr = np.load(fp)  # shape (n_samples, dim)
        # 聚为 2 类
        km = KMeans(n_clusters=2, random_state=42).fit(arr)
        labs = km.labels_
        sils = silhouette_samples(arr, labs)

        # 保存到 CSV
        df = pd.DataFrame({
            "dataset":      dataset,
            "role":         role,
            "sample_idx":   np.arange(len(sils)),
            "silhouette":   sils
        })
        out_csv = os.path.join(OUT_DIR, f"{dataset}_{role}_silhouette_samples.csv")
        df.to_csv(out_csv, index=False)
        print(f"   ↳ saved {out_csv}")

    print("\n✅ All silhouette samples saved under", OUT_DIR)

if __name__ == "__main__":
    main()
