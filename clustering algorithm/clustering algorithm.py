import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import matplotlib.pyplot as plt
from tqdm import tqdm

config = {
    "input_csv": "pyridine_based_database.csv",
    "smiles_col": "SMILES",
    "output_csv": "clustered_data.csv",
    "n_workers": 8,
    "fingerprint_params": {
        "radius": 3,
        "n_bits": 2048,
        "use_features": True
    },
    "umap_params": {
        "n_components": 30,
        "n_neighbors": 30,
        "min_dist": 0.1
    },
    "hdbscan_params": {
        "min_cluster_size": 50,
        "min_samples": 20
    }
}


def main():
    print("Loading data...")
    df = pd.read_csv(config["input_csv"])
    smiles_list = df[config["smiles_col"]].tolist()

    print("Generating fingerprints...")
    fps = []
    valid_indices = []
    for idx, smi in enumerate(tqdm(smiles_list)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=config["fingerprint_params"]["radius"],
                    nBits=config["fingerprint_params"]["n_bits"],
                    useFeatures=config["fingerprint_params"]["use_features"]
                )
                fps.append(fp)
                valid_indices.append(idx)
        except:
            continue

    np_fps = []
    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    X = np.vstack(np_fps)

    print("Performing UMAP reduction...")
    reducer = umap.UMAP(
        n_components=config["umap_params"]["n_components"],
        n_neighbors=config["umap_params"]["n_neighbors"],
        min_dist=config["umap_params"]["min_dist"],
        random_state=42,
        n_jobs=config["n_workers"]
    )
    X_umap = reducer.fit_transform(X)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config["hdbscan_params"]["min_cluster_size"],
        min_samples=config["hdbscan_params"]["min_samples"],
        gen_min_span_tree=True,
        core_dist_n_jobs=config["n_workers"]
    )
    cluster_labels = clusterer.fit_predict(X_umap)

    print("Visualizing...")
    viz = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(viz[:, 0], viz[:, 1],
                          c=cluster_labels,
                          cmap='Spectral',
                          s=5,
                          alpha=0.6)
    plt.colorbar(scatter)
    plt.title("UMAP Projection of Molecular Clusters")
    plt.savefig("cluster_visualization.png")
    plt.close()

    print("Sampling molecules...")
    df_valid = df.iloc[valid_indices].copy()
    df_valid['cluster'] = cluster_labels

    noise_mask = df_valid['cluster'] == -1
    df_valid.loc[noise_mask, 'cluster'] = df_valid['cluster'].max() + 1

    sampled_indices = []
    for cluster_id in df_valid['cluster'].unique():
        cluster_df = df_valid[df_valid['cluster'] == cluster_id]
        n_samples = max(2, int(len(cluster_df) * 0.05))
        sampled = cluster_df.sample(n=min(n_samples, len(cluster_df)),
                                    random_state=42).index
        sampled_indices.extend(sampled)

    df_valid['selected'] = False
    df_valid.loc[sampled_indices, 'selected'] = True

    print("Saving results...")
    df_valid.to_csv(config["output_csv"], index=False)
    print(f"Clustering completed. Total clusters: {cluster_labels.max() + 1}")
    print(f"Selected {len(sampled_indices)} molecules for DFT calculation.")


if __name__ == "__main__":
    main()