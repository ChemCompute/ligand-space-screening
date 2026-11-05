"""ChemBERTa-based ligand chemical-space partitioning and representative selection."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from matplotlib.patches import Ellipse
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="CSV containing ID and Smiles columns")
    p.add_argument("--output", default=Path("results"), type=Path)
    p.add_argument("--model", default="DeepChem/ChemBERTa-77M-MLM", help="Hugging Face model ID or local folder")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neighbors", type=int, default=5)
    p.add_argument("--min-dist", type=float, default=0.05)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--k", type=int, default=14, help="Fixed number of GMM clusters")
    p.add_argument("--gmm-n-init", type=int, default=10)
    p.add_argument("--covariance", choices=["spherical", "diag", "full", "tied"], default="spherical")
    p.add_argument("--posterior-threshold", type=float, default=0.70)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--check-target-groups", action="store_true", help="Report membership of original study's ID groups")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ligands(path):
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "id" not in cols or "smiles" not in cols:
        raise ValueError("Input must contain columns 'ID' and 'Smiles' (case-insensitive).")
    id_col, smiles_col = cols["id"], cols["smiles"]
    df = df.dropna(subset=[smiles_col]).copy()
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col].ne("")].reset_index(drop=True)
    if df.empty:
        raise ValueError("No nonempty SMILES found.")
    if df[id_col].duplicated().any():
        raise ValueError("ID values must be unique.")
    return df, id_col, smiles_col


def chemberta_embeddings(smiles, model_name, device, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    blocks = []
    for start in tqdm(range(0, len(smiles), batch_size), desc="ChemBERTa embeddings"):
        batch = tokenizer(smiles[start:start + batch_size], padding=True, truncation=True,
                          max_length=max_length, return_special_tokens_mask=True, return_tensors="pt")
        special_mask = batch.pop("special_tokens_mask").to(device).bool()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            valid = (batch["attention_mask"].bool() & ~special_mask).unsqueeze(-1)
            pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        blocks.append(pooled.cpu().numpy())
    return np.concatenate(blocks, axis=0)


def assign_cids(gmm, raw_labels):
    order = np.lexsort((gmm.means_[:, 1], gmm.means_[:, 0]))
    mapping = {int(old): pos + 1 for pos, old in enumerate(order)}
    return np.array([mapping[int(v)] for v in raw_labels], dtype=int), order


def select_representatives(z, cids, posteriors, df, id_col, smiles_col, threshold):
    labels = np.full(len(df), "", dtype=object)
    records = []
    for cid in sorted(np.unique(cids)):
        idx = np.flatnonzero(cids == cid)
        points = z[idx]
        # The medoid minimizes the sum of within-cluster pairwise Euclidean distances.
        dmat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        medoid = int(idx[np.argmin(dmat.sum(axis=1))])
        chosen = [medoid]
        picks = [("Medoid", medoid)]
        remaining = [int(i) for i in idx if i != medoid]
        pool = [i for i in remaining if posteriors[i] >= threshold]
        needed = min(2, len(remaining))
        if len(pool) < needed:
            fallback = sorted((i for i in remaining if i not in pool),
                              key=lambda i: (-posteriors[i], i))
            pool.extend(fallback[:needed - len(pool)])
        if pool:
            farthest = max(pool, key=lambda i: (np.linalg.norm(z[i] - z[medoid]), -i))
            chosen.append(farthest)
            picks.append(("Constrained_Farthest", farthest))
        if len(idx) >= 3:
            remaining_pool = [i for i in pool if i not in chosen]
            if not remaining_pool:
                remaining_pool = [int(i) for i in idx if i not in chosen]
            maxmin = max(remaining_pool, key=lambda i: (min(np.linalg.norm(z[i] - z[j]) for j in chosen), -i))
            picks.append(("Constrained_MaxMin", maxmin))
        for kind, i in picks:
            labels[i] = kind
            records.append({"CID": int(cid), "Top3": kind, "RepresentativeType": kind,
                            id_col: df.iloc[i][id_col], smiles_col: df.iloc[i][smiles_col],
                            "UMAP1": float(z[i, 0]), "UMAP2": float(z[i, 1]),
                            "GMM_MaxPosterior": float(posteriors[i]),
                            "PosteriorThreshold": threshold,
                            "ConstraintSatisfied": bool(posteriors[i] >= threshold),
                            "Distance_to_Medoid": float(np.linalg.norm(z[i] - z[medoid])),
                            "MinDistance_to_Previously_Selected": float(min(np.linalg.norm(z[i] - z[j]) for j in chosen if j != i)) if i != medoid else np.nan})
    return labels, pd.DataFrame(records)


def covariance_matrix(gmm, component):
    t = gmm.covariance_type
    if t == "full":
        return gmm.covariances_[component]
    if t == "tied":
        return gmm.covariances_
    if t == "diag":
        return np.diag(gmm.covariances_[component])
    return np.eye(2) * gmm.covariances_[component]


def plot_space(z, cids, gmm, order, k, dbi, out, dpi):
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab20", k)
    for cid in range(1, k + 1):
        color = cmap(cid - 1)
        subset = cids == cid
        ax.scatter(z[subset, 0], z[subset, 1], s=30, alpha=0.8, color=color, label=f"CID {cid}")
        component = int(order[cid - 1])
        mean = gmm.means_[component]
        covariance = covariance_matrix(gmm, component)
        eigval, eigvec = np.linalg.eigh(covariance)
        ranks = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[ranks], eigvec[:, ranks]
        angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
        ellipse = Ellipse(mean, 4 * np.sqrt(eigval[0]), 4 * np.sqrt(eigval[1]),
                          angle=angle, fill=False, edgecolor=color, linewidth=1.5, alpha=0.65)
        ax.add_patch(ellipse)
        ax.scatter(*mean, marker="X", s=130, color=color, edgecolor="black", linewidth=0.8, zorder=10)
        ax.text(*mean, str(cid), fontsize=10, fontweight="bold", ha="center", va="bottom")
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=f"ChemBERTa-derived ligand chemical space\nUMAP + GMM, K = {k}, DBI = {dbi:.3f}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ChemBERTa_UMAP_GMM_chemical_space.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / "ChemBERTa_UMAP_GMM_chemical_space.pdf", bbox_inches="tight")
    plt.close(fig)


def check_target_groups(df, id_col, output):
    ids = pd.to_numeric(df[id_col], errors="coerce")
    group_a = ids.isin([1587, 1588])
    group_b = ids.between(1865, 1870, inclusive="both").fillna(False)
    selected = df["Top3"].ne("")
    report = df.loc[group_a | group_b, [id_col, "CID", "Top3"]].copy()
    report["Target_Group"] = np.where(pd.to_numeric(report[id_col], errors="coerce").isin([1587, 1588]), "A: 1587/1588", "B: 1865-1870")
    report["Matched"] = report["Top3"].ne("")
    report.to_csv(output / "target_two_groups_top3_check.csv", index=False)
    hits_a = set(df.loc[group_a & selected, "CID"])
    hits_b = set(df.loc[group_b & selected, "CID"])
    print(f"Target validation: A={int((group_a & selected).sum())}, B={int((group_b & selected).sum())}; shared CIDs={sorted(hits_a & hits_b)}")
    print("Note: target groups are diagnostic only; this script does not optimize or force their selection.")


def main():
    args = parse_args()
    if args.k < 2 or args.batch_size < 1 or args.max_length < 2 or not 0 <= args.posterior_threshold <= 1:
        raise ValueError("Check --k, --batch-size, --max-length and --posterior-threshold values.")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    df, id_col, smiles_col = load_ligands(args.input)
    if len(df) <= args.k or args.neighbors >= len(df):
        raise ValueError("Require number of valid ligands > max(k, neighbors).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {len(df)} ligands; inference device: {device}")
    embeddings = chemberta_embeddings(df[smiles_col].tolist(), args.model, device, args.batch_size, args.max_length)
    np.save(args.output / "chemberta_embeddings.npy", embeddings)
    scaled = StandardScaler().fit_transform(embeddings)
    z = umap.UMAP(n_neighbors=args.neighbors, min_dist=args.min_dist, n_components=2,
                  metric=args.metric, random_state=args.seed, init="spectral").fit_transform(scaled)
    gmm = GaussianMixture(n_components=args.k, covariance_type=args.covariance,
                          random_state=args.seed, n_init=args.gmm_n_init, reg_covar=1e-6)
    raw_labels = gmm.fit_predict(z)
    cids, order = assign_cids(gmm, raw_labels)
    max_posterior = gmm.predict_proba(z).max(axis=1)
    dbi = float(davies_bouldin_score(z, raw_labels))
    sil = float(silhouette_score(z, raw_labels))
    metrics = {"K": args.k, "DBI": dbi, "Silhouette": sil, "BIC": float(gmm.bic(z)), "AIC": float(gmm.aic(z))}
    pd.DataFrame([metrics]).to_csv(args.output / "cluster_number_evaluation.csv", index=False)
    labels, representatives = select_representatives(z, cids, max_posterior, df, id_col, smiles_col, args.posterior_threshold)
    df["CID"] = cids
    df["Top3"] = labels
    remaining = [col for col in df.columns if col not in [id_col, smiles_col, "CID", "Top3"]]
    df = df[[id_col, smiles_col, "CID", "Top3"] + remaining]
    df.to_csv(args.output / "hxdata_with_CID.csv", index=False)
    coords = pd.DataFrame({id_col: df[id_col], smiles_col: df[smiles_col], "CID": cids,
                           "UMAP1": z[:, 0], "UMAP2": z[:, 1], "GMM_MaxPosterior": max_posterior})
    coords.to_csv(args.output / "hxdata_umap_coordinates.csv", index=False)
    stats = coords.groupby("CID").agg(N=(id_col, "count"), Mean_UMAP1=("UMAP1", "mean"),
                                       Mean_UMAP2=("UMAP2", "mean"), Mean_MaxPosterior=("GMM_MaxPosterior", "mean")).reset_index()
    stats.to_csv(args.output / "cluster_statistics.csv", index=False)
    representatives.to_csv(args.output / "initial_representative_ligands.csv", index=False)
    plot_space(z, cids, gmm, order, args.k, dbi, args.output, args.dpi)
    if args.check_target_groups:
        check_target_groups(df, id_col, args.output)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Finished: K={args.k}, DBI={dbi:.4f}, silhouette={sil:.4f}, selected={len(representatives)}")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
