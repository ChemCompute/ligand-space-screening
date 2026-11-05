# Chemical-space-guided ligand screening

A reproducible implementation of ligand chemical-space partitioning and representative selection using **ChemBERTa → standardized embeddings → UMAP → Gaussian mixture models (GMM)**.

## Workflow

1. Read `ID` and `Smiles` from a CSV file.
2. Extract ChemBERTa-77M-MLM features by mean-pooling non-special tokens.
3. Standardize embeddings and project them onto a 2D UMAP space.
4. Partition the UMAP space with a **fixed** number of GMM components (`K=14` by default).
5. Relabel GMM components into deterministic left-to-right cluster IDs (`CID`).
6. Select up to three representative ligands from each cluster: **Medoid**, **Constrained Farthest**, and **Constrained Max-Min**.
7. Export clusters, representatives, coordinates, summary metrics and chemical-space figures.

> **Methodological note:** This release uses a fixed `K=14` matching the supplied script. It evaluates DBI, silhouette, BIC and AIC *at that K only*; it does **not** perform a search over K or establish that K=14 minimizes DBI. The supplied script's former one-point "DBI curve" has been omitted.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On first use, the default `DeepChem/ChemBERTa-77M-MLM` model will be downloaded from Hugging Face. Alternatively specify the path to an already downloaded local model with `--model`.

## Input format

```csv
ID,Smiles
1,CCO
2,CCN
3,c1ccccc1
```

Actual screening requires enough ligands for the specified cluster count and UMAP neighborhood size. `ID` must be unique; `Smiles` cannot be empty. This script does not chemically validate or canonicalize SMILES.

## Usage

```bash
python ligand_screening.py --input data/ligands.csv --output results
```

Using a locally stored model:

```bash
python ligand_screening.py --input data/ligands.csv --model /path/to/ChemBERTa-77M-MLM --output results
```

Optional retrospective check of original-study target IDs (not used to optimize clustering or representative selection):

```bash
python ligand_screening.py --input data/ligands.csv --output results --check-target-groups
```

Options can be inspected using `python ligand_screening.py --help`.

## Defaults

| Parameter | Value |
| --- | --- |
| Seed | 42 |
| ChemBERTa | DeepChem/ChemBERTa-77M-MLM |
| Batch size / sequence length | 64 / 128 |
| UMAP n_neighbors / min_dist | 5 / 0.05 |
| UMAP dimension / metric | 2 / Euclidean |
| GMM K / covariance / n_init | 14 / spherical / 10 |
| Representative posterior threshold | 0.70 |

The medoid is the observed ligand minimizing total Euclidean distance to cluster members in the UMAP plane. The farthest representative maximizes its Euclidean distance from the medoid among candidates above the posterior threshold. The max-min representative maximizes the minimum distance to previously selected ligands. If fewer than two eligible candidates are available, highest-posterior unselected ligands are added as fallback; the CSV records whether the threshold was met. **The posterior threshold checks max GMM posterior probability, matching the supplied source implementation.**

## Output files

| File | Description |
| --- | --- |
| `chemberta_embeddings.npy` | Molecular embeddings |
| `hxdata_with_CID.csv` | Original table with `CID` and `Top3` |
| `hxdata_umap_coordinates.csv` | 2D UMAP coordinates and GMM confidence |
| `initial_representative_ligands.csv` | Representative ligands |
| `cluster_statistics.csv` | Cluster sizes and mean coordinates/confidence |
| `cluster_number_evaluation.csv` | Metrics at fixed K |
| `ChemBERTa_UMAP_GMM_chemical_space.png` | Chemical-space visualization |
| `ChemBERTa_UMAP_GMM_chemical_space.pdf` | Vector chemical-space figure |
| `run_config.json` | Run parameters |
| `target_two_groups_top3_check.csv` | Optional original-study target-group diagnostics |

**No datasets, local model weights or generated results are included.** Results can differ with package versions, hardware, and data ordering. No guarantee is made that the original-study target IDs will be chosen by the fixed-parameter pipeline.

## Model attribution

Uses the pretrained [ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) model. Check its model card, license and citation requirements for reuse.
