# FossilViT

FossilViT は、**3D標本から形態的な類似性を埋め込み空間として定量化する**ためのパイプラインです。  
具体的には、3Dメッシュを多視点レンダし、凍結した DINOv2（ViT）で画像特徴を抽出し、視点方向で統合して「標本ごとの埋め込みベクトル」を作ります。  
この埋め込み空間での距離（cosine/L2）を使って、(1) 類似標本検索 と (2) クラスタリング を行います。

> 要するに：
> - 3D形状 → 画像特徴 → 標本埋め込み
> - 埋め込み間の距離 = 分類形質（特徴量）の近さ
> - 距離に基づく検索・クラスタリングで、形態群の構造を解析

---

## パイプライン概要

1. **多視点レンダリング**（Open3D OffscreenRenderer）
   - 白背景・固定ライト
   - bboxでセンタリング + スケール正規化
   - 球面上の V 視点から画像生成
2. **特徴抽出**（凍結 DINOv2）
   - `torch.hub.load('facebookresearch/dinov2', model_name)`
   - `model.eval()` + `torch.inference_mode()` で推論のみ
   - 画像1枚ごとに特徴ベクトル `[D]`
3. **視点統合**（pooling）
   - 標本ごとに `[V, D]` を mean/max pooling
   - 最終的に標本埋め込み `[D]`
4. **活用**
   - 類似検索（NearestNeighbors）
   - クラスタリング（HDBSCAN推奨、KMeans任意）

---

## 用語（短い定義）

- **多視点レンダ**: 3D標本を複数方向から2D画像化する処理。
- **DINOv2 (ViT)**: ラベルなし学習済みの視覚Transformer。凍結特徴抽出器として使う。
- **特徴ベクトル `[D]`**: 画像1枚を D 次元で表現した数値ベクトル。
- **標本特徴 `[V, D]`**: 1標本の V 視点分の特徴を積んだ行列。
- **視点統合（pooling）**: `[V, D]` を1本の `[D]` にまとめる処理（mean がデフォルト）。
- **埋め込み（embedding）**: 標本を表す最終ベクトル。距離計算・検索・クラスタリングの入力。
- **HDBSCAN**: 密度ベースのクラスタリング。外れ点をノイズにできる。
- **ノイズ `-1`**: HDBSCAN が「どのクラスタにも属しにくい」と判断した点。
- **PCA `--pca 0.95`**: 累積寄与率95%を満たす最小次元数を自動選択する設定。
- **寄与率レポート（`--pca_report`）**: PCAで各主成分がどれだけ分散を説明したかのCSV。
- **UMAP**: クラスタリング補助の低次元化。密度構造を作りやすいが空間を歪めることがある。

---

## インストール

```bash
pip install -r requirements.txt
```

---

## 実行例（CLI）

以下は **必須 / 推奨 / 任意** の順で記載しています。

### 1) 必須（最低限、埋め込み生成まで）

```bash
python -m src.render_multiview --in data/meshes --out data/renders --views 12 --size 384
python -m src.extract_features --renders data/renders --out data/features --model dinov2_vits14 --device auto
python -m src.pool_embeddings --features data/features --out data/embeddings --pool mean
```

### 2) 推奨（まず価値が出る：類似検索）

```bash
python -m src.search --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --query 000 --topk 10 --metric cosine --out results
```

### 3) 任意（クラスタリング）

#### まずはこれ（推奨設定）

```bash
python -m src.cluster --emb data/embeddings/embeddings.npy --out results --method hdbscan \
  --normalize l2 --metric cosine --pca 64 --min_cluster_size 10 --min_samples 1 \
  --cluster_selection_method leaf --pca_report results/pca_report.csv
```

#### 困ったら試す（UMAPを追加）

```bash
python -m src.cluster --emb data/embeddings/embeddings.npy --out results --method hdbscan \
  --normalize l2 --metric cosine --pca 50 --umap --umap_n_components 15 --umap_n_neighbors 30 --umap_min_dist 0.0 \
  --min_cluster_size 10 --min_samples 1 --cluster_selection_method eom
```

> 注意: UMAPは密度クラスタリングを助ける一方で、空間を加工します。  
> ModelNetのようなラベル付きデータでは、**ARI/NMI/purity で妥当性を必ず確認**してください。

### 4) 推奨（クラスタ設定の探索）

```bash
python -m src.cluster_sweep --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --out results/sweep
```

ラベルがある場合（推奨）:

```bash
python -m src.cluster_sweep --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --labels labels.txt --out results/sweep
```

---

## HDBSCANでノイズを減らす実践原則

このリポジトリのデフォルトは、次を重視します（蒸着は最終手段）。

1. **HDBSCANを保守的にしすぎない**
   - 典型的には `min_samples` を下げるとノイズ率は下がりやすい
   - 逆に上げるとノイズ率が増えやすい
2. **クラスタリング専用の低次元化を使う**
   - まず PCA 固定次元（例: 64, 50）
   - 必要に応じて UMAP を追加
3. **埋め込みは L2 正規化 + cosine を基本にする**

---

## PCA設定の注意（重要）

- `--pca 0.95` は、**累積寄与率95%を満たす最小次元数を自動採用**する指定です。
- ただしこれは「情報保持」の観点であり、**クラスタリング適性を保証しません**。
- 0.95で次元が大きく残ると、密度推定が難しくなり HDBSCAN に不利な場合があります。
- そのためクラスタリング用途では、まず `--pca 64` や `--pca 50` の固定次元を推奨します。
- `--pca_report` は必須ではないですが、過剰圧縮や寄与率の偏り検知に有効です。

---

## cluster_sweep の評価方針（ごまかせない評価）

`src.cluster_sweep` は、ノイズ率だけでなく以下を同時評価します。

- `noise_ratio`
- `mean_prob`（HDBSCANの所属確信度）
- `largest_cluster_fraction`（最大クラスタの非ノイズ点占有率）
- `giant_cluster_penalty`（巨大ごった煮クラスタへの罰則）
- ラベルがある場合: `ARI`, `NMI`, `purity`

`final_score` は巨大混合クラスタを不利にする設計です。
`best_config.yaml` と `best_clusters.csv` は `final_score` 最大の設定で出力されます。

---

## クラスタ分析レポート生成

best_clusters.csv を論文向けに診断するスクリプト:

```bash
python -m src.analyze_clusters --clusters results/sweep/best_clusters.csv --out results
```

ラベル付きの場合:

```bash
python -m src.analyze_clusters --clusters results/sweep/best_clusters.csv --labels labels.txt --out results
```

出力:

- `results/cluster_report.json`
- `results/cluster_purity.csv`
- `results/largest_clusters.csv`

---

## 出力ディレクトリ構成

```text
project_root/
  data/
    meshes/
    renders/
    features/
    embeddings/
  results/
  src/
    render_multiview.py
    extract_features.py
    pool_embeddings.py
    cluster.py
    cluster_sweep.py
    analyze_clusters.py
    search.py
  configs/default.yaml
  requirements.txt
```

---

## 補足

- 蒸着（ノイズ点を後段で強制割当）は、解析上の都合で最終手段としてのみ使用してください。
- まずは `min_samples`・PCA固定次元・UMAP有無の見直しで密度構造を改善するのが正攻法です。
