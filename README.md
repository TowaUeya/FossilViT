# FossilViT: 化石3Dデータの多視点DINOv2埋め込みパイプライン

化石の3Dメッシュ/点群（`.ply`, `.obj`, `.stl`, `.off`）から多視点レンダリング画像を生成し、DINOv2 (ViT) で特徴を抽出、視点統合して標本ごとの埋め込みを作成するPythonパイプラインです。作成した埋め込みは、近傍検索（kNN）やクラスタリングに利用できます。

## このリポジトリは何をするか

FossilViTは、次の流れで「3D形状 → 数値ベクトル化 → 解析」を行います。

1. **多視点レンダリング**: 3D標本を複数視点から2D画像化
2. **DINOv2特徴抽出**: 各視点画像をViTで特徴ベクトル化
3. **視点統合（pooling）**: 視点ごとの特徴を標本単位に集約
4. **標本埋め込みの活用**:
   - 近傍検索（類似標本の検索）
   - クラスタリング（構造発見）

## 用語解説（最低限）

- **多視点レンダ**: 1つの3D標本を複数方向から描画し、複数画像を得る処理。
- **DINOv2 (ViT)**: 画像から高次元特徴を抽出する自己教師ありVision Transformerモデル。
- **特徴ベクトル `[D]`**: 1枚の画像から得られる長さ`D`の数値ベクトル。
- **標本特徴 `[V, D]`**: 1標本について`V`視点分の特徴を並べた行列。
- **視点統合（pooling）**: `[V, D]`を1本の`[D]`に要約する処理（mean/maxなど）。
- **埋め込み（embedding）**: 標本を表す最終的な特徴ベクトル。
- **クラスタリング**: ラベルなしで似た標本同士をグループ化する処理。
- **近傍検索**: ある標本に近い埋め込みを距離計算で探す処理。
- **PCA**: 次元削減手法。情報をなるべく保ったまま次元を圧縮する。
- **寄与率（explained variance）**: PCA各成分が保持する分散の割合。累積値で情報保持率を確認できる。

## プロジェクト構成

```text
project_root/
  data/meshes/
  data/renders/
  data/features/
  data/embeddings/
  results/
  src/
    render_multiview.py
    extract_features.py
    pool_embeddings.py
    cluster.py
    search.py
    utils/
      io.py
      geometry.py
      vision.py
  configs/default.yaml
  requirements.txt
  README.md
```

## セットアップ

### 1) PyTorchを環境に合わせてインストール

PyTorchはGPU/CUDAバージョンとの整合が重要なため、必ず公式の **Get Started** でコマンドを生成してインストールしてください。

- https://pytorch.org/get-started/locally/

例（CPU版の一例。実際は公式ページで選択して生成したコマンドを利用してください）:

```bash
pip install torch torchvision
```

### 2) その他ライブラリをインストール

```bash
pip install -r requirements.txt
```

または個別:

```bash
pip install open3d scikit-learn hdbscan timm numpy pandas Pillow tqdm PyYAML
```

## 実行例（CLI）

> すべて `python -m src.<module>` 形式で実行できます。

### A. 必須: 埋め込み生成

#### 1) 多視点レンダ

```bash
python -m src.render_multiview --in data/meshes --out data/renders --views 12 --size 768
```

#### 2) DINOv2特徴抽出

```bash
python -m src.extract_features --renders data/renders --out data/features --model dinov2_vits14 --device cuda
```

- DINOv2は `torch.hub.load('facebookresearch/dinov2', '<model_name>')` でロードされます（例: `dinov2_vits14`）。
- 初回実行時は `torch.hub` によるモデルダウンロードが発生するため、ネットワーク接続が必要です。

#### 3) 視点統合（標本埋め込み生成）

```bash
python -m src.pool_embeddings --features data/features --out data/embeddings --pool mean
```

### B. オプション（推奨）: 近傍検索

```bash
QUERY_ID=$(head -n 1 data/embeddings/ids.txt)
python -m src.search --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --query "$QUERY_ID" --topk 10 --metric cosine --out results
```

### C. オプション: クラスタリング

#### 1) PCAなし（まずは素で試す）

```bash
python -m src.cluster --emb data/embeddings/embeddings.npy --out results --method hdbscan
```

#### 2) PCAあり（ノイズが多い/不安定なら推奨）

- 固定次元（例: 64次元）

```bash
python -m src.cluster --emb data/embeddings/embeddings.npy --out results --method hdbscan --pca 64 --pca_report results/pca_report.csv
```

- 累積寄与率しきい値（例: 95%以上を満たす最小次元数を自動採用）

```bash
python -m src.cluster --emb data/embeddings/embeddings.npy --out results --method hdbscan --pca 0.95 --pca_report results/pca_report.csv
```

`--pca` は次のように解釈されます。

- `--pca` 未指定 / `--pca 0`: PCAなし
- `--pca 64`: 固定64次元
- `--pca 0.95`: 累積寄与率95%以上になる最小次元数を自動採用

`--pca_report` を指定すると、`component, explained_variance_ratio, cumulative` を含むCSVが保存されます。

## FAQ

### PCAは必須ですか？

**必須ではありません。**

- **近傍検索（kNN）**: 基本的にPCAなしで問題ありません。元の特徴情報を削らずに距離計算できます。
- **クラスタリング**: PCAが有効なケースがあります。高次元空間では距離・密度が不安定になりやすく、適度な次元削減でクラスタが見やすくなることがあります。
- **寄与率の監視**: 必須ではありませんが、次元削減が過剰でないかを検知する安全柵として有用です。`--pca_report`でログを残す運用を推奨します。

## 生成物

- `data/renders/**/*.png`: 各標本の多視点レンダ画像（入力メッシュのカテゴリ階層を維持し、`<specimen_id>_viewXX.png`で保存）
- `data/features/**/*.npy`: 標本単位の特徴行列 `[V, D]`（レンダ画像と同じカテゴリ階層を維持）
- `data/embeddings/**/*.npy`: 視点統合後の埋め込み `[D]`（featuresの階層を維持）
- `data/embeddings/embeddings.npy`: 全標本の埋め込み `[N, D]`
- `data/embeddings/ids.txt`: `embeddings.npy` の行順に対応するID
- `results/clusters.csv`: クラスタリング結果（`specimen_id, cluster_id, prob/score`）
- `results/knn_<query_id>.csv`: 近傍検索結果（`query_id, neighbor_id, distance`）
- `results/pca_report.csv`: PCAレポート（`component, explained_variance_ratio, cumulative`）

## 実装上のポイント

- Open3D `OffscreenRenderer` は、**ループ内で生成/破棄を繰り返すとメモリが増える報告がある**ため、次のいずれかを推奨します。
  - 1プロセス内でrendererを使い回す設計
  - 1標本1プロセスでレンダリングする設計
- 背景白、固定ライト、bbox基準でセンタリング+スケール正規化。
- DINOv2は `torch.hub.load("facebookresearch/dinov2", model)` で読み込み、`model.eval()` + `torch.inference_mode()` で凍結推論。
- 例外時は処理をスキップしログを残すため、全体ジョブが途中で落ちにくい構成。
- `tqdm` による進捗表示付き。

## 環境依存の注意（WSL2含む）

- Open3Dのオフスクリーンレンダは環境依存で、EGL/OSMesaなどの設定が必要になる場合があります。
- WSL2では、必要な場合のみ次を設定してください。

```bash
export GALLIUM_DRIVER=d3d12
```

## 補足

- `--device auto` を指定すると、CUDAが使える環境ではGPU、無ければCPUを自動選択します。
- 入力ディレクトリはサブディレクトリまで再帰的に探索します。
- 乱数シードは各CLIの `--seed` で固定できます。
