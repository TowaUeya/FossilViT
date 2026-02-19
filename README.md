# FossilViT: 化石3Dデータの多視点DINOv2埋め込みパイプライン

化石の3Dメッシュ/点群（`.ply`, `.obj`, `.stl`, `.off`）から多視点レンダリング画像を生成し、DINOv2 (ViT) 特徴を抽出、視点統合して標本ごとの埋め込みを作成し、クラスタリングと近傍検索を行うPythonパイプラインです。

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

```bash
GALLIUM_DRIVER=d3d12
python -m src.render_multiview --in data/meshes --out data/renders --views 12 --size 768
python -m src.extract_features --renders data/renders --out data/features --model dinov2_vits14 --device cuda
python -m src.pool_embeddings --features data/features --out data/embeddings --pool mean
python -m src.cluster --emb data/embeddings/embeddings.npy --out results --method hdbscan --pca 64
python -m src.search --emb data/embeddings/embeddings.npy --ids data/embeddings/ids.txt --query 000 --topk 10 --metric cosine --out results
```

### 補足

- `--device auto` を指定すると、CUDAが使える環境ではGPU、無ければCPUを自動選択します。
- すべてのスクリプトは `python -m src.<module>` で実行可能です（`__main__` 対応）。
- 入力ディレクトリはサブディレクトリまで再帰的に探索します。
- 乱数シードは各CLIの `--seed` で固定できます。

## 生成物

- `data/renders/**/*.png`: 各標本の多視点レンダ画像（入力メッシュのカテゴリ階層を維持し、`<specimen_id>_viewXX.png`で保存）
- `data/features/**/*.npy`: 標本単位の特徴行列 `[V, D]`（レンダ画像と同じカテゴリ階層を維持）
- `data/embeddings/**/*.npy`: 視点統合後の埋め込み `[D]`（featuresの階層を維持）
- `data/embeddings/embeddings.npy`: 全標本の埋め込み `[N, D]`
- `data/embeddings/ids.txt`: `embeddings.npy` の行順に対応するID
- `results/clusters.csv`: クラスタリング結果（`specimen_id, cluster_id, prob/score`）
- `results/knn_<query_id>.csv`: 近傍検索結果（`query_id, neighbor_id, distance`）

## 実装上のポイント

- Open3D `OffscreenRenderer` は標本ループで使い回し、メモリ増加リスクを抑える設計。
- 背景白、固定ライト、bbox基準でセンタリング+スケール正規化。
- DINOv2は `torch.hub.load("facebookresearch/dinov2", model)` で読み込み、`model.eval()` + `torch.inference_mode()` で凍結推論。
- 例外時は処理をスキップしログを残すため、全体ジョブが途中で落ちにくい構成。
- `tqdm` による進捗表示付き。

## 注意

- 初回DINOv2実行時は `torch.hub` がモデルをダウンロードするためネットワーク接続が必要です。
- Open3Dのオフスクリーンレンダは環境によりEGL/OSMesa設定が必要になる場合があります（Linux/WSL2）。
