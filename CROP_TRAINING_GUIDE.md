# Kuzushiji Text Detection with Image Cropping - Training Guide

このガイドでは、4×4グリッドに分割されたクロップ画像を使用して学習を実行するコマンドをまとめています。

## 概要

- **クロップ戦略**: 1ページの画像を4×4グリッド（16セル）に分割
- **見切れ処理**: 見切れた文字は見切れたままBBox座標を推測
- **座標系**: クロップ内の正規化座標（[0,1]範囲）に変換
- **テキストシーケンス**: 各クロップに含まれる文字を抽出

---

## 実行コマンド

### 1. デフォルト設定（クロップあり、4×4グリッド）

```bash
python main.py --dataset_file kuzushiji_text --kuzushiji_path ../kuzushiji_recognition/char_sep_datas --epochs 300 --batch_size 10 --num_workers 4 --output_dir outputs/detr_kuzushiji_crop --device cuda
```

**説明:**
- `--kuzushiji_use_crop_grid True` がデフォルトで有効
- `--kuzushiji_grid_size 4` で4×4グリッド分割
- 実効バッチサイズ = 4 × 16 = 64クロップ

---

### 2. クロップなしで実行（従来の方法）

```bash
python main.py \
  --dataset_file kuzushiji_text \
  --kuzushiji_path <path_to_dataset> \
  --kuzushiji_use_crop_grid False \
  --epochs 300 \
  --batch_size 4 \
  --num_workers 4 \
  --output_dir outputs/detr_kuzushiji_no_crop \
  --device cuda
```

**説明:**
- クロップなし、全画像を入力
- バッチサイズ = 4（クロップ処理なし）

---

### 3. グリッドサイズを変更（2×2グリッド）

```bash
python main.py \
  --dataset_file kuzushiji_text \
  --kuzushiji_path <path_to_dataset> \
  --kuzushiji_grid_size 2 \
  --epochs 300 \
  --batch_size 4 \
  --num_workers 4 \
  --output_dir outputs/detr_kuzushiji_2x2 \
  --device cuda
```

**説明:**
- 2×2グリッド分割（4セル）
- 実効バッチサイズ = 4 × 4 = 16クロップ

---

### 4. メモリ不足の場合（バッチサイズ削減）

```bash
python main.py \
  --dataset_file kuzushiji_text \
  --kuzushiji_path <path_to_dataset> \
  --batch_size 1 \
  --epochs 300 \
  --num_workers 2 \
  --output_dir outputs/detr_kuzushiji_small_batch \
  --device cuda
```

**説明:**
- バッチサイズを1に削減
- 実効バッチサイズ = 1 × 16 = 16クロップ
- ワーカー数も削減してメモリ負荷を軽減

---

### 5. デバッグ用（小規模データで快速テスト）

```bash
python main.py \
  --dataset_file kuzushiji_text \
  --kuzushiji_path <path_to_dataset> \
  --kuzushiji_max_samples 10 \
  --epochs 5 \
  --batch_size 2 \
  --num_workers 0 \
  --output_dir outputs/debug \
  --device cuda
```

**説明:**
- データセット最初の10サンプルのみ使用
- 5エポックのクイックテスト
- ワーカー数0で即座に実行

---

### 6. 検証セット専用（検証テスト）

```bash
python main.py \
  --dataset_file kuzushiji_text \
  --kuzushiji_path <path_to_dataset> \
  --eval \
  --resume <path_to_checkpoint> \
  --batch_size 4 \
  --num_workers 4 \
  --output_dir outputs/eval \
  --device cuda
```

**説明:**
- `--eval`: 評価モードのみ実行（学習なし）
- `--resume`: チェックポイント指定

---

### 7. エンコーダ矯正あり（text-query + 4×4クロップ）

```bash
python main.py --dataset_file kuzushiji_text --kuzushiji_path ../kuzushiji-recognition/char_sep_datas  --kuzushiji_use_crop_grid True --kuzushiji_grid_size 4 --enc_text_coef 1.0  --text_interp_coef 1.0 --epochs 300 --batch_size 10  --num_workers 4 --output_dir outputs/detr_kuzushiji_crop_enc_guided  --device cuda
```

**説明:**
- エンコーダ出力に対する矯正ロス（`--enc_text_coef`）を追加
- text-query とクロップ学習は従来通り

---

## 重要な設定パラメータ

| パラメータ | 説明 | デフォルト | 推奨値 |
|-----------|------|-----------|-------|
| `--kuzushiji_use_crop_grid` | クロップ有効化 | `True` | `True` |
| `--kuzushiji_grid_size` | グリッド分割数 | `4` | `4` / `2` |
| `--batch_size` | バッチサイズ | `2` | `2～4` |
| `--num_workers` | データ読み込みワーカー | `2` | `4～8` |
| `--kuzushiji_max_samples` | 学習サンプル上限 | `None` | `10`（テスト時） |

---

## 実効バッチサイズの計算

```
実効バッチサイズ = --batch_size × グリッド数

例：
- batch_size=4, grid_size=4 → 4 × 16 = 64クロップ
- batch_size=2, grid_size=4 → 2 × 16 = 32クロップ
- batch_size=1, grid_size=2 → 1 × 4 = 4クロップ
```

**注意**: GPUメモリが不足する場合、バッチサイズを削減してください。

---

## 推論時のコマンド

クロップ適用の可視化スクリプト：

```bash
python visualize_predictions.py \
  --checkpoint outputs/detr_kuzushiji_crop/checkpoint.pth \
  --kuzushiji_path <path_to_dataset> \
  --split val \
  --num_samples 10 \
  --output_dir outputs/visualizations
```

---

## トレーニングログ確認

学習中のロス推移を確認：

```bash
python plot_training_losses.py \
  --log_file outputs/detr_kuzushiji_crop/log.txt \
  --output outputs/loss_plot.png \
  --show
```

---

## 注意事項

1. **メモリ要件の増加**
   - クロップにより実効バッチサイズが16倍に
   - 低解像度GPU（e.g., RTX 2070）の場合 `--batch_size 1` を推奨

2. **学習時間**
   - クロップなし: 標準時間
   - クロップあり: 実効バッチサイズが大きいため学習が高速化

3. **座標系**
   - 出力BBox: 各クロップ内での正規化座標 [0, 1]
   - 見切れた文字: 見切れたまま推測（クランプなし展開）

4. **空のクロップ**
   - 含まれます（学習データのバランスを保つため）
   - ダミーBBox: [0.5, 0.5, 1e-6, 1e-6]

---

## トラブルシューティング

### GPU メモリ不足エラー
```bash
# バッチサイズを削減
python main.py --batch_size 1 --num_workers 2 ...
```

### torch.cuda.OutOfMemory エラー
```bash
# ワーカー数も削減
python main.py --batch_size 1 --num_workers 0 ...
```

### データセット読み込みエラー
```bash
# --kuzushiji_path が正しいか確認
# CSVファイルとimagesフォルダが存在するか確認
```

---

## 実装詳細

### クロップ処理のフロー

```
入力画像（全画像ピクセル座標）
    ↓ [compute_crop_grid()]
4×4グリッド座標計算
    ↓ [bbox_intersects_crop()]
各クロップに含まれる文字抽出
    ↓ [transform_bbox_to_crop_coords()]
ピクセル座標 → クロップ内正規化座標
    ↓
モデル入力（クロップ画像 + 正規化BBox）
```

### 座標変換の詳細

```python
# クロップ外のピクセル座標 → クロップ内座標
x1_crop, y1_crop, x2_crop, y2_crop = crop_box
x_in_crop = x - x1_crop  # ローカル座標

# 正規化座標に変換
cx_norm = (x1_in_crop + x2_in_crop) * 0.5 / crop_width
cy_norm = (y1_in_crop + y2_in_crop) * 0.5 / crop_height

# 見切れた場合もそのまま推測（[0,1]でクランプ）
cx_norm = max(0.0, min(cx_norm, 1.0))
```

---

## 関連ファイル

- **データセット実装**: `datasets/kuzushiji_text.py`
- **モデル実装**: `models/detr.py`
- **メイン学習**: `main.py`
- **可視化スクリプト**: `visualize_predictions.py`

---

**最終更新**: 2026年5月14日
