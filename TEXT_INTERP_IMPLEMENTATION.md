# Text Interpretation Guidance Module - 実装ドキュメント

## 実装概要

Transformer encoder の出力を「文字解釈に寄せる」補助モジュールを追加しました。
これにより、decoder が文字 ID 予測タスクを補助loss として学習することで、
visual features が文字情報に沿った表現になるよう誘導します。

## 主要な変更点

### 1. **TextInterpretationHead クラス (models/detr.py)**

```python
class TextInterpretationHead(nn.Module):
    """Text interpretation guidance head for decoder output refinement."""
    
    def __init__(self, hidden_dim, text_vocab_size):
        # LayerNorm + Linear(hidden_dim -> vocab_size)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_vocab = nn.Linear(hidden_dim, text_vocab_size)
    
    def get_text_logits(self, decoder_output):
        # [B, Q, hidden_dim] -> [B, Q, vocab_size]
```

**役割**:
- decoder 出力 [B, Q, d_model] から文字分類ロジット [B, Q, vocab_size] を生成
- 各 query position について、文字 ID を予測する auxiliary task

### 2. **DETR クラス修正 (models/detr.py)**

#### __init__ メソッド
```python
if self.use_text_queries:
    # ...
    self.text_interp_head = TextInterpretationHead(
        hidden_dim=hidden_dim,
        text_vocab_size=text_vocab_size,
    )
```

#### forward メソッド
```python
if self.use_text_queries and hasattr(self, 'text_interp_head'):
    final_hs = hs[-1].permute(1, 0, 2)  # [B, Q, d_model]
    pred_text_logits = self.text_interp_head.get_text_logits(final_hs)
    out['pred_text_logits'] = pred_text_logits
```

**流れ**:
1. text_query_encoder で text sequence を処理 -> query embedding
2. transformer encoder-decoder で visual features と text embedding を融合
3. decoder 最終層出力から text classification logits を抽出
4. `out['pred_text_logits']` として返す -> criterion で loss 計算

### 3. **SetCriterionAligned 修正 (models/detr.py)**

#### _loss_text_interp メソッド
```python
def _loss_text_interp(self, pred_text_logits, targets):
    # pred_text_logits: [B, Q, vocab_size]
    # targets: token_ids を含む target dicts
    
    # Query-token の対応付け（aligned supervision）
    # 最初の N 個の query を最初の N 個の token にマッピング
    # 超過分（N > Q）は無視、不足分（N < Q）はマスク
    
    # Cross-entropy loss を valid position のみで計算
    loss = F.cross_entropy(logits[mask], labels[mask])
```

#### forward メソッド
```python
def forward(self, outputs, targets):
    losses = self._loss_boxes_aligned(outputs['pred_boxes'], targets)
    
    # Add text interpretation loss if available
    if 'pred_text_logits' in outputs:
        text_losses = self._loss_text_interp(outputs['pred_text_logits'], targets)
        losses.update(text_losses)
```

**損失計算**:
- 各 query position ごとに文字 ID を予測する auxiliary loss
- padding token は loss から除外
- bbox loss と独立して計算し、重み付けして統合

### 4. **weight_dict 統合 (models/detr.py build 関数)**

```python
if bbox_only:
    weight_dict = {
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_query_dup': args.query_dup_coef,
        'loss_text_interp': getattr(args, 'text_interp_coef', 1.0),
    }
```

### 5. **コマンドライン引数追加 (main.py)**

```python
parser.add_argument('--text_interp_coef', default=1.0, type=float,
                    help='Text interpretation guidance loss coefficient')
```

## トレーニング実行方法

```bash
python main.py \
    --dataset_file kuzushiji_text \
    --kuzushiji_path <path_to_dataset> \
    --use_text_queries \
    --bbox_only \
    --text_interp_coef 1.0 \
    --query_dup_coef 0.2 \
    --output_dir <output_dir>
```

## 損失構成

### 最終損失計算

```
L_total = w_bbox * L_bbox 
        + w_giou * L_giou 
        + w_dup * L_query_dup 
        + w_text_interp * L_text_interp
```

各成分：
- `L_bbox`: L1 loss for bounding box coordinates
- `L_giou`: GIoU loss for bbox corners
- `L_query_dup`: IoU-based query diversity penalty（重複予測防止）
- `L_text_interp`: Cross-entropy for text character prediction（文字情報ガイダンス新規）

## 形状説明

| Layer | Input Shape | Output Shape | 備考 |
|-------|-------------|--------------|------|
| text_query_encoder | token_ids [B, L] | query_embed [B, L, hidden_dim] | 文字列をembedding に変換 |
| transformer.encoder | visual_features + query_embed | encoder_output [HxW, B, d] | 視覚特徴と文字特徴を融合 |
| transformer.decoder | encoder_output + query_embed | hs [num_layers, Q, B, d] | 各層での decoder 出力 |
| text_interp_head | hs[-1] [B, Q, d] | pred_text_logits [B, Q, V] | 文字分類ロジット |
| criterion | pred_text_logits + targets | loss_text_interp (scalar) | Auxiliary loss 計算 |

*ここで：
- B = batch_size
- L = text sequence length（可変長、padding あり）
- Q = num_queries（固定）
- d = hidden_dim（デフォルト 256）
- V = text_vocab_size（デフォルト 65536）
- H, W = 画像の特徴マップサイズ

## 推論モード

```python
# Training: loss 有り
outputs = model(samples, text_inputs=token_ids_list)
# outputs['pred_text_logits']: [B, Q, vocab_size] (loss 計算用)

# Inference: bbox predictionのみ
outputs = model(samples, text_inputs=token_ids_list)
# outputs['pred_boxes']: [B, Q, 4] (bbox prediction)
# outputs['pred_text_logits'] は auxiliary loss なので無視可能
```

## 実装のポイント

### 1. Aligned Supervision
- Hungarian matching を使わず、最初の N 個の query を最初の N 個の token に対応
- padding 位置は mask で除外

### 2. Lightweight Design
- TextInterpretationHead は LayerNorm + 1つの Linear layer のみ
- 追加計算量を最小化

### 3. 形状管理
- decoder 出力 [num_layers, Q, B, d] から最終層抽出
- permute で [B, Q, d] に変換して text_interp_head に入力

### 4. 損失計算の安全性
- 'token_ids' がない場合は 0 loss を返す
- valid_mask で padding 位置を除外
- 全 query が無効な場合の処理

## 検証提案

トレーニング実行時に以下を確認：

```bash
# ログ確認
tail -f <output_dir>/log.txt | grep loss_text_interp

# 期待値：
# - 最初は loss_text_interp が高い（ランダム初期化）
# - 徐々に減少していく
# - loss_bbox と loss_text_interp が相補的に動く
```

## トラブルシューティング

### Issue: "loss_text_interp が出力されない"
**原因**: use_text_queries=True が設定されていない
**解決**: --use_text_queries フラグを追加

### Issue: "形状エラーが出る"
**原因**: targets に 'token_ids' がない
**解決**: dataset が 'token_ids' を返しているか確認

### Issue: "NaN loss が発生"
**原因**: token_ids に無効な vocab index が含まれている
**解決**: token_ids を [0, text_vocab_size) 範囲に clamp
