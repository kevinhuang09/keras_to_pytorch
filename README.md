# keras_to_pytorch

> 📚 **課堂練習紀錄**
> 本專案為 huang 的課堂練習作品，用途為個人學習與練習紀錄。內容示範如何以 **PyTorch** 重現原本在 Keras 中常見的 MNIST 手寫數字分類流程，屬於練習性質，非正式產品程式碼。

---

## 專案簡介

本專案是一個以 **PyTorch** 撰寫的 MNIST 手寫數字辨識練習，重點在於將 Keras 中熟悉的建模、訓練與評估流程，轉換成 PyTorch 的寫法。整體模型採用簡單的全連接神經網路（MLP），適合作為深度學習入門與框架轉換的練習範例。

---

## 檔案結構

```
keras_to_pytorch/
├── MNIST/raw/          # 下載後的 MNIST 原始資料集
├── output/             # 訓練結果輸出資料夾（CSV 紀錄等）
├── main.py             # 單次訓練 + 評估的主程式
├── t1.py               # 進階版：重複訓練 10 次並將結果記錄到 CSV
└── requirement.txt     # 專案相依套件清單
```

---

## 程式說明

### `main.py` — 基本訓練與評估

執行一次完整的訓練與測試流程：

- **資料集**：MNIST 手寫數字（自動下載，轉換為 Tensor）
- **模型架構**（`torch.nn.Sequential`）：
  - `Flatten` — 將 28×28 影像展平為 784 維向量
  - `Linear(784, 256)` — 第一層全連接
  - `Dropout(0.2)` — 防止過擬合
  - `Linear(256, 10)` — 輸出層（對應數字 0–9）
- **損失函數**：`CrossEntropyLoss`（因此輸出層後不需再加 softmax）
- **優化器**：`Adadelta`
- **訓練流程**：跑 5 個 epoch，訓練過程中每 10 個 batch 印出一次 loss
- **評估流程**：在測試集上計算平均損失與準確率

### `t1.py` — 多次訓練並記錄結果

在 `main.py` 的基礎上進一步擴充：

- 自動建立輸出資料夾（若不存在）
- 建立 CSV 檔並寫入表頭：`Run`, `Accuracy`, `Loss`
- **重複訓練 10 次**，每次都重新初始化模型與優化器
- 每次訓練完成後，將該次的準確率與損失寫入 CSV
- 方便觀察多次訓練結果的穩定度與差異

---

## 主要超參數

| 參數 | 數值 |
|------|------|
| Epochs | 5 |
| Learning Rate | 0.1 |
| Batch Size | 1024（測試）/ 600（訓練，於 `main.py`） |
| Optimizer | Adadelta |
| Loss | CrossEntropyLoss |
| 重複次數（`t1.py`） | 10 |

---

## 環境需求

主要相依套件（完整清單見 `requirement.txt`）：

- `torch==2.4.1`
- `torchvision`
- `torchmetrics`
- Python 3.x

安裝方式：

```bash
pip install -r requirement.txt
```

> 註：`requirement.txt` 包含完整的開發環境套件（數量較多），若只想跑本練習，安裝 `torch`、`torchvision`、`torchmetrics` 即可。

---

## 執行方式

```bash
# 單次訓練與評估
python main.py

# 重複訓練 10 次並輸出 CSV 紀錄
python t1.py
```

首次執行會自動下載 MNIST 資料集至 `MNIST/raw/`。

---

## 學習重點（練習目標）

- 熟悉 PyTorch 的資料載入（`DataLoader`）、模型定義（`nn.Sequential`）與訓練迴圈
- 理解 Keras 高階 API 與 PyTorch 手動訓練迴圈之間的對應關係
- 練習模型評估、準確率計算，以及將實驗結果記錄到檔案

---

*本 README 為課堂練習紀錄用途，內容整理自專案原始程式碼。*