import os
import torch
import csv
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# --- 1. 資料準備 ---
PATH_DATASETS = "" 

train_ds = MNIST(PATH_DATASETS, train=True, download=True, 
                 transform=transforms.ToTensor())
test_ds = MNIST(PATH_DATASETS, train=False, download=True, 
                 transform=transforms.ToTensor())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 設定參數
epochs = 5
lr = 0.1
BATCH_SIZE = 1024
csv_filename = 'output/training_results.csv' 

# --- 檢查並自動建立目錄 ---
output_dir = os.path.dirname(csv_filename)
if output_dir: 
    os.makedirs(output_dir, exist_ok=True)
    print(f"已檢查/建立目錄: {output_dir}")

# --- 2. 建立 CSV 並寫入標題 ---
# 加入 'Loss' 欄位
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Run', 'Accuracy', 'Loss'])  # <--- 修改這裡：增加 Loss 標題

    # --- 3. 開始執行 10 次迴圈 ---
    for run in range(1, 11):
        print(f'\n========== 第 {run} 次訓練開始 ==========')
        
        train_loader = DataLoader(train_ds, batch_size=600, shuffle=True)
        test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE)

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 256), 
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 10), 
        ).to(device)

        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss() # 注意：預設是計算平均(Mean)

        # --- 訓練階段 ---
        model.train()
        for epoch in range(1, epochs + 1):
            loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            print(f'Run {run} - Epoch {epoch}: Loss {loss.item():.6f}')

        # --- 測試階段 ---
        model.eval()
        test_loss = 0
        correct = 0
        
        # 為了計算正確的 total loss，通常建議使用 reduction='sum' 或是手動乘回 batch size
        # 但為了保持與您原本邏輯一致，這邊沿用原本的累加方式
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # 累加 loss (注意：這裡累加的是每個 batch 的 mean loss)
                test_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()

        # 計算結果
        test_loss /= len(test_loader.dataset) # 平均損失 (依照您原本的公式)
        data_count = len(test_loader.dataset)
        acc_percentage = 100. * correct / data_count

        print(f'Run {run} 結果 -> 平均損失: {test_loss:.4f}, 準確率: {acc_percentage:.2f}%')
        
        # --- 4. 寫入 CSV ---
        # 加入 test_loss 到寫入的列表中
        writer.writerow([run, acc_percentage, test_loss]) # <--- 修改這裡：寫入 Loss 值

print(f"\n所有訓練完成，結果已儲存至 {csv_filename}")