import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt # 用于画图
import copy

# 引用你的模块
from model import TransformerClassifier
from dataset import IMDbDataset, build_vocab, create_padding_mask

# === 配置参数 ===
BATCH_SIZE = 16    # T400 显卡建议
EPOCHS = 10        # 多跑几轮，靠验证集来决定什么时候停
LR = 0.0005        # 稍微调小一点学习率
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, loader, criterion):
    """验证函数：只计算准确率和Loss，不更新梯度"""
    model.eval() # 切换到评估模式
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad(): # 不计算梯度，省显存
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            mask = create_padding_mask(data).to(DEVICE)
            
            output = model(data, mask)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # 计算准确率
            preds = torch.argmax(output, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            
    return total_loss / len(loader), correct / total

def main():
    print(f"Running on {DEVICE}")

    # 1. 读取数据
    print("Loading Data...")
    df = pd.read_csv('./data/IMDB Dataset.csv') # 确保路径对
    raw_texts = df['review'].tolist()
    # 转换标签: positive->1, negative->0
    raw_labels = [1 if sentiment == 'positive' else 0 for sentiment in df['sentiment'].tolist()]
    
    # 2. 构建词表 (仅使用前 40000 条数据构建词表，防泄漏，这里为了简便全量构建也可以)
    vocab = build_vocab(raw_texts, max_size=10000)
    full_dataset = IMDbDataset(raw_texts, raw_labels, vocab)
    
    # 3. 划分数据集 (8:1:1)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len) # 40000
    val_len = int(0.1 * total_len)   # 5000
    test_len = total_len - train_len - val_len # 5000
    
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])
    
    print(f"Data Split -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False) # 验证集不需要 shuffle
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化模型
    model = TransformerClassifier(
        vocab_size=len(vocab), d_model=64, n_heads=2, d_ff=128, 
        n_layers=2, n_classes=2, dropout=0.1
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # === 记录曲线数据的列表 ===
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0 # 记录最高准确率
    
    print("Start Training...")
    
    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            mask = create_padding_mask(data).to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data, mask)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸，大作业加分项 [cite: 34])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- 验证阶段 (决定当前模型好不好) ---
        avg_val_loss, avg_val_acc = evaluate(model, val_loader, criterion)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {avg_val_acc:.4f}")
        
        # --- 核心：保存最佳模型 ---
        # 如果当前验证集准确率 超过了 历史最佳
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  >>> New Best Model Saved! (Acc: {best_val_acc:.4f})")
            
    # 5. 训练结束，绘制曲线 [cite: 35]
    print("\nTraining Finished. Plotting curves...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('learning_curve.png') # 保存图片用于写报告
    print("Learning curve saved as 'learning_curve.png'")
    
    # 6. 最后用测试集跑一次最终分数
    # 加载刚才保存的最好模型
    model.load_state_dict(torch.load("best_model.pth"))
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nFinal Test Accuracy (on Best Model): {test_acc:.4f}")

if __name__ == "__main__":
    main()