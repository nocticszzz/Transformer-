import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import argparse
from model import TransformerClassifier
from dataset import IMDbDataset, build_vocab, create_padding_mask

# === 默认配置 ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            mask = create_padding_mask(data).to(DEVICE)
            output = model(data, mask)
            loss = criterion(output, target)
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), correct / total

def get_args():
    parser = argparse.ArgumentParser()
    # 实验变量
    parser.add_argument('--n_layers', type=int, default=2, help='Encoder Layer的数量')
    parser.add_argument('--n_heads', type=int, default=2, help='Multi-head Attention的头数')
    parser.add_argument('--d_model', type=int, default=64, help='隐藏层维度')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数(消融实验可以少跑几轮)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    return parser.parse_args()

def main():
    args = get_args()
    print(f"Starting Experiment with: Layers={args.n_layers}, Heads={args.n_heads}, Dim={args.d_model}")

    # 1. 加载数据
    print("Loading Data...")
    df = pd.read_csv('./data/IMDB Dataset.csv')
    raw_texts = df['review'].tolist()
    raw_labels = [1 if sentiment == 'positive' else 0 for sentiment in df['sentiment'].tolist()]
    
    # 词表 
    vocab = build_vocab(raw_texts[:20000], max_size=10000)
    
    # 缩小数据集规模以加快实验速度 
    full_dataset = IMDbDataset(raw_texts[:20000], raw_labels[:20000], vocab)
    
    # 划分
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. 初始化模型
    model = TransformerClassifier(
        vocab_size=len(vocab), 
        d_model=args.d_model, 
        n_heads=args.n_heads, 
        d_ff=args.d_model * 4, # 通常 d_ff 是 d_model 的 4 倍
        n_layers=args.n_layers, 
        n_classes=2
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 3. 训练 (简化版，只打印结果)
    print("Training...")
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            mask = create_padding_mask(data).to(DEVICE)
            optimizer.zero_grad()
            output = model(data, mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 每轮测一次
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            
    print(f"Final Best Acc: {best_acc:.4f}")
    # 将结果追加到日志文件
    with open("ablation_results.txt", "a") as f:
        f.write(f"Layers={args.n_layers}, Heads={args.n_heads} -> Acc={best_acc:.4f}\n")

if __name__ == "__main__":
    main()