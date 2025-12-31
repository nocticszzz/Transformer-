import torch
import time
from model import TransformerClassifier

# === 配置  ===
VOCAB_SIZE = 10002
D_MODEL = 64
N_HEADS = 2
D_FF = 128
N_LAYERS = 2
N_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_model():
    print(f"正在分析模型效率 (Device: {DEVICE})...")
    
    # 1. 初始化模型
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, 
        d_ff=D_FF, n_layers=N_LAYERS, n_classes=N_CLASSES
    ).to(DEVICE)
    model.eval()

    # 2. 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[1] 参数量统计:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 约合: {total_params/1e6:.2f} M (百万)")

    # 3. 测试推理速度
    print(f"\n[2] 推理速度测试 (模拟 1000 条数据)...")
    # 模拟一条输入 (Batch Size = 1, Length = 200)
    dummy_input = torch.randint(0, VOCAB_SIZE, (1, 200)).to(DEVICE)
    
    # 预热 
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # 正式计时
    start_time = time.time()
    n_samples = 1000
    with torch.no_grad():
        for _ in range(n_samples):
            _ = model(dummy_input)
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = n_samples / total_time
    
    print(f"  - 处理 {n_samples} 条数据耗时: {total_time:.4f} 秒")
    print(f"  - 推理速度 (FPS): {fps:.2f} samples/sec")
    print(f"  - 平均每条耗时: {total_time/n_samples*1000:.2f} ms")

if __name__ == "__main__":
    analyze_model()