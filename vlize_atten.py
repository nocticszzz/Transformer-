import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from model import TransformerClassifier

# ==========================================
# 1. 配置与准备
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "best_model.pth"  # 确保你有这个文件
CSV_PATH = "./data/IMDB Dataset.csv" # 确保路径正确

# 模拟 Tokenizer (必须与训练时一致)
def build_vocab(texts, max_size=10000):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counts.most_common(max_size):
        vocab[word] = len(vocab)
    return vocab

def text_pipeline(text, vocab, max_len=200):
    tokens = text.lower().split()
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    # 不做 Padding，为了可视化好看，直接用真实长度
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0), tokens

# ==========================================
# 2. 初始化环境
# ==========================================
print("正在加载词表 (这可能需要几秒钟)...")
df = pd.read_csv(CSV_PATH)
raw_texts = df['review'].tolist()
vocab = build_vocab(raw_texts, max_size=10000)
print("词表加载完成。")

# 加载模型
model = TransformerClassifier(
    vocab_size=len(vocab), d_model=64, n_heads=2, d_ff=128, n_layers=2, n_classes=2
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("模型加载完成。")

# ==========================================
# 3. 核心：手动提取注意力的函数
# ==========================================
def get_attention_weights(model, text):
    """
    手动执行模型的前向传播，只为了拿到最后一层的 Attention
    """
    # 处理输入
    input_tensor, tokens = text_pipeline(text, vocab)
    input_tensor = input_tensor.to(DEVICE)
    
    with torch.no_grad():
        # 1. Embedding
        x = model.embedding(input_tensor)
        
        # 2. Add [CLS]
        batch_size = x.size(0)
        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Positional Encoding
        x = model.pos_encoder(x)
        
        # 4. Pass through Encoder Layers
        # 关键点：我们要把每一层的 attention 存下来
        all_attentions = []
        for layer in model.layers:
            # EncoderLayer.forward 返回 (x, attn_weights)
            x, attn = layer(x, mask=None) 
            all_attentions.append(attn)
            
        # 取最后一层的注意力权重
        # Shape: [batch, n_heads, seq_len, seq_len]
        final_layer_attn = all_attentions[-1]
        
    return final_layer_attn, ["<CLS>"] + tokens

# ==========================================
# 4. 绘图
# ==========================================
def plot_attention(text):
    print(f"正在分析句子: '{text}' ...")
    
    # 获取权重
    attn_weights, tokens = get_attention_weights(model, text)
    
    # attn_weights shape: [1, 2, seq_len, seq_len] (batch=1, heads=2)
    # 我们取第一个样本，并把所有头(Heads)的权重平均一下，或者只看第0个头
    # 这里我们选择：平均所有头的注意力 (更能代表整体关注点)
    attn_avg = attn_weights[0].mean(dim=0).cpu().numpy() 
    
    # 绘图
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_avg, 
                xticklabels=tokens, 
                yticklabels=tokens, 
                cmap="viridis", 
                annot=True,    # 显示数值
                fmt=".2f",     # 数值保留2位小数
                cbar=True)
    
    plt.title("Self-Attention Heatmap (Avg of Heads)")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    # 保存图片
    save_name = "attention_map.png"
    plt.savefig(save_name)
    print(f"热力图已保存为: {save_name}")
    plt.show()

# ==========================================
# 5. 运行
# ==========================================
if __name__ == "__main__":
    target_sentence = "The movie is amazing"
    plot_attention(target_sentence)