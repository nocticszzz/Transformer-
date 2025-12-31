import torch
import pandas as pd
from model import TransformerClassifier
from collections import Counter

# ==========================================
# 1. 准备工作：重新构建词表 & 文本处理函数
# ==========================================
print("正在重建词表 (需与训练时一致)...")
def build_vocab(texts, max_size=10000):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counts.most_common(max_size):
        vocab[word] = len(vocab)
    return vocab

def text_pipeline(text, vocab, max_len=200):
    # 将句子转为数字序列
    tokens = text.lower().split()
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0) # 增加 batch 维度

# 读取数据来重建词表
df = pd.read_csv('./data/IMDB Dataset.csv') 
raw_texts = df['review'].tolist() 
print("词表构建完成！")

# ==========================================
# 2. 加载模型
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型结构 
model = TransformerClassifier(
    vocab_size=len(vocab),
    d_model=64,    
    n_heads=2, 
    d_ff=128, 
    n_layers=2, 
    n_classes=2
).to(DEVICE)

# 加载 .pth 权重文件
model_path = "best_model.pth" 
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval() 
print(f"成功加载模型: {model_path}")

# ==========================================
# 3. 定义预测函数
# ==========================================
def predict_sentiment(sentence):
    tensor_input = text_pipeline(sentence, vocab).to(DEVICE)
    
    with torch.no_grad():
      
        output = model(tensor_input, mask=None)
        
        # 获取概率
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    label = "Positive (好评)" if predicted_class == 1 else "Negative (差评)"
    return label, confidence

# ==========================================
# 4. 开始测试
# ==========================================
if __name__ == "__main__":
    print("\n=== 模型测试 ===")
    
    test_sentences = [
        "This movie is absolutely amazing and the plot is fantastic.",  # 明显好评
        "I hate this film, it is a complete waste of time.",          # 明显差评
        "The acting was okay but the story was boring.",              # 混合评价
        "I really enjoyed the cinematography.",                       # 细节好评
        "Don't watch this garbage."                                   # 强烈差评
    ]
    
    for text in test_sentences:
        label, conf = predict_sentiment(text)
        print(f"\n评论: {text}")
        print(f"预测: {label}")
        print(f"置信度: {conf:.4f}")