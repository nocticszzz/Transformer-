import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 1. 简单的分词与词表构建 (模拟)
def build_vocab(texts, max_size=10000):
    # 简单按空格分词，统计词频
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())
    
    # 创建词表: {word: index}
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counts.most_common(max_size):
        vocab[word] = len(vocab)
    return vocab

def text_pipeline(text, vocab, max_len=200):
    # 将文本转换为索引序列
    tokens = text.lower().split()
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    
    # 截断或填充
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
        
    return torch.tensor(indices, dtype=torch.long)

# 2. 自定义 Dataset
class IMDbDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # 返回: (token_ids, label)
        token_ids = text_pipeline(self.texts[idx], self.vocab, self.max_len)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return token_ids, label

# 3. 关键：生成 Padding Mask
def create_padding_mask(seq, pad_idx=0):
    # seq: [batch, seq_len]
    # mask: [batch, seq_len] (1 for real token, 0 for pad)
    mask = (seq != pad_idx).float() 
    return mask