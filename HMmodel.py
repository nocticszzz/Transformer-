class SDOA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SDOA, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, Q, K, V, MASK=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        if MASK is not None:
            scores = scores.masked_fill(MASK == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output