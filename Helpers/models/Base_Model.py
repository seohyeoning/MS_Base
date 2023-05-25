import math
from numpy import size
import torch
from torch import nn
import torch.nn.functional as F

########## Feature Extraction
class sub_fe(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 20,
                              kernel_size = 10, stride=1)
        self.conv2 = nn.Conv1d(in_channels = 20, out_channels = 40,
                              kernel_size = 5, stride=1)
        self.conv3 = nn.Conv1d(in_channels = 40, out_channels = 80,
                              kernel_size = 3, stride=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.15)

        self.lstm1 = nn.LSTM(80, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dense = nn.Linear(in_features=32, out_features=8)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = torch.permute(x, (0, 2, 1)) 
        
        x, _ = self.lstm1(x) # Batch, sequence, input_size(channel)
        x, _ = self.lstm2(x)

        ### 마지막 시퀀스 정보만 유지하는 경우 사용
        # x = x[:, -1, :].unsqueeze(1) # return_sequences=False -> last sequence 
        
        x = self.dense(x)
        return x


############ Cross Attention
class CrossAttn(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x1, x2):
        # x1: (seq_len1, batch_size, hidden_dim)
        # x2: (seq_len2, batch_size, hidden_dim)

        seq_len1, batch_size, _ = x1.size()
        seq_len2, _, _ = x2.size()

        # Reshaping x1 and x2 for attention calculation
        q1 = self.W_q(x1)  # (batch_size, seq_len1, hidden_dim)
        k2 = self.W_k(x2)  # (batch_size, seq_len2, hidden_dim)
        v2 = self.W_v(x2)  # (batch_size, seq_len2, hidden_dim)

        q2 = self.W_q(x2)  # (batch_size, seq_len2, hidden_dim)
        k1 = self.W_k(x1)  # (batch_size, seq_len1, hidden_dim)
        v1 = self.W_v(x1)  # (batch_size, seq_len1, hidden_dim)

        # Calculating attention weights and performing attention
        attended_x1 = self.attention(q1, k2, v2)  # (batch_size, seq_len1, hidden_dim)
        attended_x2 = self.attention(q2, k1, v1)  # (batch_size, seq_len2, hidden_dim)

        return attended_x1, attended_x2

    def attention(self, query, key, value):
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K^T, (batch_size, seq_len1, seq_len2)
        attention_score = attention_score / math.sqrt(d_k)
        attention_prob = F.softmax(attention_score, dim=-1)  # (batch_size, seq_len1, seq_len2)
        out = torch.matmul(attention_prob, value)  # (batch_size, seq_len1, hidden_dim)
        return out

########## Tensor Fusion
"""
Fusion.
감마 값은 1로 둔 baseline 모델
"""
class tfn(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.15)
        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, 3)
        
        
    def forward(self, inputs):
        x1, x2 = inputs
        # x1 = torch.matmul(x1, 감마) # weighted_x1 만들기

        fusion = torch.matmul(x1.transpose(1,2), x2)  # (batch_size, hidden_dim(8), hidden_dim(8))
        fusion = torch.reshape(fusion, (fusion.shape[0], -1)) #(16,64)
        
        fusion = self.dropout(fusion)
        fusion = self.dense1(fusion)
        fusion = self.dropout(fusion)
        fusion = self.dense2(fusion)
        output = F.softmax(fusion, dim=1)

        return output

########## New model
class AMFTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_fe1 = sub_fe(28)
        self.sub_fe2 = sub_fe(1)
        self.cross_attn = CrossAttn(8)
        self.tfn = tfn()

    def forward(self, inputs):
        (xe, xc) = (inputs[0], inputs[1])
        (xe_fe, xc_fe) = self.sub_fe1(xe), self.sub_fe2(xc)
        (xe_attn, xc_attn) = self.cross_attn(xe_fe, xc_fe)
        output = self.tfn([xe_attn, xc_attn])

        return output

