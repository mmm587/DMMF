import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import *


class Attention_multi_gate(nn.Module):
    def __init__(self, hidden_size):
        super(Attention_multi_gate, self).__init__()

        # 隐藏层 hv:v+t-> t   ha:a+t-> t
        self.W_hv = nn.Linear(VISUAL + TEXT, TEXT)
        self.W_ha = nn.Linear(ACOUSTIC + TEXT, TEXT)
        # v:v-> t  a:a-> t
        self.W_v = nn.Linear(VISUAL, TEXT)
        self.W_a = nn.Linear(ACOUSTIC, TEXT)
        self.scaling_factor = args.scaling_factor

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(args.drop)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        weight_v = F.gelu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.gelu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
        if args.Use_EFusion:
            em_norm = text_embedding.norm(1, dim=-1)
            hm_norm = h_m.norm(1, dim=-1)
            thresh_hold = torch.pow(em_norm / (hm_norm + eps), 1 / 3) * self.scaling_factor
            embedding_output = self.dropout(
                self.LayerNorm((h_m * thresh_hold.unsqueeze(dim=-1) + 1) * text_embedding)
            )
        else:
            embedding_output = text_embedding
        return embedding_output

