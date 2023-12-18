# This code borrowed from [vit-explain](https://github.com/jacobgil/vit-explain)
# Modified by Hansae Ju

import torch
import torch.nn.functional as F
import numpy as np
from doctr.models.modules.transformer.pytorch import MultiHeadAttention

class VITAttentionRollout:
    """Rollout attention weights from ViTSTR model"""
    def __init__(self, model, head_fusion="mean", discard_ratio=0.9):
        self.model = model.eval()
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        for name, module in self.model.named_modules():
                if isinstance(module, MultiHeadAttention):
                    # print("Found MultiHeadAttention")
                    module.register_forward_hook(self.attention_hook)

    def __call__(self, input_tensor):
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
        return self.rollout()
    
    def attention_hook(self, module, input_data, output):
        self.attentions.append(module.get_attention_weights().cpu())

    def rollout(self):
        result = torch.eye(self.attentions[0].size(-1))
        with torch.no_grad():
            for attention in self.attentions:
                attention_heads_fused = self.fuse_heads(attention)
                flat = attention_heads_fused.view(-1)
                _, indices = flat.topk(int(flat.size(0) * self.discard_ratio), largest=False)
                flat[indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + I) / 2
                a = a / a.sum(dim=-1, keepdim=True)

                result = torch.matmul(a, result)

        mask = result
        mask = mask / mask.max()
        return mask.numpy()
    
    def fuse_heads(self, attention):
        if self.head_fusion == "mean":
            return attention.mean(axis=1)
        elif self.head_fusion == "max":
            return attention.max(axis=1)[0]
        elif self.head_fusion == "min":
            return attention.min(axis=1)[0]
        else:
            raise "Attention head fusion type Not supported"


