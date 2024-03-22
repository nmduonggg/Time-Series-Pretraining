from torch import nn
import torch
import torch.nn.functional as F
import torch.nn as nn


class TransTFC(nn.Module):
    def __init__(self, configs):
        super(TransTFC, self).__init__()

        # temport module
        self.proj_in_t = nn.Linear(1, 256)
        encoder_layers_t = nn.TransformerEncoderLayer(d_model=256,
                                                      dim_feedforward=512,
                                                      nhead=2, batch_first=True)
        self.transformer_encoder_t = nn.TransformerEncoder(encoder_layers_t, 2)
        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.proj_out_t = nn.Linear(256, 1)
        
        # frequency module
        self.proj_in_f = nn.Linear(1, 256)
        encoder_layers_f = nn.TransformerEncoderLayer(d_model=256,
                                                      dim_feedforward=512,
                                                      nhead=2, batch_first=True)
        self.transformer_encoder_f = nn.TransformerEncoder(encoder_layers_f, 2)
        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.proj_out_f = nn.Linear(256, 1)

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        # BxCxL -> BxLxC
        x = self.proj_in_t(x_in_t.permute(0,2,1))
        x = self.transformer_encoder_t(x)
        x = self.proj_out_t(x)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        # BxCxL -> BxLxC
        f = self.proj_in_f(x_in_f.permute(0,2,1))
        f = self.transformer_encoder_f(f)
        f = self.proj_out_f(f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
    
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb = F.relu(self.logits(emb))
        pred = self.logits_simple(emb)
        return pred