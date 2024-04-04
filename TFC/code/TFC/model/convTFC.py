from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""Two contrastive encoders"""


class ConvTFC(nn.Module):
    def __init__(self, configs):
        super(ConvTFC, self).__init__()
        
        self.n_leads = 12
        self.convs = nn.ModuleList([
            ConvLead(configs) for _ in range(self.n_leads)
        ])
        
        self.projs = nn.ModuleList([
            nn.Linear(128*2, 128) for _ in range(self.n_leads)
        ])
        self.fuser = nn.Linear(128*12, 128*2)
        
    def forward(self, x_in_t, x_in_f):
        
        all_z_out = []
        for l in range(self.n_leads):
            
            x_in_t_1 = x_in_t[:, l:l+1, :]
            x_in_f_1 = x_in_f[:, l:l+1, :]
            _, z_time_1, _, z_freq_1 = self.convs[l](x_in_t_1, x_in_f_1)
            z_out = self.projs[l](torch.cat([z_time_1, z_freq_1], dim=1))
            all_z_out.append(z_out)
            
        out = self.fuser(torch.cat(all_z_out, dim=1))
        return out

class ConvLead(nn.Module):
    def __init__(self, configs):
        super(ConvLead, self).__init__()

        
        self.encoder_t = Inception1DBase(input_channels=1)

        self.projector_t = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.encoder_f = Inception1DBase(input_channels=1)

        self.projector_f = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq

class InceptionBlock1D(nn.Module):
    def __init__(self, input_channels):
        super(InceptionBlock1D, self).__init__()
        self.input_channels = input_channels
        self.bottleneck = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.convs_conv1 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.convs_conv2 = nn.Conv1d(32, 32, kernel_size=19, stride=1, padding=9, bias=False)
        self.convs_conv3 = nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.convbottle_maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.convbottle_conv = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.bnrelu_relu = nn.ReLU()
        
    def forward(self, x):
        bottled = self.bottleneck(x)
        y = torch.cat([
            self.convs_conv1(bottled),
            self.convs_conv2(bottled),
            self.convs_conv3(bottled),
            self.convbottle_conv(self.convbottle_maxpool(x))
        ], dim=1)
        out = self.bnrelu_relu(y)
        return out

class Shortcut1D(nn.Module):
    def __init__(self, input_channels):
        super(Shortcut1D, self).__init__()
        self.input_channels = input_channels
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(self.input_channels, 128, kernel_size=1, stride=1, bias=False)
    def forward(self, inp, out):
        return self.act_fn(out + self.conv(inp))
        
class Inception1DBase(nn.Module):
    def __init__(self, input_channels=1):
        super(Inception1DBase, self).__init__()
        self.input_channels = input_channels
        # inception backbone
        self.inceptionbackbone_1 = InceptionBlock1D(input_channels=self.input_channels)
        self.inceptionbackbone_2 = InceptionBlock1D(input_channels=128)
        # shortcut
        self.shortcut_1 = Shortcut1D(input_channels=self.input_channels)
        # pooling
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        self.mp = nn.AdaptiveMaxPool1d(output_size=1)
        # flatten
        self.flatten = nn.Flatten()
        self.dropout_1 = nn.Dropout(p=0.1, inplace=False)
        self.ln_1 = nn.Linear(256, 128, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # self.ln_2 = nn.Linear(128, 71, bias=True)
    def forward(self, x):
        # inception backbone
        input_res = x
        x = self.inceptionbackbone_1(x)
        x = self.shortcut_1(input_res, x)
        x = self.inceptionbackbone_2(x)
        
        # head
        x = torch.cat([self.mp(x), self.ap(x)], dim=1)
        x = self.flatten(x)
        x = self.dropout_1(x)
        x = self.ln_1(x)

        return x

class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb = torch.tanh(self.logits(emb))
        pred = self.logits_simple(emb)
        return pred