from torch import nn
import torch
import torch.nn.functional as F
from loss import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss

"""Two contrastive encoders"""

class ConvMTM(nn.Module):
    def __init__(self, configs, args):
        super(ConvMTM, self).__init__()
        self.training_mode = args.training_mode
        self.n_leads = 12
        
        self.convs = nn.ModuleList([
            ConvLead(configs, args) for _ in range(self.n_leads)
        ])
        self.projs = nn.ModuleList([
            nn.Linear(128, 128) for _ in range(self.n_leads)
        ])
        
        self.fuser = nn.Linear(128*12, 128*2)
        
    def forward(self, x_in_t):
        
        all_h_out = []
        for l in range(self.n_leads):
            
            x_in_t_1 = x_in_t[:, l:l+1, :]

            h_t, z_t = self.convs[l](x_in_t_1, False)
            h_t = self.projs[l](h_t)
            
            all_h_out.append(h_t)
            
        out = self.fuser(torch.cat(all_h_out, dim=1))
        return out

class ConvLead(nn.Module):
    def __init__(self, configs, args):
        super(ConvLead, self).__init__()
        self.training_mode = args.training_mode
        
        self.encoder_t = Inception1DBase(input_channels=1)

        self.dense = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        if self.training_mode == 'pre_train':
            self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(args)
            self.aggregation = AggregationRebuild(args)
            self.head = nn.Linear(128, 1000)
            self.mse = torch.nn.MSELoss()

    def forward(self, x_in_t, pretrain=False):

        if pretrain:
            x = self.encoder_t(x_in_t)
            h = x.reshape(x.shape[0], -1)
            z = self.dense(h)

            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
            rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x)
            pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))

            loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss = self.awl(loss_cl, loss_rb)

            return loss, loss_cl, loss_rb
        
        else:
            x = self.encoder_t(x_in_t)
            h = x.reshape(x.shape[0], -1)
            z = self.dense(h)

            return h, z

class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)
        
    def forward(self, emb):
        emb = F.relu(self.logits(emb))
        pred = self.logits_simple(emb)
        return pred
    
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