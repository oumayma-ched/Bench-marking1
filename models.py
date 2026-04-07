import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── CNN-Attention model helpers ────────────────────────────────────────────── #

def get_same_padding_1d(L_in, kernel_size, stride):
    L_out = math.ceil(L_in / stride)
    pad_total = max((L_out - 1) * stride + kernel_size - L_in, 0)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return pad_left, pad_right


class CNNAttConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0.3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                   # (B, C, T)
        pad_left, pad_right = get_same_padding_1d(x.shape[-1], self.kernel_size, self.stride)
        x = nn.functional.pad(x, (pad_left, pad_right))
        x = self.conv(x)
        x = x.transpose(1, 2)              # → (B, T, C)
        x = self.norm(x)
        x = F.silu(x)
        x = self.dropout(x)
        return x.transpose(1, 2)           # → (B, C, T)


class CNNAttInvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, expanded_channels, out_channels, stride=1, depth_kernel_size=6):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        self.depth_kernel_size = depth_kernel_size
        self.stride = stride
        self.expand    = nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.norm1     = nn.LayerNorm(expanded_channels)
        self.norm2     = nn.LayerNorm(expanded_channels)
        self.norm3     = nn.LayerNorm(out_channels)
        self.depthwise = nn.Conv1d(expanded_channels, expanded_channels,
                                   kernel_size=depth_kernel_size, stride=stride,
                                   groups=expanded_channels, bias=False)
        self.project   = nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):                   # (B, C, T)
        identity = x
        x = self.expand(x)                  # (B, expanded, T)
        x = F.silu(self.norm1(x.transpose(1, 2))).transpose(1, 2)
        pad_left, pad_right = get_same_padding_1d(x.shape[-1], self.depth_kernel_size, self.stride)
        x = self.depthwise(nn.functional.pad(x, (pad_left, pad_right)))
        x = F.silu(self.norm2(x.transpose(1, 2))).transpose(1, 2)
        x = self.project(x)
        x = self.norm3(x.transpose(1, 2)).transpose(1, 2)
        if self.use_residual:
            x = x + identity
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):                   # (B, C, T)
        x = self.encoder(x.transpose(1, 2)) # (B, T, C)
        return x.transpose(1, 2)            # (B, C, T)


class CNNAttentionModel(nn.Module):
    """CNN + Inverted-Residual + Transformer backbone for 1-D EMG signals.
    Input : (B, 1, channels, T)  — same format as other EMG models in this repo.
    Output: logits (B, num_classes)   [feats accessible via forward_features if needed]
    Uses LayerNorm throughout → safe for FL with small / heterogeneous local batches.
    """
    def __init__(self, channels=12, stride_increase=0, expansion=2, depthwise=2,
                 outputchannels=1, num_classes=17, deep_layers=2):
        super().__init__()
        c24 = outputchannels * 24
        c48 = outputchannels * 48
        self.stage1 = nn.Sequential(
            CNNAttConvBlock(channels, c24, kernel_size=4, stride=(3 + stride_increase)),
            CNNAttInvertedResidualBlock(c24, (3 + expansion) * c24, c24,
                                        depth_kernel_size=(5 + depthwise)),
            TransformerBlock(embed_dim=c24, num_heads=4, num_layers=1),
            nn.Dropout(0.5),
        )
        self.stage2 = nn.Sequential(
            CNNAttConvBlock(c24, c48, kernel_size=6, stride=(4 + stride_increase)),
            CNNAttInvertedResidualBlock(c48, (5 + expansion) * c48, c48,
                                        depth_kernel_size=(8 + depthwise)),
            TransformerBlock(embed_dim=c48, num_heads=8, num_layers=1),
            nn.Dropout(0.5),
        )
        self.global_pool  = nn.AdaptiveAvgPool1d(1)
        self.deep_layers  = deep_layers

        # Classifier — LayerNorm instead of BatchNorm1d (BN breaks FL small batches)
        if deep_layers > 1:
            self.fc1   = nn.Linear(c48, 128)
            self.ln1   = nn.LayerNorm(128)
            self.drop1 = nn.Dropout(0.5)
            self.fc2   = nn.Linear(128, 32)
            self.ln2   = nn.LayerNorm(32)
            self.drop2 = nn.Dropout(0.5)
            self.fc3   = nn.Linear(32, num_classes)
        else:
            self.fc2   = nn.Linear(c48, 32)
            self.ln2   = nn.LayerNorm(32)
            self.drop2 = nn.Dropout(0.5)
            self.fc3   = nn.Linear(32, num_classes)

    def forward(self, x):                   # (B, 1, 12, T)
        if x.dim() == 4:
            x = x.squeeze(1)               # (B, 12, T)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.global_pool(x).squeeze(-1) # (B, C)
        if self.deep_layers > 1:
            x = self.drop1(F.relu(self.ln1(self.fc1(x))))
            x = self.drop2(F.relu(self.ln2(self.fc2(x))))
        else:
            x = self.drop2(F.relu(self.ln2(self.fc2(x))))
        return self.fc3(x)


# ── Squeeze-and-Excitation channel attention (1-D) ────────────────────────── #
class SE1d(nn.Module):
    """Channel attention: recalibrates each feature-map channel globally.
    GAP → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid → scale
    Uses GroupNorm-friendly design (no BN) so it is safe for FL.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):          # x: (B, C, T)
        w = x.mean(dim=2)          # (B, C)  global average pooling over time
        w = self.fc(w)             # (B, C)
        return x * w.unsqueeze(2)  # (B, C, T)  channel-wise rescaling


class client_model(nn.Module):
    def __init__(self, name, n_cls, args=True):
        super(client_model, self).__init__()
        self.name = name
        self.n_cls = n_cls
        
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
          
        if self.name == 'mnist_2NN':
            # self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)
            
        if self.name == 'emnist_NN':
            # self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
        
        if self.name == 'LeNet':
            # self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'EMG_CNN':
            # Input: (B, 12, 400)  – 12 EMG channels, 400 time samples (200 ms)
            # Uses GroupNorm instead of BatchNorm: safe for FL with small/heterogeneous
            # local batches (same reason ResNet18 uses GN in this codebase).
            self.conv1 = nn.Conv1d(12,  64, kernel_size=3, padding=1)   # → (64, 400)
            self.gn1   = nn.GroupNorm(4, 64)
            self.conv2 = nn.Conv1d(64,  64, kernel_size=3, padding=1)   # → (64, 400)
            self.gn2   = nn.GroupNorm(4, 64)
            self.pool1 = nn.MaxPool1d(2)                                  # → (64, 200)
            self.conv3 = nn.Conv1d(64,  128, kernel_size=3, padding=1)  # → (128, 200)
            self.gn3   = nn.GroupNorm(4, 128)
            self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # → (128, 200)
            self.gn4   = nn.GroupNorm(4, 128)
            self.pool2 = nn.MaxPool1d(2)                                  # → (128, 100)
            self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # → (256, 100)
            self.gn5   = nn.GroupNorm(4, 256)
            self.pool3 = nn.MaxPool1d(2)                                  # → (256,  50)
            self.drop  = nn.Dropout(p=0.5)
            self.fc1   = nn.Linear(256 * 50, 512)
            self.fc2   = nn.Linear(512, self.n_cls)

        if self.name == 'EMG_ATTN':
            # Input: (B, 12, 400)  – 12 EMG channels, 400 time samples (200 ms)
            # Architecture: 1D-CNN backbone with SE channel-attention after every block.
            # GroupNorm throughout → safe for FL with small / heterogeneous local batches.
            #
            #  Block 1:  Conv(12→64,  k=7) + GN + ReLU + SE          → (64,  400)
            #  Block 2:  Conv(64→64,  k=5) + GN + ReLU + SE + Pool/2  → (64,  200)
            #  Block 3:  Conv(64→128, k=5) + GN + ReLU + SE          → (128, 200)
            #  Block 4:  Conv(128→128,k=3) + GN + ReLU + SE + Pool/2  → (128, 100)
            #  Block 5:  Conv(128→256,k=3) + GN + ReLU + SE + Pool/2  → (256,  50)
            #  Global Average Pooling → (256,)
            #  FC(256→128) + Dropout(0.5) → FC(128→n_cls)

            self.conv1 = nn.Conv1d(12,  64,  kernel_size=7, padding=3)
            self.gn1   = nn.GroupNorm(4, 64);   self.se1 = SE1d(64)
            self.conv2 = nn.Conv1d(64,  64,  kernel_size=5, padding=2)
            self.gn2   = nn.GroupNorm(4, 64);   self.se2 = SE1d(64)
            self.pool2 = nn.MaxPool1d(2)
            self.conv3 = nn.Conv1d(64,  128, kernel_size=5, padding=2)
            self.gn3   = nn.GroupNorm(4, 128);  self.se3 = SE1d(128)
            self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
            self.gn4   = nn.GroupNorm(4, 128);  self.se4 = SE1d(128)
            self.pool4 = nn.MaxPool1d(2)
            self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.gn5   = nn.GroupNorm(4, 256);  self.se5 = SE1d(256)
            self.pool5 = nn.MaxPool1d(2)
            self.drop  = nn.Dropout(p=0.5)
            self.fc1   = nn.Linear(256, 128)
            self.fc2   = nn.Linear(128, self.n_cls)

        if self.name == 'CNNATTEN':
            # CNN + Inverted-Residual + Transformer for EMG (12 channels, 400 time samples)
            self.model = CNNAttentionModel(channels=12, num_classes=self.n_cls)

        if self.name == 'ResNet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, self.n_cls)

            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
            
            self.model = resnet18

        
    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)
            
        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
  
        if self.name == 'emnist_NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'EMG_CNN':
            # Input expected: (B, 1, 12, 400) from DatasetObject → squeeze to (B, 12, 400)
            if x.dim() == 4:
                x = x.squeeze(1)                              # (B, 12, 400)
            x = F.relu(self.gn1(self.conv1(x)))              # (B,  64, 400)
            x = F.relu(self.gn2(self.conv2(x)))              # (B,  64, 400)
            x = self.pool1(x)                                 # (B,  64, 200)
            x = F.relu(self.gn3(self.conv3(x)))              # (B, 128, 200)
            x = F.relu(self.gn4(self.conv4(x)))              # (B, 128, 200)
            x = self.pool2(x)                                 # (B, 128, 100)
            x = F.relu(self.gn5(self.conv5(x)))              # (B, 256, 100)
            x = self.pool3(x)                                 # (B, 256,  50)
            x = x.view(x.size(0), -1)                        # (B, 12800)
            x = self.drop(F.relu(self.fc1(x)))
            x = self.fc2(x)

        if self.name == 'EMG_ATTN':
            if x.dim() == 4:
                x = x.squeeze(1)                                        # (B,  12, 400)
            x = self.se1(F.relu(self.gn1(self.conv1(x))))               # (B,  64, 400)
            x = self.se2(F.relu(self.gn2(self.conv2(x))))               # (B,  64, 400)
            x = self.pool2(x)                                            # (B,  64, 200)
            x = self.se3(F.relu(self.gn3(self.conv3(x))))               # (B, 128, 200)
            x = self.se4(F.relu(self.gn4(self.conv4(x))))               # (B, 128, 200)
            x = self.pool4(x)                                            # (B, 128, 100)
            x = self.se5(F.relu(self.gn5(self.conv5(x))))               # (B, 256, 100)
            x = self.pool5(x)                                            # (B, 256,  50)
            x = x.mean(dim=2)                                            # (B, 256) global avg pool
            x = self.drop(F.relu(self.fc1(x)))
            x = self.fc2(x)

        if self.name == 'CNNATTEN':
            x = self.model(x)

        if self.name == 'ResNet18':
            x = self.model(x)

        return x
    
