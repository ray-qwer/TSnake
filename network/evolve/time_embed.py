import torch.nn as nn
import torch
from .snake import BasicBlock

class trainable_pos_encoding(nn.Module):
    def __init__(self, max_evolve, channels, device='cuda'):
        super().__init__()
        self.pos_enc = nn.Embedding(max_evolve, channels, device='cuda')
        nn.init.constant_(self.pos_enc.weight, 0)
        self.device = device

    def forward(self, t):
        t = torch.tensor([t], device=self.device)
        return self.pos_enc(t)

class pos_encoding(nn.Module):
    def __init__(self, channels, device='cuda'):
        super().__init__()
        self.channels = channels
        self.device = device
        self.inv_freq = 1.0 / (
        10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
    
    def forward(self, t):
        t = torch.tensor([t], device=self.device)
        pos_enc_a = torch.sin(t.repeat(1, self.channels // 2) * self.inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.channels // 2) * self.inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1) # channel

        return pos_enc

class timeEmbedBasicBlock(BasicBlock):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1, use_GN=False, emb_dim=128):
        super().__init__(state_dim, out_state_dim, conv_type, n_adj, dilation, use_GN)
        self.emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, out_state_dim)
        )
    
    def forward(self, x, t):
        x = super().forward(x)
        t_emb = self.emb_layer(t)[:,:,None].repeat(1,1, x.shape[-1])
        return x + t_emb

class timePredictor(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.emb_layer1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, 256)
        )
        self.emb_layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, 64)
        )
        self.conv1 = nn.Conv1d(input_dim, 256, 1)
        self.conv2 = nn.Conv1d(256, 64, 1)
        self.conv3 = nn.Conv1d(64, 2, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        x = self.relu(self.conv1(x))
        t_emb = self.emb_layer1(t)[:,:,None].repeat(1,1, x.shape[-1])
        x = self.relu(self.conv2(x + t_emb))
        t_emb = self.emb_layer2(t)[:,:,None].repeat(1,1, x.shape[-1])
        x = self.conv3(x + t_emb)
        return x

class timeSnake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', use_GN=False, emb_dim=128):
        super().__init__()
        self.head = timeEmbedBasicBlock(feature_dim, state_dim, conv_type, use_GN=use_GN, emb_dim=emb_dim)
        self.res_layer_num = 7
        self.state_dim = state_dim
        dilation = [1, 1, 1, 2, 2, 4, 4]
        n_adj = 4
        for i in range(self.res_layer_num):
            if dilation[i] == 0:
                conv_type = 'grid'
            else:
                conv_type = 'dgrid'
            conv = timeEmbedBasicBlock(state_dim, state_dim, conv_type, n_adj=n_adj, dilation=dilation[i], use_GN=use_GN, emb_dim=emb_dim)
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )
        # self.predictor = timePredictor(state_dim * (self.res_layer_num + 1) + fusion_state_dim, emb_dim)
    def forward(self, x, t):
        states = []
        # t = t.unsqueeze(-1).type(torch.float)
        # t = pos_encoding(t, self.state_dim, device=x.device)
        x = self.head(x, t)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x, t) + x
            states.append(x)

        # if extract_hidden:
        #     feat = states[-1]
        state = torch.cat(states, dim=1)
        
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        
        x = self.prediction(state)
        # x = self.predictor(state, t)
        return x

class dilatedTimeSnake(timeSnake):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', use_GN=False, is_local=False, local_kernel=3):
        super().__init__(state_dim, feature_dim, conv_type, use_GN, is_local)
        self.local_kernel = local_kernel
        self.dilated_cnn = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=local_kernel**2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, feature_dim, kernel_size=1),
        )

        
    def forward(self, x, t):
        num_poly, feat_dim, pts, k = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, feat_dim, k)
        x = self.dilated_cnn(x).reshape(num_poly, pts, -1).permute(0, 2, 1)  #(num_poly, dim, pts)

        states = []
        x = self.head(x, t)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x, t) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        
        x = self.prediction(state)

        return x

class timeLocalBasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, kernel, emb_dim=128):
        super().__init__()
        self.conv = nn.Conv1d(state_dim, out_state_dim, kernel)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)
    
        self.emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, out_state_dim)
        )
    def forward(self, x, t):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        t_emb = self.emb_layer(t)[:,:,None].repeat(1,1,x.shape[-1])
        return x + t_emb

class timeLocalMoving(nn.Module):
    def __init__(self, local_feature_dim, state_dim=128, local_kernel=5, moving_range=5.):
        super().__init__()
        self.moving_range = moving_range
        self.local_kernel = local_kernel
        self.local_conv_1 = timeLocalBasicBlock(local_feature_dim, state_dim, kernel=local_kernel**2)
        self.local_conv_2 = timeLocalBasicBlock(state_dim, state_dim, kernel=1)
    
        fusion_state_dim = 128
        self.fusion = nn.Conv1d(state_dim * 2, fusion_state_dim, 1)
        
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * 2 + fusion_state_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x, t):
        states = []
        num_poly, feat_dim, pts, k = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, feat_dim, k)
        x = self.local_conv_1(x, t)
        states.append(x.reshape(num_poly, pts, -1).permute(0,2,1))
        x = self.local_conv_2(x, t) + x
        states.append(x.reshape(num_poly, pts, -1).permute(0,2,1))

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)

        x = self.prediction(state)
        return x * self.moving_range
