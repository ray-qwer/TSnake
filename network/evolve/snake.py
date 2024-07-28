import torch.nn as nn
import torch
from .utils import get_local_feature, get_average_length
GROUPS = 32

class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
        return self.fc(input)


_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}

class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1, use_GN=False):
        super(BasicBlock, self).__init__()
        if conv_type == 'grid':
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj)
        else:
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.GroupNorm(GROUPS, out_state_dim) if use_GN else nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', use_GN=False, is_local=False):
        super(Snake, self).__init__()
        self.head = BasicBlock(feature_dim, state_dim, conv_type, use_GN=use_GN)
        self.res_layer_num = 4 if is_local else 7
        dilation = [1, 1, 2, 2] if is_local else [1, 1, 1, 2, 2, 4, 4]
        assert len(dilation) == self.res_layer_num
        n_adj = 4
        for i in range(self.res_layer_num):
            if dilation[i] == 0:
                conv_type = 'grid'
            else:
                conv_type = 'dgrid'
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=n_adj, dilation=dilation[i], use_GN=use_GN)
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

    def forward(self, x, extract_hidden=False):
        states = []
        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x) + x
            states.append(x)

        if extract_hidden:
            feat = states[-1]
        state = torch.cat(states, dim=1)
        
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)
        
        x = self.prediction(state)

        return (x, feat) if extract_hidden else x
    
class LocalMovingSnake(nn.Module):
    def __init__(self, state_dim, feature_dim, local_feature_dim, localmoving_embed_dim=64, local_kernel=3, conv_type='dgrid', use_GN=False):
        super(LocalMovingSnake, self).__init__()
        self.head = BasicBlock(feature_dim, state_dim, conv_type, use_GN=use_GN)
        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        n_adj = 4

        for i in range(self.res_layer_num):
            if dilation[i] == 0:
                conv_type = 'grid'
            else:
                conv_type = 'dgrid'
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=n_adj, dilation=dilation[i], use_GN=use_GN)
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        
        self.local_kernel = local_kernel
        self.localmoving_embed_dim = localmoving_embed_dim
        self.localmoving_embed = nn.Sequential(
            nn.Conv2d(local_feature_dim, 64, kernel_size=local_kernel, padding=local_kernel//2, stride=1),    # feature dim
            nn.ReLU(inplace=True),
            nn.Conv2d(64, localmoving_embed_dim, kernel_size=local_kernel)
        )
        
        self.state_embed = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, localmoving_embed_dim, 1),
            nn.ReLU(inplace=True)
        )

        self.prediction = nn.Sequential(
            nn.Conv1d(localmoving_embed_dim, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )
    
    def forward(self, x, local_feat):   # x is features
        
        states = []
        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))

        # local feat: (b, dim, num_point, kernel_size**2) -> permute to (b * num_point, dim, kernel_size**2) 
        #             reshape to (b * num_point, dim, kernel_size, kernel_size)
        batch_size, feat_dim, num_point, _ = local_feat.shape
        local_feat = local_feat.permute(0, 2, 1, 3).reshape(-1, feat_dim, self.local_kernel, self.local_kernel)
        local_feat = self.localmoving_embed(local_feat).reshape(batch_size, num_point, -1).permute(0, 2, 1) # (b, dim, num_points)

        state = torch.cat([global_state, state], dim=1)
        
        x = self.state_embed(state)
        x = self.prediction(x + local_feat)

        return x

class LocalMovingSepSnake(nn.Module):
    # maybe layernorm much suitable
    def __init__(self, local_feature_dim, localmoving_embed_dim=64, local_kernel=3, moving_range=5.):
        super(LocalMovingSepSnake, self).__init__()
        self.moving_range = moving_range
        self.local_kernel = local_kernel
        self.localmoving_embed = nn.Sequential(
            nn.Conv2d(local_feature_dim, 64, kernel_size=local_kernel, padding=local_kernel//2, stride=1),
            nn.ReLU(inplace=True),
            # nn.LayerNorm([64,3,3]),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, localmoving_embed_dim, kernel_size=local_kernel),
            nn.ReLU(inplace=True),
            # nn.LayerNorm([localmoving_embed_dim,1,1]),
            nn.BatchNorm2d(localmoving_embed_dim)
        )
    
        self.state_dim = 128
        self.prediction = nn.Sequential(
            nn.Conv1d(localmoving_embed_dim + self.state_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, state):
        # x.shape: (b, dim, num_point, kernel_size**2)
        num_poly, feat_dim, num_point, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, feat_dim, self.local_kernel, self.local_kernel)
        x = self.localmoving_embed(x).reshape(num_poly, num_point, -1).permute(0, 2, 1)    # (num_poly , dim, num_point)
        state = torch.cat([x, state], dim=1)
        x = self.prediction(state)
        return x * self.moving_range

class AttentiveLocalMoving(nn.Module):
    def __init__(self, local_feature_dim, localmoving_embed_dim=64, local_kernel=3, moving_range=5.):
        super().__init__()
        # variables
        self.moving_range = moving_range    # choose bigger?
        self.local_kernel = local_kernel
        self.state_dim = 64

        # local features
        self.localmoving_embed = nn.Sequential(
            nn.Conv2d(local_feature_dim, 64, kernel_size=local_kernel, padding=local_kernel//2, stride=1),
            nn.ReLU(inplace=True),
            nn.LayerNorm([64,3,3]),
            nn.Conv2d(64, localmoving_embed_dim, kernel_size=local_kernel),
            nn.ReLU(inplace=True),
            nn.LayerNorm([localmoving_embed_dim,1,1])
        )

        # prediction
        self.prediction = nn.Sequential(
            nn.Conv1d(localmoving_embed_dim + self.state_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 2, 1),
            nn.Tanh()
        )

        # attention
        self.attentivePredict = nn.Sequential(
            nn.Linear(64, 256),    # the first dim is not sure
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x, attenFeat):
        # what features is attenFeat
        attenFeat_ = attenFeat.permute(0, 2, 1)
        attentive = self.attentivePredict(attenFeat_)
        num_poly, feat_dim, num_point, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, feat_dim, self.local_kernel, self.local_kernel)
        x = self.localmoving_embed(x).reshape(num_poly, num_point, -1).permute(0, 2, 1)
        state = torch.cat([x, attenFeat], dim=1)
        x = self.prediction(state)
        x = (x * self.moving_range).permute(0,2,1) * attentive
        return x.permute(0,2,1)
    
class TransFeat(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.transfeat = torch.nn.Sequential(   torch.nn.Conv2d(c_in, 64, kernel_size=3, padding=1, bias=True),
                                                torch.nn.BatchNorm2d(64),
                                                torch.nn.ReLU(inplace=True),
                                                torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True),
                                                torch.nn.BatchNorm2d(64),
                                                torch.nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.transfeat(x) + x
        return torch.relu(x)

class DilatedCNNSnake(Snake):
    """
        according to the neighbor length, to get its receptive field
    """
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', use_GN=False, is_local=False, local_kernel=3):
        super().__init__(state_dim, feature_dim, conv_type, use_GN, is_local)
        self.local_kernel = local_kernel
        self.dilated_cnn = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=local_kernel**2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, feature_dim, kernel_size=1),
        )
    
    def forward(self, x):
        # x.shape: (b, dim, pts, k**2), e.g. (b, 64, 128, 9)
        num_poly, feat_dim, pts, k = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, feat_dim, k)
        x = self.dilated_cnn(x).reshape(num_poly, pts, -1).permute(0, 2, 1)  #(num_poly, dim, pts)
        x = super().forward(x)
        return x

class LocalBasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, kernel):
        super().__init__()
        self.conv = nn.Conv1d(state_dim, out_state_dim, kernel)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class dilatedLocalMoving(nn.Module):
    def __init__(self, local_feature_dim, state_dim=128, local_kernel=5):
        super().__init__()
        self.local_kernel = local_kernel
        self.local_conv_1 = LocalBasicBlock(local_feature_dim, state_dim, kernel=local_kernel**2)
        self.local_conv_2 = LocalBasicBlock(state_dim, state_dim, kernel=1)
    
        fusion_state_dim = 128
        self.fusion = nn.Conv1d(state_dim * 2, fusion_state_dim, 1)
        
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * 2 + fusion_state_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 2, 1),
        )
    
    def forward(self, x):
        states = []
        num_poly, feat_dim, pts, k = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, feat_dim, k)
        x = self.local_conv_1(x)
        states.append(x.reshape(num_poly, pts, -1).permute(0,2,1))
        x = self.local_conv_2(x) + x
        states.append(x.reshape(num_poly, pts, -1).permute(0,2,1))

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        state = torch.cat([global_state, state], dim=1)

        x = self.prediction(state)
        return x

class ResDilatedCNNSnake(Snake):
    """
        according to the neighbor length, to get its receptive field
    """
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', use_GN=False, is_local=False, local_kernel=3):
        super().__init__(state_dim, state_dim, conv_type, use_GN, is_local)
        self.local_kernel = local_kernel
        self.local_kernel = local_kernel
        self.local_conv_1 = LocalBasicBlock(feature_dim, state_dim, kernel=local_kernel**2)
        self.local_conv_2 = LocalBasicBlock(state_dim, state_dim, kernel=1)
    
    def forward(self, x):
        # x.shape: (b, dim, pts, k**2), e.g. (b, 66, 128, 9)
        num_poly, feat_dim, pts, k = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, feat_dim, k)
        x = self.local_conv_1(x)
        x = self.local_conv_2(x) + x
        x = x.reshape(num_poly, pts, -1).permute(0, 2, 1)
        x = super().forward(x)
        return x