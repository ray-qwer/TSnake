import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..evolve.utils import get_gcn_feature
import collections
from .utils import *
"""
    use attention map
"""
class MultiLayerAttentionCombine(nn.Module):
    def __init__(self, in_c, num_points, n_embd, heads, layers, pe_method="abs", rpe_mode=None):
        super().__init__()
        self.heads = heads
        self.head_dim = n_embd // heads
        self.n_embd = n_embd
        self.contour_stride = 4
        self.contour_feat_proj = nn.Conv1d(in_c, n_embd, kernel_size=num_points // self.contour_stride)
        self.points_kernel_size = num_points // self.contour_stride
        # deeper make it worse
        # self.contour_feat_proj = nn.Sequential(
        #     nn.Conv1d(in_c, n_embd, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(n_embd, n_embd, kernel_size=num_points // self.contour_stride)
        # )
        self.trans_feat = torch.nn.Sequential(torch.nn.Conv2d(in_c - 2, 256, kernel_size=3, padding=1, bias=True),
                                                torch.nn.ReLU(inplace=True),
                                                torch.nn.Conv2d(256, in_c - 2, kernel_size=1, stride=1, padding=0, bias=True))
        assert pe_method in ['abs', 'rel']
        self.pe_method = pe_method
        if pe_method == "rel":
            if rpe_mode is None:
                rpe_mode = "bias"
            assert rpe_mode in ["bias", "contextual"]

        # position encoding
        self.patch = 16
        pos_embed = getPositionEmbeddingSine(num_pos_feats=n_embd, patch=self.patch, normalize=True)
        self.register_buffer('pos_embed', pos_embed)

        # layers
        num_encode_layer = layers - 1
        pred_layer = [Block(n_embd, n_embd, heads, pe_method=pe_method, rpe_mode=rpe_mode, layer=i) for i in range(num_encode_layer)]
        # pred_layer.append(RPEPredAttention(n_embd, n_embd, heads))
        pred_layer.append(PredAttention(n_embd, n_embd, heads))
        self.predictor = nn.Sequential(*pred_layer)
        
    
    def forward(self, data_input, output, cnn_feature, from_dataset=True):
        down_ratio = data_input['inp'].size(-1) // cnn_feature.size(-1)
        trans_feature = self.trans_feat(cnn_feature)
        if from_dataset:
            contour_features, ct_x, ct_y, ct_01 = self.preprocess_from_dataset(data_input, output, trans_feature, down_ratio)
        else:
            img_hw = data_input['inp'].shape[-2:]
            contour_features, ct_x, ct_y, ct_01 = self.preprocess_from_model(output, trans_feature, img_hw, down_ratio)
        
        b, c, p = contour_features.shape
        assert p == self.points_kernel_size, f"got contour feature shape {contour_features.shape}"
        batch_size, max_len = ct_01.shape
        contour_features = self.contour_feat_proj(contour_features).squeeze(-1)
        if self.pe_method == "abs":
            pos_embed = self.pos_embed[:, ct_y, ct_x].transpose(1,0)
            contour_features = contour_features + pos_embed
            x = torch.zeros(batch_size, max_len, self.n_embd, device=cnn_feature.device)
            x[ct_01] = contour_features
            att = self.predictor(x)
        else:
            x = torch.zeros(batch_size, max_len, self.n_embd, device=cnn_feature.device)
            ct = torch.zeros(batch_size, max_len, 2, device=cnn_feature.device)
            x[ct_01] = contour_features
            ct[ct_01] = torch.stack([ct_y, ct_x], dim=-1).to(torch.float32)
            att = self.predictor((x, ct))
        return att

    def preprocess_from_dataset(self, data_input, output, cnn_feature, down_ratio=4):
        ct_01 = data_input['ct_01'].bool()
        ct_img_idx = data_input['ct_img_idx'][ct_01]
        contours = output['py_pred'][-1] # get the last contours
        h, w = data_input['inp'].shape[-2:]
        # ct_x, ct_y
        batch, _, height, width = cnn_feature.shape
        ct_ind = data_input['ct_ind'][ct_01]
        # ct_x: (batch, max_len)
        ct_x, ct_y = ct_ind % width, torch.div(ct_ind, width, rounding_mode='floor')
        # get cnn features, concat with coordinates
        contour_sample = contours[:, ::self.contour_stride]
        # 0417 add center points
        # cts = (torch.stack((ct_x, ct_y), dim=-1) * down_ratio).unsqueeze(1)
        # contour_sample = torch.cat([contour_sample, cts], dim=1)
        # normed contour coordinates into [0,1]
        contour_features = get_gcn_feature(cnn_feature, contour_sample, ct_img_idx, h, w)
        normed_contours = torch.div(contour_sample, torch.tensor([w,h], device=contours.device))
        contour_features = torch.cat([contour_features, normed_contours.permute(0,2,1)], dim=1)    
        
        if self.pe_method == "abs":
            ct_x = torch.div(ct_x * self.patch, width, rounding_mode='floor')
            ct_y = torch.div(ct_y * self.patch, height, rounding_mode='floor')

        return contour_features, ct_x, ct_y, ct_01

    def preprocess_from_model(self, output, cnn_feature, img_hw, down_ratio=4):
        h, w = img_hw
        img_idx = output["img_idx"]
        cnn_idx, counts = torch.unique(img_idx, return_counts=True)
        if len(cnn_idx) == 0:
            batch_size, max_len = 0, 0
        else:
            batch_size, max_len = torch.max(cnn_idx) + 1, torch.max(counts)
        ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool, device=cnn_feature.device)
        for idx, count in zip(cnn_idx, counts):
            ct_01[idx.item(), :count.item()] = 1

        # ct_x, ct_y
        height, width = cnn_feature.shape[-2:]
        ct_x, ct_y = output['detection'][:, 0], output['detection'][:, 1]

        contours = output['py'][-1]
        contour_sample = contours[:,::self.contour_stride]
        # # 0417 added
        # cts = (torch.stack((ct_x, ct_y), dim=-1) * down_ratio).unsqueeze(1)
        # contour_sample = torch.cat([contour_sample, cts], dim=1)
        # ###########
        contour_features = get_gcn_feature(cnn_feature, contour_sample, img_idx, h, w)
        normed_contours = torch.div(contour_sample, torch.tensor([w,h], device=contours.device))
        contour_features = torch.cat([contour_features, normed_contours.permute(0,2,1)], dim=1)    
        output.update({'ct_01': ct_01.detach()})

        
        
        if self.pe_method == "abs":
            ct_x = torch.div(ct_x * self.patch, width, rounding_mode='floor').to(torch.int64)
            ct_y = torch.div(ct_y * self.patch, height, rounding_mode='floor').to(torch.int64)

        return contour_features, ct_x, ct_y, ct_01

class MultiLayerAttentionCombine_Bbox(MultiLayerAttentionCombine):
    def __init__(self, in_c, num_points, n_embd, heads, layers, pe_method="abs", rpe_mode=None):
        super().__init__(in_c, num_points, n_embd, heads, layers, pe_method, rpe_mode)
        num_encode_layer = layers - 1
        pred_layer = [Block(n_embd, n_embd, heads, pe_method=pe_method, rpe_mode=rpe_mode, layer=i) for i in range(num_encode_layer)]
        pred_layer.append(PredAttentionValue(n_embd, n_embd, heads))
        self.predictor = nn.Sequential(*pred_layer)
        self.bbox_mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(n_embd, 2 * n_embd)),
            ('act0', nn.GELU()), # default: nn.GELU(approximate='tanh')
            ('c_proj', nn.Linear(2 * n_embd, n_embd)),
            ('act1', nn.ReLU(inplace=True)),
            ('box_pred', nn.Linear(n_embd, 4)),
            ('act2', nn.ReLU(inplace=True)),
        ]))
    def forward(self, data_input, output, cnn_feature, from_dataset=True):
        down_ratio = data_input['inp'].size(-1) // cnn_feature.size(-1)
        if from_dataset:
            contour_features, ct_x, ct_y, ct_01 = self.preprocess_from_dataset(data_input, output, cnn_feature)
        else:
            img_hw = data_input['inp'].shape[-2:]
            contour_features, ct_x, ct_y, ct_01 = self.preprocess_from_model(output, cnn_feature, img_hw)
        b, c, p = contour_features.shape
        batch_size, max_len = ct_01.shape
        contour_features = self.contour_feat_proj(contour_features).squeeze(-1)
        
        if self.pe_method == "abs":
            pos_embed = self.pos_embed[:, ct_y, ct_x].transpose(1,0)
            contour_features = contour_features + pos_embed
            x = torch.zeros(batch_size, max_len, self.n_embd, device=cnn_feature.device)
            x[ct_01] = contour_features
            pred_weight, value = self.predictor(x)
        else:
            x = torch.zeros(batch_size, max_len, self.n_embd, device=cnn_feature.device)
            ct = torch.zeros(batch_size, max_len, 2, device=cnn_feature.device)
            x[ct_01] = contour_features
            ct[ct_01] = torch.stack([ct_y, ct_x], dim=-1).to(torch.float32)
            pred_weight, value = self.predictor((x, ct))
        
        bbox_offset = self.bbox_mlp(value)
        bbox_offset = bbox_offset[ct_01]
        ct_x, ct_y = ct_x.to(torch.float32), ct_y.to(torch.float32)
        bbox_pred = torch.stack([ct_x - bbox_offset[:, 0], ct_y - bbox_offset[:,1], ct_x + bbox_offset[:, 2], ct_y + bbox_offset[:,3]], dim=-1)
        return pred_weight, bbox_pred * down_ratio

class PredAttentionValue(nn.Module):
    def __init__(self, in_c, n_embd, heads):
        """
            This is the last layer, to output the prediction of each component
        """
        super(PredAttentionValue, self).__init__()
        self.heads = heads
        self.head_dim = n_embd // heads
        self.n_embd = n_embd
        self.c_attn = nn.Linear(in_c, 3 * n_embd)
        self.p_proj = nn.Conv1d(heads, 1, 1, bias=False)
        size = 500
        self.register_buffer('bias', (torch.ones(size, size)).view(1, 1, size, size))
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if B == 0:
            return torch.zeros(B,T,T).to(x.device)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        pred_weight = att.reshape(B, self.heads, T * T)
        
        # generate prediction weight
        pred_weight = self.p_proj(pred_weight).reshape(B, T, T)
        pred_weight = torch.sigmoid(pred_weight)
        pred_weight = torch.where(torch.isnan(pred_weight), torch.full_like(pred_weight, 0), pred_weight)

        # bbox
        att = F.softmax(att, dim=-1)
        value = self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, self.n_embd))
        return pred_weight, value

class PredAttention(nn.Module):
    def __init__(self, in_c, n_embd, heads):
        """
            This is the last layer, to output the prediction of each component
        """
        super(PredAttention, self).__init__()
        self.heads = heads
        self.head_dim = n_embd // heads
        self.n_embd = n_embd
        self.c_attn = nn.Linear(in_c, 2 * n_embd)
        self.p_proj = nn.Conv1d(heads, 1, 1, bias=False)
        size = 500
        self.register_buffer('bias', (torch.ones(size, size)).view(1, 1, size, size))
    
    def forward(self, x):
        if isinstance(x, tuple):
            x, ct = x
        B, T, C = x.size()
        if B == 0:
            return torch.zeros(B,T,T).to(x.device)
        q, k = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.reshape(B, self.heads, T * T)
        att = self.p_proj(att).reshape(B, T, T)
        att = torch.sigmoid(att)
        att = torch.where(torch.isnan(att), torch.full_like(att, 0), att)
        return att

class RPEPredAttention(PredAttention):
    def __init__(self, in_c, n_embd, heads, num_buckets=None):
        """
            This is the last layer, to output the prediction of each component
        """
        super().__init__(in_c, n_embd, heads)
        self.rpe_mode = 'bias'
        # relative
        self.patch = 16 # origin 16, v1: 16
        self.pix_per_patch = 200 // self.patch # origin 200, v1 200
        self.alpha = 12 # origin 8, v1: 12
        self.beta = 2 * self.alpha
        self.gamma = 8 * self.alpha
        self.num_buckets = 2 * self.beta + 1 if num_buckets is None else num_buckets
        
        def initializer(x): return None
        self.initializer = initializer

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # initialize parameters of iRPE
        # transposed
        if self.rpe_mode == 'bias':
            self.lookup_table_bias = nn.Parameter(torch.zeros(self.heads, self.num_buckets))
        else:
            self.lookup_table_weight = nn.Parameter(torch.zeros(self.heads, self.head_dim, self.num_buckets))

    def forward(self, x):
        x, ct = x
        B, T, _ = ct.shape
        ct_1 = ct.reshape(B, T, 1, 2)
        ct_2 = ct.reshape(B, 1, T, 2)
        # convert distance into patch
        ct_ = (ct_1 - ct_2).square().sum(-1).float().sqrt() # absolute position in pixel
        ct_ = torch.div(ct_, self.pix_per_patch, rounding_mode='trunc')
        # convert patch to piecewise index
        bucket_ids = piecewise_index(ct_, self.alpha, self.beta, self.gamma, torch.int)
        beta_int = int(self.beta)
        bucket_ids = bucket_ids + beta_int
        bucket_ids = bucket_ids.view(B, T, T)
        
        B, T, C = x.size()
        if B == 0:
            return torch.zeros(B,T,T).to(x.device)
        q, k = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        if self.rpe_mode == 'bias':
            rpe = self.lookup_table_bias[:, bucket_ids.flatten().long()].view(B, self.heads, T, T)
        else:
            lookup_table = torch.matmul(q.transpose(0,1).reshape(self.heads, B * T, self.head_dim), self.lookup_table_weight).\
                    view(-1, B, T, self.num_buckets)    # (H, B, T, BU)
            offset = torch.arange(0, T * self.num_buckets, self.num_buckets, dtype=bucket_ids.dtype, device=bucket_ids.device).view(-1, 1)
            rp_bucket = (bucket_ids + offset).long().flatten(-2)  # (B*T*T)
            batch_offset = torch.arange(0, B * T * self.num_buckets, T * self.num_buckets, dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1)
            brp_bucket = (rp_bucket + batch_offset).flatten()
            rpe = lookup_table.flatten(1)[:, brp_bucket].reshape(-1, B, T, T).transpose(0,1)
        att = att + rpe
        att = att.reshape(B, self.heads, T * T)

        att = self.p_proj(att).reshape(B, T, T)
        att = torch.sigmoid(att)
        att = torch.where(torch.isnan(att), torch.full_like(att, 0), att)
        return att

class Attention(nn.Module):
    def __init__(self, in_c, n_embd, heads):
        """
            Here is ordinary attention decoder, get (K@Q)@V for output
            positional encoding only add at the first layer
        """
        super(Attention, self).__init__()
        self.heads = heads
        self.head_dim = n_embd // heads
        self.n_embd = n_embd
        self.c_attn = nn.Linear(in_c, 3 * n_embd)
        size = 500
        self.register_buffer('bias', (torch.ones(size, size)).view(1, 1, size, size)) # mask for causal attention
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.heads, self.n_embd // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, self.n_embd // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, self.n_embd // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, self.n_embd)) # (b,H,T,R) -> (b,T,C)

class Block(nn.Module):
    def __init__(self, in_c, n_embd, heads, pe_method="abs", rpe_mode=None, layer=0):
        super().__init__()
        assert pe_method in ['abs', 'rel'], 'only "abs" and "rel" two type!!'
        self.is_rpe = (pe_method == 'rel' and layer == 0)
        self.layer = 0
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.attn = Attention(in_c, n_embd, heads) if not self.is_rpe else RPEAttention(in_c, n_embd, heads, rpe_mode)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(n_embd, 2 * n_embd)),
            ('act', nn.GELU()), # default: nn.GELU(approximate='tanh')
            ('c_proj', nn.Linear(2 * n_embd, n_embd))
        ]))
        print("block pe method", pe_method)
    def forward(self, x):
        if self.is_rpe:
            x, ct = x
        B, T, C = x.size()
        if B == 0:
            return (torch.zeros(B,T,T).to(x.device),ct)
        
        if self.is_rpe:
            assert ct is not None
            x = self.ln_1(x + self.attn(x, ct))
        else:
            x = self.ln_1(x + self.attn(x))
        
        x = self.ln_2(x + self.mlp(x))
        return (x, ct)

"""
Relative Position Encoding on Attention, 
ref: https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.pdf
test bias mode with euclidean distance
"""
class RPEAttention(nn.Module):
    def __init__(self, in_c, n_embd, heads, rpe_mode='bias', num_buckets=None):
        super(RPEAttention, self).__init__()
        assert rpe_mode in ['bias', 'contextual']
        self.rpe_mode = rpe_mode
        self.heads = heads
        self.head_dim = n_embd // heads
        self.n_embd = n_embd
        self.c_attn = nn.Linear(in_c, 3 * n_embd)
        size = 500
        self.register_buffer('bias', (torch.ones(size, size)).view(1, 1, size, size)) # mask for causal attention
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # relative
        self.patch = 16 # origin 16, v1: 16
        self.pix_per_patch = 200 // self.patch # origin 200, v1 200
        self.alpha = 12 # origin 8, v1: 12
        self.beta = 2 * self.alpha
        self.gamma = 8 * self.alpha

        # rpe
        self.num_buckets = 2 * self.beta + 1 if num_buckets is None else num_buckets
        
        def initializer(x): return None
        self.initializer = initializer

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # initialize parameters of iRPE
        # transposed
        if self.rpe_mode == 'bias':
            self.lookup_table_bias = nn.Parameter(torch.zeros(self.heads, self.num_buckets))
        else:
            self.lookup_table_weight = nn.Parameter(torch.zeros(self.heads, self.head_dim, self.num_buckets))
    def forward(self, x, ct):
        """
        x: input of attention, in shape (B,T,C)
        ct: the center of each T, in shape (B,T,2)
        """
        B, T, _ = ct.shape
        ct_1 = ct.reshape(B, T, 1, 2)
        ct_2 = ct.reshape(B, 1, T, 2)
        # convert distance into patch
        ct_ = (ct_1 - ct_2).square().sum(-1).float().sqrt() # absolute position in pixel
        ct_ = torch.div(ct_, self.pix_per_patch, rounding_mode='trunc')
        # convert patch to piecewise index
        bucket_ids = piecewise_index(ct_, self.alpha, self.beta, self.gamma, torch.int)
        beta_int = int(self.beta)
        bucket_ids = bucket_ids + beta_int
        bucket_ids = bucket_ids.view(B, T, T)

        # attention
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.heads, self.n_embd // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, self.n_embd // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, self.n_embd // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # add relative position encoding
        if self.rpe_mode == 'bias':
            rpe = self.lookup_table_bias[:, bucket_ids.flatten().long()].view(B, self.heads, T, T)
        else:
            lookup_table = torch.matmul(q.transpose(0,1).reshape(self.heads, B * T, self.head_dim), self.lookup_table_weight).\
                    view(-1, B, T, self.num_buckets)    # (H, B, T, BU)
            offset = torch.arange(0, T * self.num_buckets, self.num_buckets, dtype=bucket_ids.dtype, device=bucket_ids.device).view(-1, 1)
            rp_bucket = (bucket_ids + offset).long().flatten(-2)  # (B*T*T)
            batch_offset = torch.arange(0, B * T * self.num_buckets, T * self.num_buckets, dtype=rp_bucket.dtype, device=rp_bucket.device).view(-1, 1)
            brp_bucket = (rp_bucket + batch_offset).flatten()
            rpe = lookup_table.flatten(1)[:, brp_bucket].reshape(-1, B, T, T).transpose(0,1)
        att = att + rpe
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, self.n_embd)) # (b,H,T,R) -> (b,T,C)

class AttentionCombine(nn.Module):
    def __init__(self, in_c, num_points, n_embd, heads):
        super(AttentionCombine, self).__init__()
        self.heads = heads
        self.head_dim = n_embd // heads
        self.n_embd = n_embd
        self.contour_stride = 4
        self.contour_feat_proj = nn.Conv1d(in_c, n_embd, kernel_size=num_points//self.contour_stride)
        self.c_attn = nn.Linear(n_embd, 2 * n_embd)
        self.p_proj = nn.Conv1d(self.heads, 1, 1, bias=False)
        size = 500
        self.register_buffer('bias', (torch.ones(size, size)).view(1, 1, size, size)) # mask for causal attention
        # position encoding
        self.patch = 16
        pos_embed = getPositionEmbeddingSine(num_pos_feats=n_embd, patch=self.patch, normalize=True)
        self.register_buffer('pos_embed', pos_embed)
        self.ct_score = 0.2
    def forward(self, data_input, output, cnn_feature, from_dataset=True):
        if from_dataset:
            Q, K = self.preprocess_from_dataset(data_input, output, cnn_feature)
        else:
            img_hw = data_input['inp'].shape[-2:]
            Q, K = self.preprocess_from_model(output, cnn_feature, img_hw)
        
        # multiheads
        B,T,C = Q.shape
        if B == 0:
            return torch.zeros(B,T,T).to(Q.device)
        K = K.view(B, T, self.heads, C//self.heads).transpose(1,2)
        Q = Q.view(B, T, self.heads, C//self.heads).transpose(1,2)
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        """
            B: batch size, H: heads, T: max_len
            now the att shape is (B,H,T,T), what i need is (B,T,T)
            the last layer is probability, and i want probability of H heads could be one probs
            then reshape to (B, H, T * T), then 1*1 convolution to get (B, 1, T * T)
            end, reshape back to (B,T,T)
        """
        att = att.reshape(B, self.heads, T * T)
        att = self.p_proj(att).reshape(B, T, T)
        att = torch.sigmoid(att)
        att = torch.where(torch.isnan(att), torch.full_like(att, 0), att)
        return att

    def preprocess_from_model(self, output, cnn_feature, img_hw):
        """
            what i need is cnn_features' idx for each polys if batch > 1
            others are the same
            TODO: get ct_01 and ct_img_idx
        """
        h, w = img_hw
        img_idx = output["img_idx"]
        cnn_idx, counts = torch.unique(img_idx, return_counts=True)
        if len(cnn_idx) == 0:
            batch_size, max_len = 0, 0
        else:
            batch_size, max_len = torch.max(cnn_idx) + 1, torch.max(counts)
        ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool, device=cnn_feature.device)
        for idx, count in zip(cnn_idx, counts):
            ct_01[idx.item(), :count.item()] = 1
        contours = output['py'][-1]
        contour_sample = contours[:,::self.contour_stride]
        contour_features = get_gcn_feature(cnn_feature, contour_sample, img_idx, h, w)
        normed_contours = torch.div(contour_sample, torch.tensor([w,h], device=contours.device))
        contour_features = torch.cat([contour_features, normed_contours.permute(0,2,1)], dim=1)    

        """
            add position encoding
            get ct from output['detection'][:, :2]
            ct_x, ct_y = ct
        """
        height, width = cnn_feature.shape[-2:]
        ct_x, ct_y = output['detection'][:, 0], output['detection'][:, 1]
        ct_x = torch.div(ct_x * self.patch, width, rounding_mode='floor').to(torch.int64)
        ct_y = torch.div(ct_y * self.patch, height, rounding_mode='floor').to(torch.int64)

        b, c, p = contour_features.shape
        batch_size, max_len = ct_01.shape
        contour_features = self.contour_feat_proj(contour_features).squeeze(-1)
        # add position encoding on contour features
        pos_embed = self.pos_embed[:, ct_y, ct_x].transpose(1,0)
        contour_features = contour_features + pos_embed

        q, k = self.c_attn(contour_features).split(self.n_embd, dim=1)
        
        pad_q, pad_k = torch.zeros(batch_size, max_len, self.n_embd, device=cnn_feature.device), torch.zeros(batch_size, max_len, self.n_embd, device=cnn_feature.device)
        pad_q[ct_01] = q
        pad_k[ct_01] = k
        output.update({'ct_01': ct_01.detach()})
        return pad_q, pad_k

    def preprocess_from_dataset(self, data_input, output, cnn_feature):
        ct_01 = data_input['ct_01'].bool()
        ct_img_idx = data_input['ct_img_idx'][ct_01]
        contours = output['py_pred'][-1] # get the last contours
        h, w = data_input['inp'].shape[-2:]
        # get cnn features, concat with coordinates
        contour_sample = contours[:, ::self.contour_stride]
        contour_features = get_gcn_feature(cnn_feature, contour_sample, ct_img_idx, h, w)
        # normed contour coordinates into [0,1]
        normed_contours = torch.div(contour_sample, torch.tensor([w,h], device=contours.device))
        contour_features = torch.cat([contour_features, normed_contours.permute(0,2,1)], dim=1)    
        
        """
            prepare center points for position embedding
        """
        batch, _, height, width = cnn_feature.shape
        ct_ind = data_input['ct_ind'][ct_01]
        # ct_x: (batch, max_len)
        ct_x, ct_y = ct_ind % width, torch.div(ct_ind, width, rounding_mode='floor')
        ct_x = torch.div(ct_x * self.patch, width, rounding_mode='floor')
        ct_y = torch.div(ct_y * self.patch, height, rounding_mode='floor')

        """
            contour_features shape: (total num, channel, points) 
            channel now is 64 + 2, points is 128
            to go through the next step, turn it to (batch_size, max_len, channel * points)
            TODO: this is resource consuming, thus take 32 points for representation.
                    strides = 128 / 32 = 4
                    and the features could be lower by conv1d
        """
        b, c, p = contour_features.shape
        batch_size, max_len = ct_01.shape
        contour_features = self.contour_feat_proj(contour_features).squeeze(-1)
        # add position encoding on contour features
        pos_embed = self.pos_embed[:, ct_y, ct_x].transpose(1,0)
        contour_features = contour_features + pos_embed
        
        q, k = self.c_attn(contour_features).split(self.n_embd, dim=1)
        # padding to max length
        pad_q, pad_k = torch.zeros(batch_size, max_len, self.n_embd, device=q.device), torch.zeros(batch_size, max_len, self.n_embd, device=q.device)
        pad_q[ct_01] = q
        pad_k[ct_01] = k
        
        return pad_q, pad_k
        
def getPositionEmbeddingSine(num_pos_feats=64, patch=16, temperature=10000, normalize=False, scale=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi
    x = torch.arange(patch, dtype=torch.float32)
    y = torch.arange(patch, dtype=torch.float32)
    y_embed, x_embed = torch.meshgrid(x, y)
    if normalize:  # no normalize
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, -1:] + eps) * scale
    
    dim_t = torch.arange(num_pos_feats//2, dtype=torch.float32)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
    return pos

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, patch=16, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.patch = patch

    def forward(self, cts):
        device = cts.device
        x = torch.arange(self.patch, dtype=torch.float32)
        y = torch.arange(self.patch, dtype=torch.float32)
        y_embed, x_embed = torch.meshgrid(x, y, device=device, dtype=torch.float32)
        if self.normalize:  # no normalize
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        return pos

if __name__ == "__main__":
    model = RPEAttention(64, 256, 8)
    B = 2
    T = 10
    x = torch.randn(B,T,64)
    ct = torch.rand(B,T,2) * 800
    print(ct)
    print(model(x, ct).shape)
