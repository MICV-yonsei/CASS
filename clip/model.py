### CLIP source code from OpenAI:
# https://github.com/openai/CLIP/blob/main/clip/clip.py

from collections import OrderedDict
from typing import Tuple, Union

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
import pdb

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.width = width
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.arch, self.attn_strategy, self.gaussian_std = None, None, 0
        self.addition_cache = dict()

    def set_params(self, arch, attn_strategy, gaussian_std):
        assert arch in ['reduced']
        assert attn_strategy in ['spectral', 'sequential', 'vanilla']
        assert gaussian_std > 0
        self.arch, self.attn_strategy, self.gaussian_std = arch, attn_strategy, gaussian_std

    def forward(
            self, 
            x: torch.Tensor, 
            inv_image, 
            dino_type, 
            dino_model, 
            dataset, 
            return_all=True, 
            return_cls=False
            ):
        
        B, nc, w, h = x.shape
        n_patches = (w // self.patch_size, h // self.patch_size)

        small_v = min(w,h)
        inv_image = inv_image.resize((small_v, small_v))

        with torch.no_grad():
            dino_attn = dino_model(inv_image)
            dino_attn = dino_attn.half()

        if dino_type == 'dino_vitb8':
            dino_n_patches = (small_v // 8, small_v // 8)
        elif dino_type == 'dinov2_vitb14':
            dino_n_patches = (518 // 14, 518 // 14)
        # pdb.set_trace()

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for blk in self.transformer.resblocks[:-1]:
            x = blk(x)
        blk = self.transformer.resblocks[-1]

        x_last = blk(x)
        x_last = x_last.permute(1, 0, 2)  # LND -> NLD [197, 1, 768]
        x_last = self.ln_post(x_last[:,0,:]) # [1, 768]
        x_last = x_last @ self.proj # [1,512]

        if return_cls:
            return x_last

        if self.arch == 'reduced':
            x = self.custom_attn(blk.attn, blk.ln_1(x), n_patches, dino_type, dino_attn, dino_n_patches, dataset)
        else:
            raise NotImplemented

        x = x.permute(1, 0, 2)  # LND -> NLD

        if return_all:
            return self.ln_post(x) @ self.proj, x_last

        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), mode='bicubic',
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    @staticmethod
    def gaussian_window(dim1, dim2, std=1.):
        constant = 1 / (std * math.sqrt(2))
        ks = list()
        for dim in [dim1, dim2]:
            start = -(dim - 1) / 2.0
            k = torch.linspace(start=start * constant,
                               end=(start + (dim - 1)) * constant,
                               steps=dim,
                               dtype=torch.float)
            ks.append(k)
        dist_square_to_mu = (torch.stack(torch.meshgrid(*ks, indexing='ij')) ** 2).sum(0)
        return torch.exp(-dist_square_to_mu)

    @staticmethod
    def get_attention_addition(dim1, dim2, window, adjust_for_cls=False):
        m = torch.einsum('ij,kl->ijkl', torch.eye(dim1), torch.eye(dim2))
        m = m.permute((0, 3, 1, 2)).contiguous()  # m[ijkl] = 1 iff (i, j) == (k, l)
        out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1), window.unsqueeze(0).unsqueeze(1), padding='same').squeeze(1)
        out = out.view(dim1 * dim2, dim1 * dim2)
        if adjust_for_cls:
            v_adjusted = torch.vstack([torch.zeros((1, dim1 * dim2)), out])
            out = torch.hstack([torch.zeros((dim1 * dim2 + 1, 1)), v_adjusted])
        return out
    
    @staticmethod
    def optimal_q(dino_key_adj, energy_threshold=0.95, initial_q=10, step_size=5):
        max_q = min(dino_key_adj.shape)
        current_q = initial_q

        while current_q <= max_q:
            u, s, vh = torch.svd_lowrank(dino_key_adj.type(torch.float64), q=current_q)

            total_energy = torch.sum(s)
            cumulative_energy = torch.cumsum(s, dim=0)
            if (cumulative_energy / total_energy >= energy_threshold).any():
                optimal_q = torch.searchsorted(cumulative_energy / total_energy, energy_threshold).item() + 1
                return optimal_q, u, s, vh
            current_q += step_size

        return max_q, u, s, vh
    
    
    @staticmethod
    def compute_spectrum(adj_matrices, k_eigen=30):
        eigenvalues = []
        for i in range(adj_matrices.shape[0]):
            adj = adj_matrices[i]
            vals = torch.linalg.eigvalsh(adj) 
            vals, _ = torch.sort(vals, descending=True)
            vals = vals[:k_eigen]
            eigenvalues.append(vals)
        return torch.stack(eigenvalues)

    @staticmethod
    def wasserstein_distance_pytorch(u_values, v_values):
        u_sort, _ = torch.sort(u_values)
        v_sort, _ = torch.sort(v_values)
        return torch.sum(torch.abs(u_sort - v_sort))

    def custom_attn(
        self, 
        attn_layer: nn.MultiheadAttention, 
        x: torch.Tensor, 
        n_patches: Tuple[int, int], 
        dino_type: str, 
        dino_attn: torch.Tensor, 
        dino_n_patches: Tuple[int, int], 
        dataset: str, 
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        config = {
            'return_attn': False,
            'with_attn': False,
            'scale_factor': 10.0,
            'k_eigen': 20,
            'energy_threshold': 0.95,
        }
        config.update(kwargs)
        
        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size() 
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        q, v, k = q[:, 1:, :], v[:, 1:, :], k[:, 1:, :]

        dino_key = self._get_dino_key(dino_type, dino_attn)
        dino_key_final = self._process_dino_key(dino_key, num_heads, head_dim, dino_n_patches, n_patches)
        dino_key_final = self._normalize_key_range(dino_key_final, k)
        addition = self._get_attention_addition(n_patches)
        
        if self.attn_strategy == 'spectral':
            attn_weights = self._apply_spectral_strategy(
                k, dino_key_final, scale, num_heads, 
                config['k_eigen'], config['scale_factor'], 
                config['energy_threshold'], dataset
            )
            omega = addition
            
        elif self.attn_strategy == 'sequential':
            attn_weights = self._apply_sequential_strategy(
                k, dino_key_final, scale, num_heads, 
                config['energy_threshold'], dataset
            )
            omega = addition
            
        elif self.attn_strategy == 'vanilla':
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            raise NotImplementedError(f'attn_strategy {self.attn_strategy} is not implemented')
        
        if self.attn_strategy != 'vanilla':
            attn_weights += omega
            attn_weights = F.softmax(attn_weights, dim=-1)

        if config['return_attn']:
            return attn_weights

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        if config['with_attn']:
            return attn_output, attn_weights

        return attn_output
        
    def _get_dino_key(self, dino_type: str, dino_attn: torch.Tensor) -> torch.Tensor:

        if dino_type == 'dino_vitb8':
            return dino_attn[1, :, :, 1:, :].squeeze(0).permute(0, 2, 1)
        elif dino_type == 'dinov2_vitb14':
            return dino_attn[1, :, :, 5:, :].squeeze(0).permute(0, 2, 1)
        else:
            raise ValueError(f"Unsupported DINO type: {dino_type}")
            
    def _process_dino_key(
        self, 
        dino_key: torch.Tensor, 
        num_heads: int, 
        head_dim: int, 
        dino_n_patches: Tuple[int, int], 
        n_patches: Tuple[int, int]
    ) -> torch.Tensor:

        dino_key_reshaped = dino_key.reshape(num_heads, head_dim, dino_n_patches[0], dino_n_patches[1])
        dino_key_interpolated = F.interpolate(
            dino_key_reshaped, 
            size=n_patches, 
            mode='bilinear', 
            align_corners=False
        )
        return dino_key_interpolated.reshape(
            num_heads, head_dim, n_patches[0] * n_patches[1]
        ).permute(0, 2, 1)
        
    def _normalize_key_range(self, dino_key: torch.Tensor, k: torch.Tensor) -> torch.Tensor:

        k_min, k_max = k.min(), k.max()
        dino_min, dino_max = dino_key.min(), dino_key.max()

        normalized_key = (dino_key - dino_min) / (dino_max - dino_min)

        return normalized_key * (k_max - k_min) + k_min
        
    def _get_attention_addition(self, n_patches: Tuple[int, int]) -> torch.Tensor:

        addition = self.addition_cache.get(n_patches)
        
        if addition is None:
            window_size = [side * 2 - 1 for side in n_patches]
            window = self.gaussian_window(*window_size, std=self.gaussian_std)
            addition = self.get_attention_addition(
                *n_patches, window
            ).unsqueeze(0).to(self.class_embedding.dtype).to(self.class_embedding.device)
            self.addition_cache[n_patches] = addition
            
        return addition
        
    def _apply_spectral_strategy(
        self, 
        k: torch.Tensor, 
        dino_key_final: torch.Tensor, 
        scale: float, 
        num_heads: int, 
        k_eigen: int,
        scale_factor: float,
        energy_threshold: float,
        dataset: str
    ) -> torch.Tensor:

        attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale
        dino_attn_weights = torch.bmm(dino_key_final, dino_key_final.transpose(1, 2)) * scale
        
        dino_eigenvalues = self.compute_spectrum(dino_attn_weights.type(torch.float64), k_eigen=k_eigen)
        clip_eigenvalues = self.compute_spectrum(attn_weights.type(torch.float64), k_eigen=k_eigen)
        
        dino_eigenvalues = F.normalize(dino_eigenvalues, p=2, dim=1)
        clip_eigenvalues = F.normalize(clip_eigenvalues, p=2, dim=1)
        
        cost_matrix = torch.zeros(num_heads, num_heads).to(k.device)
        for i in range(num_heads):
            dino_vals = dino_eigenvalues[i].unsqueeze(0)
            for j in range(num_heads):
                clip_vals = clip_eigenvalues[j].unsqueeze(0)
                
                dino_vals_prob = F.normalize(dino_vals, p=1, dim=-1)
                clip_vals_prob = F.normalize(clip_vals, p=1, dim=-1)
                
                correlation = self.wasserstein_distance_pytorch(dino_vals_prob, clip_vals_prob)
                cost_matrix[i, j] = 1 - correlation
        
        cost_matrix_cpu = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu)
        
        combined_attn = torch.zeros_like(attn_weights)
        
        for idx in range(num_heads):
            i, j = row_ind[idx], col_ind[idx]
            w_d = 1 - cost_matrix[i, j]
            
            dino_key_adj = dino_attn_weights[j]
            clip_key_adj = attn_weights[i]
            
            _, u_topk, s_topk, v_topk = self.optimal_q(
                dino_key_adj, 
                energy_threshold=energy_threshold, 
                initial_q=10, 
                step_size=5
            )
            
            s_scaled = self._scale_singular_values(s_topk, dataset)
            
            dino_sparse_approx = u_topk @ torch.diag(s_scaled) @ v_topk.T
            dino_sparse_approx.fill_diagonal_(0)
            
            dino_sparse_adj = self._normalize_attention_range(dino_sparse_approx, clip_key_adj)
            
            combined_attn[idx] = (w_d * scale_factor * dino_sparse_adj + clip_key_adj) / (w_d * scale_factor + 1)
        
        return combined_attn
        
    def _apply_sequential_strategy(
        self, 
        k: torch.Tensor, 
        dino_key_final: torch.Tensor, 
        scale: float, 
        num_heads: int,
        energy_threshold: float,
        dataset: str
    ) -> torch.Tensor:

        attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale
        dino_attn_weights = torch.bmm(dino_key_final, dino_key_final.transpose(1, 2)) * scale
        
        combined_attn = torch.zeros_like(attn_weights)
        
        for idx in range(num_heads):
            dino_key_adj = dino_attn_weights[idx]
            clip_key_adj = attn_weights[idx]
            
            _, u_topk, s_topk, v_topk = self.optimal_q(
                dino_key_adj, 
                energy_threshold=energy_threshold, 
                initial_q=10, 
                step_size=5
            )
                        
            dino_sparse_approx = u_topk @ torch.diag(s_topk) @ v_topk.T
            dino_sparse_approx.fill_diagonal_(0)
            
            dino_sparse_adj = self._normalize_attention_range(dino_sparse_approx, clip_key_adj)
            
            combined_attn[idx] = (dino_sparse_adj + clip_key_adj) / 2
        
        return combined_attn
        
    def _scale_singular_values(self, s_topk: torch.Tensor, dataset: str) -> torch.Tensor:

        if dataset in ['voc20', 'city_scapes']:
            return s_topk
        else:
            s_new = s_topk.clone()
            singular_min, singular_max = s_topk.min(), s_topk.max()
            scaled_range = (1.5 * singular_max) - (0.5 * singular_min)
            return (s_new - singular_min) / (singular_max - singular_min) * scaled_range + (0.5 * singular_min)
            
    def _normalize_attention_range(self, source_attn: torch.Tensor, target_attn: torch.Tensor) -> torch.Tensor:

        source_min, source_max = source_attn.min(), source_attn.max()
        target_min, target_max = target_attn.min(), target_attn.max()
        normalized = (source_attn - source_min) / (source_max - source_min)
        
        return normalized * (target_max - target_min) + target_min


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,  # 512
                 # vision
                 image_resolution: int,  # 224
                 vision_layers: Union[Tuple[int, int, int, int], int],  # 12
                 vision_width: int,  # 768
                 vision_patch_size: int,  # 16
                 # text
                 context_length: int,  # 77
                 vocab_size: int,  # 49408
                 transformer_width: int,  # 512
                 transformer_heads: int,  # 8
                 transformer_layers: int  # 12
                 ):
        super().__init__()
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, inv_image, dino_type, dino_model, dataset, return_all=False, return_cls=False):
        return self.visual(image.type(self.dtype), inv_image, dino_type, dino_model, dataset, return_all=return_all, return_cls=return_cls)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
