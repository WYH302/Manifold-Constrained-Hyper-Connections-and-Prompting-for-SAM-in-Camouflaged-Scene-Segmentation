
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import numpy as np
from typing import Optional, Tuple, Type
import math


from segment_anything.modeling.common import LayerNorm2d, MLPBlock



class SinkhornProjection(nn.Module):
    

    def __init__(self, iters=3, epsilon=1e-8):
        super().__init__()
        self.iters = iters
        self.epsilon = epsilon

    def forward(self, cost_matrix):
        
        
        log_K = -cost_matrix

        
        log_K = log_K - torch.max(log_K, dim=-1, keepdim=True)[0].detach()

        for i in range(self.iters):
            
            log_K = log_K - torch.logsumexp(log_K, dim=-1, keepdim=True)
            
            log_K = log_K - torch.logsumexp(log_K, dim=-2, keepdim=True)

        K = torch.exp(log_K)
        return K



class RankDiceRMAModule(nn.Module):
    

    def __init__(self, use_in_training=True, weight=0.1, eps=1e-8):
        super().__init__()
        self.use_in_training = use_in_training
        self.weight = weight
        self.eps = eps

        
        self.temperature = nn.Parameter(torch.tensor(0.1))

        
        self.register_buffer('step', torch.tensor(0))

    def compute_optimal_threshold(self, prob_map):
        
        B, C, H, W = prob_map.shape
        thresholds = []

        for b in range(B):
            single_prob = prob_map[b, 0]
            d = single_prob.numel()

            
            p_sorted, _ = torch.sort(single_prob.flatten(), descending=True)

            
            q_tau = torch.cumsum(p_sorted, dim=0)
            mu = p_sorted.sum()

            
            tau_range = torch.arange(1, d + 1, device=prob_map.device, dtype=torch.float32)
            pi_rma = 2 * q_tau / (tau_range + mu + self.eps)

            
            tau_star = torch.argmax(pi_rma) + 1
            threshold_value = p_sorted[tau_star - 1]
            thresholds.append(threshold_value)

        return torch.stack(thresholds)

    def forward(self, logits, gt_mask=None):
        
        if self.training and gt_mask is not None and self.use_in_training:
            return self._compute_rank_dice_loss(logits, gt_mask)
        else:
            return self._inference(logits)

    def _compute_rank_dice_loss(self, logits, gt_mask):
        
        prob_map = torch.sigmoid(logits)
        batch_size = prob_map.shape[0]

        
        thresholds = self.compute_optimal_threshold(prob_map)

        total_loss = 0.0
        for i in range(batch_size):
            single_prob = prob_map[i, 0]
            single_gt = gt_mask[i, 0]
            threshold_value = thresholds[i]

            
            temp = torch.clamp(self.temperature, min=0.01)
            pred_mask = torch.sigmoid((single_prob - threshold_value) / temp)

            
            intersection = (pred_mask * single_gt).sum()
            union = pred_mask.sum() + single_gt.sum() + self.eps
            dice = (2.0 * intersection + self.eps) / union

            dice_loss = 1.0 - dice
            total_loss += dice_loss

        rank_dice_loss = (total_loss / batch_size) * self.weight
        self.step += 1

        return logits, rank_dice_loss

    def _inference(self, logits):
        
        prob_map = torch.sigmoid(logits)
        thresholds = self.compute_optimal_threshold(prob_map)

        binary_masks = []
        for i in range(prob_map.shape[0]):
            single_prob = prob_map[i, 0]
            threshold_value = thresholds[i]
            binary_mask = (single_prob >= threshold_value).float()
            binary_masks.append(binary_mask.unsqueeze(0).unsqueeze(0))

        return torch.cat(binary_masks, dim=0)



class HyperCondModule(nn.Module):
    

    def __init__(self, cond_dim=128, hidden_dim=64):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        
        self.threshold_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim // 2)
        )

        self.boundary_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim // 2)
        )

        
        self.fusion_net = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.LayerNorm(cond_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, threshold, boundary_weight, device, batch_size=1):
        
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor([threshold], dtype=torch.float32)
        if not isinstance(boundary_weight, torch.Tensor):
            boundary_weight = torch.tensor([boundary_weight], dtype=torch.float32)

        
        threshold = threshold.to(device).view(-1, 1)
        boundary_weight = boundary_weight.to(device).view(-1, 1)

        if threshold.shape[0] == 1 and batch_size > 1:
            threshold = threshold.expand(batch_size, -1)
        if boundary_weight.shape[0] == 1 and batch_size > 1:
            boundary_weight = boundary_weight.expand(batch_size, -1)

        
        threshold_enc = self.threshold_encoder(threshold)
        boundary_enc = self.boundary_encoder(boundary_weight)

        
        cond = torch.cat([threshold_enc, boundary_enc], dim=-1)
        cond = self.fusion_net(cond)

        return cond



class RMSNorm(nn.Module):
    

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm



class ManifoldConstrainedAdapter(nn.Module):
    

    def __init__(self, embed_dim, n_streams=4, sinkhorn_iters=20):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.embed_dim = embed_dim

        
        self.input_adjust = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        
        self.norm = RMSNorm(embed_dim)

        
        self.adapter_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim * n_streams)
        )

        
        self.alpha = nn.Parameter(0.01 * torch.ones(n_streams))

        
        self.mixing_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, n_streams * n_streams)
        )

        
        self.sinkhorn = SinkhornProjection(iters=sinkhorn_iters)

    def forward(self, sam_features, blip_features):
        
        B, N, C = sam_features.shape

        
        combined = torch.cat([sam_features, blip_features], dim=-1)
        combined = self.input_adjust(combined)

        
        combined = self.norm(combined)

        
        raw_residual = self.adapter_mlp(combined)
        raw_residual = raw_residual.view(B, N, self.n_streams, C)

        
        mixing_scores = self.mixing_mlp(combined)
        mixing_scores = mixing_scores.view(B, N, self.n_streams, self.n_streams)

        
        
        H_res = self.sinkhorn(mixing_scores)

        
        
        
        
        
        mixed_residual = torch.einsum('bnsa,bnac->bnsc', H_res, raw_residual)

        
        alpha_gate = F.softmax(self.alpha, dim=0)
        final_residual = torch.einsum('s,bnsc->bnc', alpha_gate, mixed_residual)

        
        output = sam_features + 0.1 * final_residual  

        return output



class CrossModalStablePromptGenerator(nn.Module):
    

    def __init__(
            self,
            text_dim: int = 768,
            vision_dim: int = 768,
            prompt_dim: int = 256,
            n_prompt: int = 2,
            use_sinkhorn: bool = True,
            sinkhorn_iters: int = 3,
            debug: bool = False,
    ):
        super().__init__()
        self.n_prompt = n_prompt
        self.prompt_dim = prompt_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.use_sinkhorn = use_sinkhorn
        self.debug = debug

        
        self.W_text = nn.Linear(text_dim, prompt_dim, bias=True)
        self.W_vision = nn.Linear(vision_dim, prompt_dim, bias=True)

        
        self.stream_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        
        self.modality_controller = nn.Sequential(
            nn.Linear(text_dim + vision_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        
        self.base_alpha = nn.Parameter(torch.tensor(0.0))
        self.base_beta = nn.Parameter(torch.tensor(0.0))

        
        self.rms_norm = RMSNorm(prompt_dim)

        
        self.sinkhorn = SinkhornProjection(iters=sinkhorn_iters) if use_sinkhorn else None

        
        self.register_buffer('text_ratio', torch.tensor(0.0))
        self.register_buffer('vision_ratio', torch.tensor(0.0))

        
        self.register_buffer('step', torch.tensor(0))
        self.warmup_steps = 1000

        self._init_weights()

    def _init_weights(self):
        
        nn.init.xavier_normal_(self.W_text.weight, gain=0.3)
        nn.init.xavier_normal_(self.W_vision.weight, gain=1.0)

        with torch.no_grad():
            self.W_text.bias.data.uniform_(-0.1, 0.1)
            self.W_vision.bias.data.uniform_(-0.1, 0.1)
            nn.init.xavier_normal_(self.modality_controller[0].weight)
            nn.init.xavier_normal_(self.modality_controller[2].weight)

    def get_lr_multiplier(self):
        
        if self.step < self.warmup_steps:
            return float(self.step) / self.warmup_steps
        return 1.0

    def compute_H_pre(self, text_score, vision_score, text_features, vision_features):
        
        B = text_score.shape[0]

        
        if len(text_features.shape) == 3:
            text_global = text_features.mean(dim=1)
        else:
            text_global = text_features

        if len(vision_features.shape) == 4:
            if vision_features.shape[-1] == self.vision_dim:
                vision_global = vision_features.mean(dim=[1, 2])
            elif vision_features.shape[1] == self.vision_dim:
                vision_global = vision_features.mean(dim=[2, 3])
            else:
                vision_global = vision_features.view(B, -1, self.vision_dim).mean(dim=1)
        elif len(vision_features.shape) == 3:
            vision_global = vision_features.mean(dim=1)
        else:
            vision_global = vision_features

        
        if vision_global.shape[-1] != self.vision_dim:
            vision_global = vision_global.view(B, -1)
            if vision_global.shape[-1] > self.vision_dim:
                
                vision_global = vision_global.view(B, self.vision_dim, -1).mean(dim=-1)
            elif vision_global.shape[-1] < self.vision_dim:
                padding = torch.zeros(B, self.vision_dim - vision_global.shape[-1]).to(vision_global.device)
                vision_global = torch.cat([vision_global, padding], dim=-1)

        
        combined = torch.cat([text_global, vision_global], dim=-1)
        modality_weights = self.modality_controller(combined)

        
        scores = torch.cat([text_score, vision_score], dim=-1)
        temperature = 0.5
        raw_probs = torch.softmax(scores / temperature, dim=-1)

        
        alpha_base = 0.5 + 0.5 * torch.sigmoid(self.base_alpha)
        beta_base = 0.5 + 0.5 * torch.sigmoid(self.base_beta)

        text_ratio = alpha_base * modality_weights[:, 0:1] + (1 - alpha_base) * raw_probs[:, 0:1]
        vision_ratio = beta_base * modality_weights[:, 1:2] + (1 - beta_base) * raw_probs[:, 1:2]

        
        text_ratio = 0.4 + 0.2 * torch.sigmoid(text_ratio)
        vision_ratio = 0.4 + 0.2 * torch.sigmoid(vision_ratio)

        
        H_pre = torch.zeros(B, self.n_prompt, self.n_prompt, device=text_score.device)
        H_pre[:, 0, 0] = text_ratio.squeeze()
        H_pre[:, 0, 1] = 1.0 - text_ratio.squeeze()
        H_pre[:, 1, 0] = 1.0 - vision_ratio.squeeze()
        H_pre[:, 1, 1] = vision_ratio.squeeze()

        
        if self.use_sinkhorn and self.sinkhorn is not None:
            H_pre = self.sinkhorn(H_pre)

        
        text_ratio_val = H_pre[:, 0, 0].mean().detach()
        vision_ratio_val = H_pre[:, 1, 1].mean().detach()

        self.text_ratio.copy_(text_ratio_val)
        self.vision_ratio.copy_(vision_ratio_val)

        return H_pre, text_ratio_val, vision_ratio_val

    def compute_H_post(self):
        
        H_post = torch.softmax(self.stream_weights * 2.0, dim=0)
        return H_post

    def forward(self, text_embed, vision_embed, text_features, vision_features):
        
        B = text_embed.shape[0]

        
        text_proj = self.W_text(text_embed)
        vision_proj = self.W_vision(vision_embed)

        
        text_score = torch.mean(text_proj, dim=-1, keepdim=True) + torch.std(text_proj, dim=-1, keepdim=True) * 0.5
        vision_score = torch.max(vision_proj, dim=-1, keepdim=True)[0] + torch.min(vision_proj, dim=-1, keepdim=True)[
            0] * 0.5

        
        H_pre, text_ratio, vision_ratio = self.compute_H_pre(
            text_score, vision_score, text_features, vision_features
        )

        
        input_streams = torch.stack([text_proj, vision_proj], dim=1)
        mixed_streams = torch.einsum('bnk,bkd->bnd', H_pre, input_streams)

        
        H_post = self.compute_H_post()

        
        
        
        sparse_prompt = mixed_streams * H_post.unsqueeze(0).unsqueeze(-1)  

        
        prompt_list = []
        for i in range(self.n_prompt):
            prompt_list.append(self.rms_norm(sparse_prompt[:, i, :]))
        sparse_prompt = torch.stack(prompt_list, dim=1)  

        
        if self.training:
            self.step += 1

        return sparse_prompt

    def apply_gradient_constraints(self):
        
        if hasattr(self, 'stream_weights') and self.stream_weights.grad is not None:
            torch.nn.utils.clip_grad_norm_([self.stream_weights], max_norm=1.0)

        
        for name, param in self.named_parameters():
            if param.grad is not None and (
                     in name or 'base_alpha' in name or 'base_beta' in name):
                torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)



class BoundaryAwareLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, boundary_weight=0.5, kernel_size=3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size

        self.dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        self.ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

        
        self.kernel_size = kernel_size

    def _create_kernel(self, device):
        
        return torch.ones(1, 1, self.kernel_size, self.kernel_size, device=device)

    def get_boundary_mask(self, gt):
        
        with torch.no_grad():
            
            gt_binary = (gt > 0.5).float()

            
            pad_size = self.kernel_size // 2
            padded = F.pad(gt_binary, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

            
            kernel = self._create_kernel(gt_binary.device)

            
            eroded = F.conv2d(padded, kernel, padding=0, stride=1)
            eroded = (eroded == kernel.sum()).float()

            
            dilated = F.conv2d(padded, kernel, padding=0, stride=1)
            dilated = (dilated > 0).float()

            
            boundary = dilated - eroded

            
            boundary = boundary[:, :, pad_size:-pad_size, pad_size:-pad_size]

            
            if boundary.shape[-2:] != gt_binary.shape[-2:]:
                boundary = F.interpolate(
                    boundary,
                    size=gt_binary.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            
            if self.kernel_size > 1:
                boundary = F.avg_pool2d(boundary, kernel_size=3, stride=1, padding=1)

            
            if boundary.shape[-2:] != gt_binary.shape[-2:]:
                boundary = F.interpolate(
                    boundary,
                    size=gt_binary.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            
            if boundary.max() > 0:
                boundary = boundary / (boundary.max() + 1e-8)

            
            assert boundary.shape == gt_binary.shape,                f"边界掩码尺寸不匹配: {boundary.shape} != {gt_binary.shape}"

            return boundary

    def forward(self, pred, gt, boundary_weight=None):
        if boundary_weight is None:
            boundary_weight = self.boundary_weight

        
        dice_ce_loss = self.dice_loss(pred, gt) + self.ce_loss(pred, gt)

        
        boundary_mask = self.get_boundary_mask(gt)
        boundary_pixels = boundary_mask.sum()

        if boundary_pixels > 10:  
            
            pred_sigmoid = torch.sigmoid(pred)

            
            intersection = (pred_sigmoid * boundary_mask * gt).sum()
            union = (pred_sigmoid * boundary_mask).sum() + (boundary_mask * gt).sum() + 1e-8
            boundary_dice = 1.0 - (2.0 * intersection) / union

            
            boundary_ce = F.binary_cross_entropy_with_logits(
                pred * boundary_mask,
                gt * boundary_mask,
                reduction='mean'
            )

            boundary_loss = (boundary_dice + boundary_ce) * 0.5 * boundary_weight * self.beta
        else:
            boundary_loss = torch.tensor(0.0).to(pred.device)

        
        total_loss = self.alpha * dice_ce_loss + boundary_loss

        return total_loss, dice_ce_loss, boundary_loss



class Attention(nn.Module):
    

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = self.add_decomposed_rel_pos(attn, q, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x

    def add_decomposed_rel_pos(self, attn, q, q_size, k_size):
        
        q_h, q_w = q_size
        k_h, k_w = k_size

        Rh = self.get_rel_pos(q_h, k_h, self.rel_pos_h)
        Rw = self.get_rel_pos(q_w, k_w, self.rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
                attn.view(B, q_h, q_w, k_h, k_w)
                + rel_h[:, :, :, :, None]
                + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        
        if rel_pos.shape[0] != max_rel_dist:
            
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]


class Block(nn.Module):
    

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = self.window_partition(x, self.window_size)

        x = self.attn(x)

        
        if self.window_size > 0:
            x = self.window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def window_partition(self, x: torch.Tensor, window_size: int):
        
        B, H, W, C = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = (
            x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        )
        return windows, (Hp, Wp)

    def window_unpartition(self, windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]):
        
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.view(
            B, Hp // window_size, Wp // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x


class PatchEmbed(nn.Module):
    

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        
        x = x.permute(0, 2, 3, 1)
        return x


class IntegratedImageEncoderViT(nn.Module):
    

    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            n_streams: int = 4,
            blip_feature_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.blip_feature_dim = blip_feature_dim

        
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            grid_size = img_size // patch_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, grid_size, grid_size, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

        
        self.n_streams = n_streams
        self.manifold_adapters = nn.ModuleList()

        
        adapter_layers = []
        if depth >= 4:
            
            indices = [depth // 4, depth // 2, 3 * depth // 4]
            for idx in indices:
                if idx < depth:
                    adapter_layers.append(idx)

        for i in range(depth):
            if i in adapter_layers:
                self.manifold_adapters.append(
                    ManifoldConstrainedAdapter(embed_dim, n_streams=n_streams)
                )
            else:
                self.manifold_adapters.append(None)

        
        self.blip_feature_adjust = nn.Sequential(
            nn.Conv2d(blip_feature_dim, embed_dim, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.GELU()
        )

        print(f"初始化图像编码器: embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}")
        print(f"窗口注意力: window_size={window_size}, 全局注意力层: {global_attn_indexes}")
        print(f"BLIP特征维度: {blip_feature_dim}")
        print(f"适配器层: {adapter_layers}")

    def prepare_blip_features(self, blip_features, target_spatial_size):
        
        if blip_features is None:
            return None

        B = blip_features.shape[0]

        
        if len(blip_features.shape) == 3:
            
            seq_len, hidden_dim = blip_features.shape[1], blip_features.shape[2]

            
            spatial_size = int(math.sqrt(seq_len))
            if spatial_size * spatial_size == seq_len:
                
                blip_features = blip_features.reshape(B, spatial_size, spatial_size, hidden_dim)
                blip_features = blip_features.permute(0, 3, 1, 2)
            else:
                
                blip_features = blip_features.transpose(1, 2)  
                blip_features = F.adaptive_avg_pool1d(blip_features, 1)  
                blip_features = blip_features.unsqueeze(-1)  

        
        
        if blip_features.shape[1] != self.blip_feature_dim:
            
            blip_features = blip_features.permute(0, 2, 3, 1)  
            blip_features = F.adaptive_avg_pool1d(
                blip_features.reshape(-1, blip_features.shape[-1]).unsqueeze(1),
                self.blip_feature_dim
            ).squeeze(1).reshape(blip_features.shape[0], blip_features.shape[1], blip_features.shape[2],
                                 self.blip_feature_dim)
            blip_features = blip_features.permute(0, 3, 1, 2)  

        
        if blip_features.shape[2:] != target_spatial_size:
            blip_features = F.interpolate(
                blip_features,
                size=target_spatial_size,
                mode='bilinear',
                align_corners=False
            )

        
        blip_features = self.blip_feature_adjust(blip_features)  
        blip_features = blip_features.permute(0, 2, 3, 1)  

        return blip_features

    def forward(self, x: torch.Tensor, adapter_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        B, H, W, C = x.shape

        
        blip_features_prepared = None
        if adapter_input is not None:
            blip_features_prepared = self.prepare_blip_features(adapter_input, (H, W))
            if blip_features_prepared is not None:
                blip_features_prepared = blip_features_prepared.reshape(B, H * W, C)

        
        for i, (blk, adapter) in enumerate(zip(self.blocks, self.manifold_adapters)):
            x = blk(x)

            
            if adapter is not None and blip_features_prepared is not None:
                
                x_reshaped = x.reshape(B, H * W, C)

                
                x_reshaped = adapter(x_reshaped, blip_features_prepared)

                
                x = x_reshaped.reshape(B, H, W, C)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x



class MMSAM_Integrated(nn.Module):
    

    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder=None,
            image_feature_dim=768,
            use_rankdice=True,
            use_hypercond=True,
            n_streams=4,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.use_rankdice = use_rankdice
        self.use_hypercond = use_hypercond

        
        from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom
        self.pe_layer = PositionEmbeddingRandom(256 // 2)

        
        self.prompt_generator = CrossModalStablePromptGenerator(
            text_dim=768,
            vision_dim=image_feature_dim,
            prompt_dim=256,
            n_prompt=2,
            use_sinkhorn=True,
            sinkhorn_iters=5,
            debug=False
        )

        
        if use_hypercond:
            self.hyper_cond = HyperCondModule(cond_dim=128, hidden_dim=64)
            self.cond_channel_adapter = nn.Conv2d(128, 256, kernel_size=1, bias=False)
            self.cond_spatial_adapter = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.GELU()
            )

        
        if use_rankdice:
            self.rankdice_module = RankDiceRMAModule(use_in_training=True, weight=0.1)

        
        self.text_adapter = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        
        self.image_feature_projection = nn.Sequential(
            nn.Linear(image_feature_dim, 768),
            nn.GELU()
        ) if image_feature_dim != 768 else nn.Identity()

        self.pseudo_mask_embed = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GELU()
        )

        
        self.register_buffer('training_step', torch.tensor(0))

        self._init_weights()

    def _init_weights(self):
        
        
        new_modules = [self.text_adapter, self.pseudo_mask_embed]

        
        if not isinstance(self.image_feature_projection, nn.Identity):
            new_modules.append(self.image_feature_projection)

        if self.use_hypercond:
            new_modules.extend([self.cond_channel_adapter, self.cond_spatial_adapter])

        for module in new_modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, image, text_embeddings, image_features,
                gt_mask=None, hyper_cond=None, return_logits=False):
        
        B = image.shape[0]

        
        
        if len(image_features.shape) == 3:
            _, seq_len, hidden_dim = image_features.shape

            
            if hidden_dim != 768:
                image_features = self.image_feature_projection(image_features)

            
            if seq_len == 576:  
                h = w = 24
                image_features_for_sam = image_features.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            elif seq_len == 64 * 64:  
                h = w = 64
                image_features_for_sam = image_features.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            else:
                
                image_features_for_sam = image_features.mean(dim=1, keepdim=True)
                image_features_for_sam = image_features_for_sam.unsqueeze(-1)
                image_features_for_sam = F.interpolate(image_features_for_sam, size=(64, 64))
        else:
            image_features_for_sam = image_features

        
        image_features_original = image_features.clone()

        
        if hyper_cond is None:
            hyper_cond = {'threshold': 0.5, 'boundary_weight': 1.0}

        threshold = hyper_cond['threshold']
        boundary_weight = hyper_cond['boundary_weight']

        
        image_embedding = self.image_encoder(image, image_features_for_sam)

        
        
        cond_embedding = None
        if self.use_hypercond:
            cond_embedding = self.hyper_cond(threshold, boundary_weight, image_embedding.device, B)

        
        text_global = text_embeddings.mean(dim=1)

        
        if len(image_features_original.shape) == 4:
            vision_global = image_features_original.mean(dim=[2, 3])
        elif len(image_features_original.shape) == 3:
            vision_global = image_features_original.mean(dim=1)
        else:
            vision_global = image_features_original.flatten(1).mean(dim=1, keepdim=True)

        
        if vision_global.dim() == 1:
            vision_global = vision_global.unsqueeze(1)

        
        if self.training:
            self.training_step += 1

        
        sparse_prompt = self.prompt_generator(
            text_global, vision_global, text_embeddings, image_features_original
        )  

        
        dense_embeddings = self.pseudo_mask_embed(image_embedding)

        if cond_embedding is not None:
            
            cond_dense = cond_embedding.view(B, -1, 1, 1)  
            cond_dense = self.cond_channel_adapter(cond_dense)  
            cond_dense = F.interpolate(cond_dense, size=image_embedding.shape[-2:])
            cond_dense = self.cond_spatial_adapter(cond_dense)

            
            dense_embeddings = dense_embeddings + 0.1 * cond_dense

        
        image_pe = self.pe_layer((image_embedding.shape[2], image_embedding.shape[3])).unsqueeze(0)

        
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        
        logits = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        
        rank_dice_loss = torch.tensor(0.0).to(logits.device)

        if self.training:
            
            if self.use_rankdice and gt_mask is not None:
                logits, rank_dice_loss = self.rankdice_module(logits, gt_mask)

            if return_logits:
                return logits, rank_dice_loss, hyper_cond
            else:
                return torch.sigmoid(logits), rank_dice_loss, hyper_cond
        else:
            
            if self.use_rankdice:
                pred_mask = self.rankdice_module._inference(logits)
            else:
                pred_mask = torch.sigmoid(logits)

            return pred_mask, rank_dice_loss, hyper_cond

    def apply_gradient_constraints(self):
        
        
        if hasattr(self.prompt_generator, 'apply_gradient_constraints'):
            self.prompt_generator.apply_gradient_constraints()

        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def set_training_mode(self, mode=True):
        
        self.train(mode)
        if mode:
            self.training_step = torch.tensor(0)

        
        if hasattr(self, 'prompt_generator'):
            self.prompt_generator.train(mode)
        if hasattr(self, 'rankdice_module'):
            self.rankdice_module.train(mode)
        if hasattr(self, 'hyper_cond'):
            self.hyper_cond.train(mode)



def create_integrated_model(
        sam_checkpoint_path,
        model_type="vit_l",
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device="cuda",
        blip_feature_dim=768
):
    
    from segment_anything import sam_model_registry

    
    print(f"加载SAM模型: {model_type}")
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)

    
    if model_type == "vit_b":
        embed_dim = 768
        num_heads = 12
        depth = 12
        window_size = 14
        global_attn_indexes = [2, 5, 8, 11]
    elif model_type == "vit_l":
        embed_dim = 1024
        num_heads = 16
        depth = 24
        window_size = 14
        global_attn_indexes = [5, 11, 17, 23]
    elif model_type == "vit_h":
        embed_dim = 1280
        num_heads = 16
        depth = 32
        window_size = 14
        global_attn_indexes = [7, 15, 23, 31]
    else:
        
        embed_dim = 768
        num_heads = 12
        depth = 12
        window_size = 0
        global_attn_indexes = []

    print(f"SAM配置: embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}")
    print(f"窗口注意力: window_size={window_size}, 全局注意力层: {global_attn_indexes}")

    
    image_encoder = IntegratedImageEncoderViT(
        img_size=image_size,
        patch_size=16,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        out_chans=256,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=window_size,
        global_attn_indexes=global_attn_indexes,
        n_streams=n_streams,
        blip_feature_dim=blip_feature_dim
    )

    
    
    
    
    sam_encoder_state = sam_model.image_encoder.state_dict()
    integrated_encoder_state = image_encoder.state_dict()

    transferred_keys = []
    skipped_keys = []
    for key in sam_encoder_state:
        if key in integrated_encoder_state:
            if sam_encoder_state[key].shape == integrated_encoder_state[key].shape:
                integrated_encoder_state[key] = sam_encoder_state[key]
                transferred_keys.append(key)
            else:
                skipped_keys.append(
                    )
        else:
            skipped_keys.append(f"{key}: 不存在于集成编码器中")

    image_encoder.load_state_dict(integrated_encoder_state)
    print(f"[C1修复] 从SAM预训练模型成功迁移了 {len(transferred_keys)}/{len(sam_encoder_state)} 个参数")
    if skipped_keys:
        print(f"[C1修复] 跳过 {len(skipped_keys)} 个参数:")
        for sk in skipped_keys[:5]:  
            print(f"  - {sk}")
        if len(skipped_keys) > 5:
            print(f"  ... 及其余 {len(skipped_keys) - 5} 个")

    
    integrated_model = MMSAM_Integrated(
        image_encoder=image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        image_feature_dim=blip_feature_dim,
        use_rankdice=use_rankdice,
        use_hypercond=use_hypercond,
        n_streams=n_streams,
    ).to(device)

    return integrated_model



def create_optimizer_for_integrated_model(model, lr=0.0002, weight_decay=0.01):
    
    
    param_groups = []

    
    encoder_adapter_params = []
    for name, param in model.named_parameters():
        if 'image_encoder' in name and (
                 in name or 'manifold_adapters' in name or 'blip_feature_adjust' in name):
            encoder_adapter_params.append(param)

    if encoder_adapter_params:
        param_groups.append({
            : encoder_adapter_params,
            : lr * 1.0,
            : weight_decay,
            : 'encoder_adapters'
        })

    
    prompt_generator_params = []
    for name, param in model.named_parameters():
        if 'prompt_generator' in name:
            prompt_generator_params.append(param)

    if prompt_generator_params:
        param_groups.append({
            : prompt_generator_params,
            : lr * 0.5,
            : weight_decay * 0.1,
            : 'prompt_generator'
        })

    
    hypercond_params = []
    for name, param in model.named_parameters():
        if 'hyper_cond' in name or 'cond_channel_adapter' in name or 'cond_spatial_adapter' in name:
            hypercond_params.append(param)

    if hypercond_params:
        param_groups.append({
            : hypercond_params,
            : lr * 0.1,
            : weight_decay * 0.1,
            : 'hyper_cond'
        })

    
    rankdice_params = []
    for name, param in model.named_parameters():
        if 'rankdice_module' in name:
            rankdice_params.append(param)

    if rankdice_params:
        param_groups.append({
            : rankdice_params,
            : lr * 0.1,
            : weight_decay * 0.1,
            : 'rankdice'
        })

    
    
    already_assigned = set()
    for group in param_groups:
        for param in group['params']:
            already_assigned.add(id(param))

    other_params = []
    for name, param in model.named_parameters():
        if id(param) not in already_assigned:
            other_params.append(param)

    if other_params:
        param_groups.append({
            : other_params,
            : lr * 0.01,
            : weight_decay,
            : 'others'
        })

    
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    print(f"优化器参数组:")
    for group in param_groups:
        print(f"  {group['name']}: {len(group['params'])}个参数, lr={group['lr']}")

    return optimizer



def train_integrated_model(
        model,
        train_dataloader,
        val_dataloader,
        processor,
        vlm_model,
        tokenizer,
        mamba_model,
        optimizer,
        num_epochs,
        device,
        use_amp=False,
        model_save_path="./checkpoints",
        eval_interval=1,
):
    
    import os
    from tqdm import tqdm
    from datetime import datetime
    import torch.nn.functional as F

    os.makedirs(model_save_path, exist_ok=True)

    
    seg_loss = BoundaryAwareLoss(alpha=1.0, beta=0.1)

    
    from torch.distributions import Beta
    beta_dist = Beta(torch.tensor([2.0]), torch.tensor([2.0]))

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_dataloader),
        eta_min=1e-6
    )

    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    
    history = {
        : [],
        : [],
        : [],
        : [],
    }

    best_val_score = 0.0

    for epoch in range(num_epochs):
        model.train()
        model.set_training_mode(True)

        epoch_loss = 0.0
        epoch_rank_dice_loss = 0.0
        epoch_boundary_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for step, (image, gt2D, img_1024_ori) in enumerate(pbar):
            optimizer.zero_grad()

            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            
            tau_sample = beta_dist.sample().item()
            tau = 0.4 + tau_sample * 0.2  

            lambda_sample = beta_dist.sample().item()
            boundary_weight = 0.3 + lambda_sample * 0.7  

            hyper_cond = {
                : tau,
                : boundary_weight
            }

            
            with torch.no_grad():
                vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
                vlm_outputs = vlm_model.generate(**vlm_inputs)
                description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

                mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
                mamba_outputs = mamba_model(**mamba_inputs)
                vision_outputs = vlm_model.vision_model(**vlm_inputs)
                image_features_raw = vision_outputs.last_hidden_state[:, 1:, :]

            
            batch_size, seq_len, hidden_dim = image_features_raw.shape
            if seq_len == 576:  
                image_features = image_features_raw.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:  
                image_features = image_features_raw.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features_raw.mean(dim=1, keepdim=True)
                image_features = image_features.unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

            text_features = mamba_outputs.last_hidden_state

            
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, rank_dice_loss, _ = model(
                        image=image,
                        text_embeddings=text_features,
                        image_features=image_features,
                        gt_mask=gt2D,
                        hyper_cond=hyper_cond,
                        return_logits=True
                    )

                    
                    total_loss, dice_ce_loss, boundary_loss = seg_loss(
                        logits, gt2D, boundary_weight
                    )
                    total_loss = total_loss + rank_dice_loss

                
                scaler.scale(total_loss).backward()

                
                model.apply_gradient_constraints()

                scaler.step(optimizer)
                scaler.update()
            else:
                logits, rank_dice_loss, _ = model(
                    image=image,
                    text_embeddings=text_features,
                    image_features=image_features,
                    gt_mask=gt2D,
                    hyper_cond=hyper_cond,
                    return_logits=True
                )

                
                total_loss, dice_ce_loss, boundary_loss = seg_loss(
                    logits, gt2D, boundary_weight
                )
                total_loss = total_loss + rank_dice_loss

                
                total_loss.backward()

                
                model.apply_gradient_constraints()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            
            epoch_loss += total_loss.item()
            epoch_rank_dice_loss += rank_dice_loss.item() if isinstance(rank_dice_loss,
                                                                        torch.Tensor) else rank_dice_loss
            epoch_boundary_loss += boundary_loss.item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss

            
            scheduler.step()

            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                : f'{total_loss.item():.4f}',
                : f'{rank_dice_loss.item():.4f}' if isinstance(rank_dice_loss,
                                                                          torch.Tensor) else f'{rank_dice_loss:.4f}',
                : f'{current_lr:.2e}',
                : f'{tau:.3f}',
                : f'{boundary_weight:.3f}'
            })

        
        epoch_loss /= len(train_dataloader)
        epoch_rank_dice_loss /= len(train_dataloader)
        epoch_boundary_loss /= len(train_dataloader)

        history['train_loss'].append(epoch_loss)
        history['train_rank_dice_loss'].append(epoch_rank_dice_loss)
        history['train_boundary_loss'].append(epoch_boundary_loss)

        print(f"\nEpoch {epoch + 1} 训练结果:")
        print(f"  总损失: {epoch_loss:.4f}")
        print(f"  RankDice损失: {epoch_rank_dice_loss:.4f}")
        print(f"  边界损失: {epoch_boundary_loss:.4f}")

        
        if (epoch + 1) % eval_interval == 0 and val_dataloader is not None:
            val_metrics = evaluate_model(
                model,
                val_dataloader,
                processor,
                vlm_model,
                tokenizer,
                mamba_model,
                device
            )

            history['val_metrics'].append(val_metrics)

            print(f"  验证指标:")
            for key, value in val_metrics.items():
                print(f"    {key}: {value:.4f}")

            
            val_score = (val_metrics['sm'] + val_metrics['em'] + val_metrics['wfm']) / 3
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save({
                    : epoch,
                    : model.state_dict(),
                    : optimizer.state_dict(),
                    : scheduler.state_dict(),
                    : epoch_loss,
                    : val_score,
                    : val_metrics,
                    : history,
                }, os.path.join(model_save_path, 'model_best.pth'))
                print(f"  ✅ 保存最佳模型，验证分数: {val_score:.4f}")

        
        if (epoch + 1) % 5 == 0:
            torch.save({
                : epoch,
                : model.state_dict(),
                : optimizer.state_dict(),
                : scheduler.state_dict(),
                : epoch_loss,
                : history,
            }, os.path.join(model_save_path, f'model_epoch_{epoch + 1}.pth'))

    print("训练完成!")

    
    torch.save({
        : num_epochs,
        : model.state_dict(),
        : optimizer.state_dict(),
        : scheduler.state_dict(),
        : history,
    }, os.path.join(model_save_path, 'model_final.pth'))

    return history



def evaluate_model(model, dataloader, processor, vlm_model, tokenizer, mamba_model, device):
    
    model.eval()
    model.set_training_mode(False)

    
    
    try:
        from utils_downstream.saliency_metric import (
            cal_mae, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber
        )
    except ImportError:
        
        class MockMetric:
            def __init__(self):
                self.values = []

            def update(self, pred, gt):
                self.values.append(0.5)

            def show(self):
                return 0.5 if self.values else 0.0

        cal_mae = cal_sm = cal_em = cal_wfm = cal_dice = cal_iou = cal_ber = MockMetric

    
    mae, sm, em, wfm, m_dice, m_iou, ber = cal_mae(), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber()

    from tqdm import tqdm
    import torch.nn.functional as F

    pbar = tqdm(dataloader, desc='评估进度')

    with torch.no_grad():
        for step, (image, gt2D, img_1024_ori) in enumerate(pbar):
            image, gt2D = image.to(device), gt2D.to(device)
            img_1024_ori = img_1024_ori.to(device)

            
            vlm_inputs = processor(img_1024_ori, return_tensors="pt").to(device)
            vlm_outputs = vlm_model.generate(**vlm_inputs)
            description = processor.decode(vlm_outputs[0], skip_special_tokens=True)

            mamba_inputs = tokenizer(description, padding=True, return_tensors="pt").to(device)
            mamba_outputs = mamba_model(**mamba_inputs)
            vision_outputs = vlm_model.vision_model(**vlm_inputs)
            image_features_raw = vision_outputs.last_hidden_state[:, 1:, :]

            
            batch_size, seq_len, hidden_dim = image_features_raw.shape
            if seq_len == 576:
                image_features = image_features_raw.reshape(batch_size, 24, 24, hidden_dim)
                image_features = image_features.permute(0, 3, 1, 2)
                image_features = F.interpolate(image_features, size=(64, 64), mode='bilinear', align_corners=False)
            elif seq_len == 64 * 64:
                image_features = image_features_raw.reshape(batch_size, 64, 64, hidden_dim).permute(0, 3, 1, 2)
            else:
                image_features = image_features_raw.mean(dim=1, keepdim=True)
                image_features = image_features.unsqueeze(-1)
                image_features = F.interpolate(image_features, size=(64, 64))

            text_features = mamba_outputs.last_hidden_state

            
            pred_mask, _, _ = model(
                image=image,
                text_embeddings=text_features,
                image_features=image_features,
                gt_mask=None,
                hyper_cond={'threshold': 0.5, 'boundary_weight': 1.0},
                return_logits=False
            )

            
            pred_np = pred_mask.squeeze().cpu().numpy()
            gt_np = gt2D.squeeze().cpu().numpy()

            
            if pred_np.ndim == 0:
                pred_np = np.expand_dims(pred_np, 0)
            if gt_np.ndim == 0:
                gt_np = np.expand_dims(gt_np, 0)

            mae.update(pred_np, gt_np)
            sm.update(pred_np, gt_np)
            em.update(pred_np, gt_np)
            wfm.update(pred_np, gt_np)
            m_dice.update(pred_np, gt_np)
            m_iou.update(pred_np, gt_np)
            ber.update(pred_np, gt_np)

    
    metrics = {
        : sm.show(),
        : em.show(),
        : wfm.show(),
        : mae.show(),
        : m_dice.show(),
        : m_iou.show(),
        : ber.show(),
    }

    return metrics



def inference_integrated_model(
        model,
        image,
        text_features,
        image_features,
        device="cuda",
        threshold=0.5,
        boundary_weight=1.0,
        use_rankdice=True,
):
    
    model.eval()
    model.set_training_mode(False)

    hyper_cond = {
        : threshold,
        : boundary_weight
    }

    with torch.no_grad():
        pred_mask, _, _ = model(
            image=image.to(device),
            text_embeddings=text_features.to(device),
            image_features=image_features.to(device),
            gt_mask=None,
            hyper_cond=hyper_cond,
            return_logits=False
        )

    return pred_mask.cpu()



def main():
    
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Train integrated MMSAM model")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM pre-trained weights path")
    parser.add_argument("--model_type", type=str, default="vit_l", choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type")
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--val_sample_size", type=int, default=1000, help="Validation sample size")

    args = parser.parse_args()

    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    print("Creating integrated model...")
    model = create_integrated_model(
        sam_checkpoint_path=args.sam_checkpoint,
        model_type=args.model_type,
        image_size=1024,
        use_rankdice=True,
        use_hypercond=True,
        n_streams=4,
        device=device,
        blip_feature_dim=768
    )

    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    
    optimizer = create_optimizer_for_integrated_model(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    
    

    
    
    
    
    
    
    

    print("Model created successfully, ready for training!")


if __name__ == "__main__":
    main()