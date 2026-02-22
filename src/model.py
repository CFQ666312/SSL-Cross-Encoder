# models.py
import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ViT blocks from timm
from timm.models.vision_transformer import Block


# =========================================================
# 1) CNN building blocks
# =========================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act="leaky_relu", dropout=0.0, num_conv=1):
        super().__init__()
        if conv_act == "relu":
            act = nn.ReLU(inplace=True)
        elif conv_act == "leaky_relu":
            act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"No implementation of {conv_act}")

        layers = [
            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(out_num_ch),
            act,
            nn.Dropout3d(dropout),
        ]
        if num_conv == 2:
            layers += [
                nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                nn.BatchNorm3d(out_num_ch),
                act,
                nn.Dropout3d(dropout),
            ]
        elif num_conv != 1:
            raise ValueError("Number of conv can only be 1 or 2")

        layers.append(nn.MaxPool3d(2))
        self.conv = nn.Sequential(*layers)
        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act="leaky_relu", dropout=0.0, num_conv=1):
        super().__init__()
        if conv_act == "relu":
            act = nn.ReLU(inplace=True)
        elif conv_act == "leaky_relu":
            act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"No implementation of {conv_act}")

        layers = [
            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(out_num_ch),
            act,
            nn.Dropout3d(dropout),
        ]
        if num_conv == 2:
            layers += [
                nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                nn.BatchNorm3d(out_num_ch),
                act,
                nn.Dropout3d(dropout),
            ]
        elif num_conv != 1:
            raise ValueError("Number of conv can only be 1 or 2")

        layers.append(nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True))
        self.conv = nn.Sequential(*layers)
        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.conv(x)


class CNNEncoder3D(nn.Module):
    """
    Default input: (B,1,64,64,64)
    Output: (B,16,4,4,4) -> flatten 1024
    """
    def __init__(self, in_num_ch=1, inter_num_ch=16, kernel_size=3, conv_act="leaky_relu", num_conv=1, dropout=False):
        super().__init__()
        if dropout:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2 * inter_num_ch, kernel_size, conv_act, dropout=0.1, num_conv=num_conv)
            self.conv3 = EncoderBlock(2 * inter_num_ch, 4 * inter_num_ch, kernel_size, conv_act, dropout=0.2, num_conv=num_conv)
            self.conv4 = EncoderBlock(4 * inter_num_ch, inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
        else:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2 * inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
            self.conv3 = EncoderBlock(2 * inter_num_ch, 4 * inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
            self.conv4 = EncoderBlock(4 * inter_num_ch, inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class CNNDecoder3D(nn.Module):
    """
    Input latent: (B,1024) -> reshape (B,16,4,4,4) -> reconstruct (B,1,64,64,64)
    """
    def __init__(self, out_num_ch=1, inter_num_ch=16, kernel_size=3, conv_act="leaky_relu", num_conv=1):
        super().__init__()
        self.inter_num_ch = inter_num_ch
        self.conv4 = DecoderBlock(inter_num_ch, 4 * inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
        self.conv3 = DecoderBlock(4 * inter_num_ch, 2 * inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
        self.conv2 = DecoderBlock(2 * inter_num_ch, inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
        self.conv1 = DecoderBlock(inter_num_ch, inter_num_ch, kernel_size, conv_act, dropout=0.0, num_conv=num_conv)
        self.conv0 = nn.Conv3d(inter_num_ch, out_num_ch, kernel_size=3, padding=1)

    def forward(self, z_flat):
        x = z_flat.view(z_flat.shape[0], self.inter_num_ch, 4, 4, 4)
        x = self.conv4(x)
        x = self.conv3(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return self.conv0(x)


# =========================================================
# 2) ViT (3D patch embedding + encoder + MAE-style decoder)
# =========================================================
class PatchEmbed3D(nn.Module):
    """3D patch embedding for ViT"""
    def __init__(self, img_size=(64, 64, 64), patch_size=(8, 8, 8), in_chans=1, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)      # [B, num_patches, embed_dim]
        return x


class ViTEncoder(nn.Module):
    """ViT Encoder with PatchEmbed3D"""
    def __init__(self, img_size=(64, 64, 64), patch_size=(8, 8, 8), in_chans=1, embed_dim=1024, depth=12, num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # include cls token
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                       # [B, N, C]
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, C]
        x = torch.cat([cls_tokens, x], dim=1)         # [B, N+1, C]
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x                                      # [B, N+1, C]


class MAEDecoder(nn.Module):
    """
    Simple MAE-style decoder to reconstruct volume from tokens.
    NOTE: This assumes N is a perfect cube (for 64^3 with 8^3 patch, N=512, cube root=8).
    """
    def __init__(self, embed_dim=1024, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, patch_size=(8, 8, 8), out_ch=1):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, out_ch * patch_size[0] * patch_size[1] * patch_size[2])

    def forward(self, x):
        x = self.decoder_embed(x)     # [B, N+1, C']
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x[:, 1:, :])  # remove cls token -> [B, N, patch_vol]
        B, N, C = x.shape

        d = int(round(N ** (1 / 3)))        # for N=512 -> d=8
        pz, py, px = self.patch_size

        # reshape into (B, out_ch, D, H, W)
        x = x.view(B, N, pz, py, px)        # [B, N, pz, py, px]
        x = x.view(B, d, d, d, pz, py, px)  # [B, dz, dy, dx, pz, py, px]
        x = x.permute(0, 4, 1, 5, 2, 6, 3).contiguous()  # [B, pz, dz, py, dy, px, dx]
        x = x.view(B, 1, d * pz, d * py, d * px)         # [B,1,64,64,64] if patch 8 and d 8
        return x


# =========================================================
# 3) Classifier (shared)
# =========================================================
class Classifier(nn.Module):
    def __init__(self, latent_size=1024, inter_num_ch=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(latent_size),
            nn.Dropout(0.5),
            nn.Linear(latent_size, inter_num_ch),
            nn.LeakyReLU(0.2),
            nn.Linear(inter_num_ch, 1),
        )
        self._init()

    def _init(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        # batch=1 -> bypass BN/Dropout
        if x.size(0) == 1:
            return self.fc[2:](x)
        return self.fc(x)


# =========================================================
# 4) Cross-Sim (CNN)
# =========================================================
class Cross_Sim(nn.Module):
    """
    CNN Cross-Sim (your original logic):
      - encode img1,img2 -> z1,z2 (flatten)
      - split into s/d by ratio selection
      - swap s between paired scans
      - decode recon
      - optional input-gradient reg computed outside via compute_img_gradients
    """
    def __init__(self, selection=0.75, temperature=0.5, dropout=False):
        super().__init__()
        self.encoder = CNNEncoder3D(in_num_ch=1, inter_num_ch=16, num_conv=1, dropout=dropout)
        self.decoder = CNNDecoder3D(out_num_ch=1, inter_num_ch=16, num_conv=1)
        self.selection = selection
        self.temperature = temperature

    def forward(self, img1, img2):
        bs = img1.shape[0]
        img = torch.cat([img1, img2], 0).requires_grad_(True)

        zs = self.encoder(img)                  # (2B,16,4,4,4)
        zs_flatten = zs.view(bs * 2, -1)        # (2B,1024)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]

        split_size = int(z1.size(1) * self.selection)
        s1, s2 = z1[:, :split_size], z2[:, :split_size]
        d1, d2 = z1[:, split_size:].clone(), z2[:, split_size:].clone()

        z1_swapped = torch.cat([s2, d1], dim=1)
        z2_swapped = torch.cat([s1, d2], dim=1)

        recon1 = self.decoder(z1_swapped)
        recon2 = self.decoder(z2_swapped)

        return [s1, s2], [d1, d2], [recon1, recon2], img

    def compute_img_gradients(self, img, d1, d2, p=1):
        """
        Input-gradient regularization:
          - build a simple loss on d (mean squared)
          - compute grad wrt img
          - return L1 (p=1) or L2 (p=2) norm
        """
        zd = torch.cat([d1, d2], dim=0)
        loss = (zd ** 2).mean()
        grads = torch.autograd.grad(loss, img, retain_graph=True, create_graph=False)[0]
        if p == 1:
            return grads.abs().sum()
        elif p == 2:
            return (grads ** 2).sum()
        else:
            raise ValueError("p must be 1 or 2")

    @staticmethod
    def compute_recon_loss(x, recon):
        return ((x - recon) ** 2).mean()

    @staticmethod
    def compute_residual_loss(x1, recon1, x2, recon2):
        return torch.abs((x1 - recon1) - (x2 - recon2)).mean()

    def Contrastiveloss(self, proj_z1, proj_z2, batch_size):
        z_i = F.normalize(proj_z1, dim=1)
        z_j = F.normalize(proj_z2, dim=1)
        reps = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)

        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=reps.device)).float() * torch.exp(sim / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        return loss_partial.mean()


# =========================================================
# 5) Cross-Sim (ViT)
# =========================================================
class Cross_Sim_ViT(nn.Module):
    """
    Your ViT Cross-Sim:
      - ViTEncoder on 3D patches -> tokens
      - selector chooses dynamic tokens (top-k with straight-through)
      - swap dynamic part across timepoints (token-wise)
      - MAE decoder reconstructs volume
      - optional contrastive on pooled dynamic tokens
    """
    def __init__(self, img_size=(64, 64, 64), patch_size=(8, 8, 8), embed_dim=1024, depth=12, num_heads=16,
                 dynamic_ratio=0.25, temperature=0.5):
        super().__init__()
        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.decoder = MAEDecoder(embed_dim=embed_dim, patch_size=patch_size, out_ch=1)
        self.selection = dynamic_ratio     # dynamic token ratio
        self.temperature = temperature

        # learned selector per token
        self.selector = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, img1, img2):
        bs = img1.shape[0]
        img = torch.cat([img1, img2], dim=0).requires_grad_(True)
        zs = self.encoder(img)                     # [2B, N+1, C]

        zs_tokens = zs[:, 1:, :]                   # remove cls -> [2B, N, C]
        z1, z2 = zs_tokens[:bs], zs_tokens[bs:]    # [B, N, C]

        B, N, C = z1.shape
        k = max(1, int(self.selection * N))

        scores1 = self.selector(z1).squeeze(-1)    # [B, N]
        scores2 = self.selector(z2).squeeze(-1)

        topk_idx1 = scores1.topk(k, dim=1).indices
        topk_idx2 = scores2.topk(k, dim=1).indices

        hard1 = torch.zeros_like(scores1)
        hard2 = torch.zeros_like(scores2)
        hard1.scatter_(1, topk_idx1, 1.0)
        hard2.scatter_(1, topk_idx2, 1.0)

        # straight-through
        soft1 = torch.sigmoid(scores1 / 0.1)
        soft2 = torch.sigmoid(scores2 / 0.1)
        mask1 = hard1 + (soft1 - soft1.detach())
        mask2 = hard2 + (soft2 - soft2.detach())

        d1 = z1 * mask1.unsqueeze(-1)
        s1 = z1 * (1.0 - mask1).unsqueeze(-1)
        d2 = z2 * mask2.unsqueeze(-1)
        s2 = z2 * (1.0 - mask2).unsqueeze(-1)

        # swap dynamic token contribution
        z1_swapped_tokens = s1 + d2
        z2_swapped_tokens = s2 + d1

        # add cls back (use original cls from zs)
        cls = zs[:, :1, :]  # [2B,1,C]
        z1_swapped = torch.cat([cls[:bs], z1_swapped_tokens], dim=1)
        z2_swapped = torch.cat([cls[bs:], z2_swapped_tokens], dim=1)

        recon1 = self.decoder(z1_swapped)
        recon2 = self.decoder(z2_swapped)

        return [s1, s2], [d1, d2], [recon1, recon2], img

    @staticmethod
    def pool_dynamic(d):
        return d.mean(dim=1)  # [B, C]

    def compute_img_gradients(self, img, d1, d2, p=1):
        """
        Gradient reg on pooled dynamic tokens (as in your code).
        """
        zd = torch.cat([d1, d2], dim=0)          # [2B, N, C]
        zd_pool = zd.mean(dim=1)                # [2B, C]
        loss = (zd_pool ** 2).mean()
        grads = torch.autograd.grad(loss, img, retain_graph=True, create_graph=False)[0]
        if p == 1:
            return grads.abs().sum()
        elif p == 2:
            return (grads ** 2).sum()
        else:
            raise ValueError("p must be 1 or 2")

    @staticmethod
    def compute_recon_loss(x, recon):
        return ((x - recon) ** 2).mean()

    @staticmethod
    def compute_residual_loss(x1, recon1, x2, recon2):
        return torch.abs((x1 - recon1) - (x2 - recon2)).mean()

    def Contrastiveloss(self, d1, d2, batch_size):
        proj_z1 = self.pool_dynamic(d1)
        proj_z2 = self.pool_dynamic(d2)
        z_i = F.normalize(proj_z1, dim=1)
        z_j = F.normalize(proj_z2, dim=1)
        reps = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)

        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=reps.device)).float() * torch.exp(sim / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        return loss_partial.mean()


# =========================================================
# 6) AE baseline (CNN)
# =========================================================
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder3D(in_num_ch=1, inter_num_ch=16, num_conv=1)
        self.decoder = CNNDecoder3D(out_num_ch=1, inter_num_ch=16, num_conv=1)

    def forward(self, img1, img2):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        zs_flat = zs.view(bs * 2, -1)
        z1, z2 = zs_flat[:bs], zs_flat[bs:]
        recon = self.decoder(zs_flat)
        recon1, recon2 = recon[:bs], recon[bs:]
        return [z1, z2], [recon1, recon2]

    @staticmethod
    def compute_recon_loss(x, recon):
        return ((x - recon) ** 2).mean()


# =========================================================
# 7) LSP baseline (CNN) - minimal kept
# =========================================================
class LSP(nn.Module):
    def __init__(self, latent_size=1024, num_neighbours=3, agg_method="gaussian", N_km=[120, 60, 30], device=None):
        super().__init__()
        self.encoder = CNNEncoder3D(in_num_ch=1, inter_num_ch=16, num_conv=1)
        self.decoder = CNNDecoder3D(out_num_ch=1, inter_num_ch=16, num_conv=1)

        self.mapping = nn.Linear(1024, latent_size) if latent_size < 1024 else nn.Identity()
        self.num_neighbours = num_neighbours
        self.agg_method = agg_method
        self.N_km = N_km
        self.device = device

    def forward(self, img1, img2):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        zs_flat = zs.view(bs * 2, -1)
        z_mapped = self.mapping(zs_flat)

        z1, z2 = z_mapped[:bs], z_mapped[bs:]
        recon = self.decoder(zs_flat)
        recon1, recon2 = recon[:bs], recon[bs:]
        return [z1, z2], [recon1, recon2]

    def build_graph_batch(self, z1):
        bs = z1.shape[0]
        dis_mx = torch.zeros(bs, bs, device=z1.device)
        for i in range(bs):
            for j in range(i + 1, bs):
                dis = torch.sum((z1[i] - z1[j]) ** 2)
                dis_mx[i, j] = dis
                dis_mx[j, i] = dis

        if self.agg_method == "gaussian":
            adj_mx = torch.exp(-dis_mx / 100.0)
        else:
            raise ValueError("agg_method not supported")

        if self.num_neighbours < bs:
            adj_f = torch.zeros(bs, bs, device=z1.device)
            for i in range(bs):
                ks = torch.argsort(dis_mx[i], descending=False)[: self.num_neighbours + 1]
                adj_f[i, ks] = adj_mx[i, ks]
                adj_f[i, i] = 0.0
            return adj_f
        else:
            return adj_mx * (1.0 - torch.eye(bs, device=z1.device))

    @staticmethod
    def compute_recon_loss(x, recon):
        return ((x - recon) ** 2).mean()

    @staticmethod
    def compute_direction_loss(delta_z, delta_h):
        dz_norm = torch.norm(delta_z, dim=1) + 1e-12
        dh_norm = torch.norm(delta_h, dim=1) + 1e-12
        cos = torch.sum(delta_z * delta_h, 1) / (dz_norm * dh_norm)
        return (1.0 - cos).mean()

    @staticmethod
    def compute_distance_loss(delta_z, delta_h):
        dz_norm = torch.norm(delta_z, dim=1) + 1e-12
        dis = torch.norm(delta_z - delta_h, dim=1)
        return (dis / dz_norm).mean()


# =========================================================
# 8) CLS models (CNN + ViT)
# =========================================================
class CLS(nn.Module):
    """
    CNN classifier baseline (your previous CLS-style):
      - encode img1,img2 (shared CNN)
      - use z1 or delta_z (here default: z1)
    """
    def __init__(self, dropout=False):
        super().__init__()
        self.encoder = CNNEncoder3D(in_num_ch=1, inter_num_ch=16, num_conv=1, dropout=dropout)
        self.classifier = Classifier(latent_size=1024, inter_num_ch=64)

    def forward(self, img1, img2, interval=None):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        zs_flat = zs.view(bs * 2, -1)
        z1, z2 = zs_flat[:bs], zs_flat[bs:]
        # delta_z = (z2 - z1) / interval.unsqueeze(1) if interval is not None else (z2 - z1)
        pred = self.classifier(z1)
        return pred

    def forward_single(self, img):
        z = self.encoder(img).view(img.shape[0], -1)
        pred = self.classifier(z)
        return pred, z

    @staticmethod
    def compute_classification_loss(pred, label, pos_weight=1.0):
        # label assumed already mapped to {0,1}
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=pred.device, dtype=torch.float))(pred.squeeze(1), label.float())
        return loss, torch.sigmoid(pred)


class CLS_Cross_ViT(nn.Module):
    """
    ViT classifier with selector (as you posted):
      - ViTEncoder -> tokens
      - selector top-k tokens as dynamic -> mean pool -> MLP classifier
    """
    def __init__(self, img_size=(64, 64, 64), patch_size=(8, 8, 8), embed_dim=1024, depth=12, num_heads=16, dynamic_ratio=0.25):
        super().__init__()
        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.selection = dynamic_ratio
        self.classifier = Classifier(latent_size=embed_dim, inter_num_ch=64)
        self.selector = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, img1, img2, interval=None):
        bs = img1.shape[0]
        img = torch.cat([img1, img2], dim=0)
        zs = self.encoder(img)               # [2B, N+1, C]
        tokens = zs[:, 1:, :]                # [2B, N, C]
        z1, z2 = tokens[:bs], tokens[bs:]    # [B, N, C]
        B, N, C = z1.shape
        k = max(1, int(self.selection * N))

        scores1 = self.selector(z1).squeeze(-1)
        scores2 = self.selector(z2).squeeze(-1)

        topk_idx1 = scores1.topk(k, dim=1).indices
        topk_idx2 = scores2.topk(k, dim=1).indices

        mask1 = torch.zeros_like(scores1)
        mask2 = torch.zeros_like(scores2)
        mask1.scatter_(1, topk_idx1, 1.0)
        mask2.scatter_(1, topk_idx2, 1.0)

        soft1 = torch.sigmoid(scores1 / 0.1)
        soft2 = torch.sigmoid(scores2 / 0.1)
        mask1_st = mask1 + (soft1 - soft1.detach())
        mask2_st = mask2 + (soft2 - soft2.detach())

        d1 = z1 * mask1_st.unsqueeze(-1)
        d_pool = d1.mean(dim=1)              # [B, C]
        pred = self.classifier(d_pool)
        return pred

    @staticmethod
    def compute_classification_loss(pred, label, pos_weight=1.0):
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=pred.device, dtype=torch.float))(pred.squeeze(1), label.float())
        return loss, torch.sigmoid(pred)


__all__ = [
    # CNN blocks
    "EncoderBlock",
    "DecoderBlock",
    "CNNEncoder3D",
    "CNNDecoder3D",
    # ViT blocks
    "PatchEmbed3D",
    "ViTEncoder",
    "MAEDecoder",
    # Models
    "Cross_Sim",
    "Cross_Sim_ViT",
    "AE",
    "LSP",
    "CLS",
    "CLS_Cross_ViT",
    "Classifier",
]
