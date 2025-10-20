# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block

def get_2d_sincos_pos_embed(embed_dim, grid_sizeh, grid_sizew, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_sizeh, dtype=np.float32)
    grid_w = np.arange(grid_sizew, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_sizeh, grid_sizew])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_Rope(embed_dim, grid_sizeh, grid_sizew, cls_token=False):
    """
    Generate 2D positional embeddings with RoPE (Rotary Position Embedding).
    embed_dim: embedding dimension
    grid_sizeh: grid height
    grid_sizew: grid width
    cls_token: whether to add cls token
    return: [grid_sizeh*grid_sizew, embed_dim] or [1+grid_sizeh*grid_sizew, embed_dim]
    """
    assert embed_dim % 2 == 0

    # Generate position index grid
    pos_h = np.arange(grid_sizeh, dtype=np.float32)
    pos_w = np.arange(grid_sizew, dtype=np.float32)
    pos_h, pos_w = np.meshgrid(pos_h, pos_w, indexing='ij')

    # Flatten positions
    pos_h = pos_h.reshape(-1)  # (H*W,)
    pos_w = pos_w.reshape(-1)  # (H*W,)

    # Use half of the dims for h and half for w
    h_dim = w_dim = embed_dim // 2

    def get_rope_embed(pos, dim):
        """Generate RoPE encoding"""
        # Compute frequencies
        freqs = 10000 ** (-2 * np.arange(dim // 2) / dim)  # (D/4,)

        # Compute rotation angles
        theta = pos.reshape(-1, 1) * freqs  # (H*W, D/4)

        # Generate sin and cos
        emb_cos = np.cos(theta)  # (H*W, D/4)
        emb_sin = np.sin(theta)  # (H*W, D/4)

        # Interleave
        emb = np.zeros((pos.shape[0], dim))  # (H*W, D/2)
        emb[:, 0::2] = emb_cos
        emb[:, 1::2] = emb_sin
        return emb

    # Compute RoPE encoding for h and w separately
    emb_h = get_rope_embed(pos_h, h_dim)  # (H*W, D/2)
    emb_w = get_rope_embed(pos_w, w_dim)  # (H*W, D/2)

    # Concatenate both directions
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    # Optionally add cls token position embedding
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed





class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    


    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16, image_C=3, dropout=0.1,
                 args=None):
        super().__init__()
         # ------------------------
        #
        # --------------------------------------------------
        # MAE encoder specifics
        # img_size=224
        if hasattr(args, 'expandData'):
            expandData=args.expandData
        else:
            expandData=1
        self.img_sizeh=args.h
        self.predictionL=args.size[-1]*expandData
        self.inputL=args.size[0]*expandData
        self.patch_sizeh,self.patch_sizew=args.patch_size
        in_chans=image_C
        embed_dim=int(1024*args.modelSize)
        depth=int(12*args.modelSize)
        num_heads=int(16*args.modelSize)
        decoder_embed_dim=int(embed_dim/2)
        decoder_depth=int(depth/2)
        decoder_num_heads=num_heads
        mlp_ratio = 4.
        norm_layer = nn.LayerNorm
        norm_pix_loss = False
        self.patch_embed = PatchEmbed((self.img_sizeh,self.predictionL+self.inputL),args.patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_sizew*self.patch_sizeh * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        if hasattr(args,'ROPE'):
            self.ROPE=args.ROPE
        else:
            self.ROPE = False
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.ROPE:
            pos_embed = get_2d_sincos_pos_embed_Rope(self.pos_embed.shape[-1], int(self.img_sizeh / self.patch_sizeh),
                                                int((self.inputL + self.predictionL) / self.patch_sizew),
                                                cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed_Rope(self.decoder_pos_embed.shape[-1],
                                                        int(self.img_sizeh / self.patch_sizeh),
                                                        int((self.inputL + self.predictionL) / self.patch_sizew),
                                                        cls_token=True)
        else:

            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.img_sizeh/self.patch_sizeh),int((self.inputL+self.predictionL)/self.patch_sizew),
                                                cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                        int(self.img_sizeh/self.patch_sizeh),int((self.inputL+self.predictionL)/self.patch_sizew), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        c=imgs.shape[1]
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = int(self.img_sizeh / self.patch_sizeh)
        w = int((self.inputL + self.predictionL) / self.patch_sizew)
        x = imgs.reshape(shape=(imgs.shape[0], c, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph * pw * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]
        h = int(self.img_sizeh/self.patch_sizeh)
        w = int((self.inputL+self.predictionL)/self.patch_sizew)

        c=int(x.shape[2]/ph/pw)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        #xbs ,c ,w,h
        # embed patches
        # xbs ,c ,w,h ->
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio   x bs, w/patchsize*h/patchsize,embed_dim
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token x bs, w/patchsize*h/patchsize,embed_dim-> x bs, w/patchsize*h/patchsize*(1-mask_ratio),embed_dim
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens bs,w/patchsize*h/patchsize*(1-mask_ratio)+1,encoder_dim ->bs,w/patchsize*h/patchsize*(1-mask_ratio)+1,decoder_dim
        x = self.decoder_embed(x)

        # append mask tokens to sequence bs,w/patchsize*h/patchsize*(mask_ratio),decoder_dim
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token bs,w/patchsize*h/patchsize,decoder_dim
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token bs,w/patchsize*h/patchsize+1,decoder_dim

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection bs,w/patchsize*h/patchsize+1,decoder_dim-> bs,w/patchsize*h/patchsize+1,patchsize*patchsize*c
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :] #bs,w/patchsize*h/patchsize,patchsize*patchsize*c

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0):
        imgs = torch.einsum('bcwh->bchw', imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        reconstructImg=self.unpatchify(pred)
        reconstructImg = torch.einsum('bchw->bcwh', reconstructImg)
        return reconstructImg







class MaskedAutoencoderViT2(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    


    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16, image_C=3, dropout=0.1,
                 args=None):
        super().__init__()
         # ------------------------
        #
        # --------------------------------------------------
        # MAE encoder specifics
        # img_size=224
        self.img_sizeh=args.h
        self.predictionL=args.size[-1]
        self.inputL=args.size[0]
        self.patch_sizeh,self.patch_sizew=(1,1)
        in_chans=image_C
        embed_dim=self.img_sizeh
        depth=int(12*args.modelSize)
        num_heads=int(16*args.modelSize)
        decoder_embed_dim=int(embed_dim/2)
        decoder_depth=int(depth/2)
        decoder_num_heads=num_heads
        mlp_ratio = 4.
        norm_layer = nn.LayerNorm
        norm_pix_loss = False
        self.patch_embed = PatchEmbed((self.img_sizeh,self.predictionL+self.inputL),args.patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_sizew*self.patch_sizeh * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int((self.inputL+self.predictionL)/self.patch_sizew),1,
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        c=imgs.shape[1]
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = int(self.img_sizeh / self.patch_sizeh)
        w = int((self.inputL + self.predictionL) / self.patch_sizew)
        x = imgs.reshape(shape=(imgs.shape[0], c, h, ph, w, pw))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, ph * pw * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        ph = self.patch_embed.patch_size[0]
        pw = self.patch_embed.patch_size[1]
        h = int(self.img_sizeh/self.patch_sizeh)
        w = int((self.inputL+self.predictionL)/self.patch_sizew)

        c=int(x.shape[2]/ph/pw)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * ph, w * pw))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # xbs ,c ,w,h
        x = x + self.pos_embed[:, 1:, :]
        

        # append cls token x bs, w/patchsize*h/patchsize,embed_dim-> x bs, w/patchsize*h/patchsize*(1-mask_ratio),embed_dim
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, ids_restore):
        # embed tokens bs,w/patchsize*h/patchsize*(1-mask_ratio)+1,encoder_dim ->bs,w/patchsize*h/patchsize*(1-mask_ratio)+1,decoder_dim
        x = self.decoder_embed(x)

        # append mask tokens to sequence bs,w/patchsize*h/patchsize*(mask_ratio),decoder_dim
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token bs,w/patchsize*h/patchsize,decoder_dim
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token bs,w/patchsize*h/patchsize+1,decoder_dim

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection bs,w/patchsize*h/patchsize+1,decoder_dim-> bs,w/patchsize*h/patchsize+1,patchsize*patchsize*c
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :] #bs,w/patchsize*h/patchsize,patchsize*patchsize*c

        return x


    def forward(self, imgs, mask_ratio=0):
        imgs=imgs.squeeze(1)
        reconstructImg = self.forward_encoder(imgs, mask_ratio)
        reconstructImg=reconstructImg[:, 1:, :]
        reconstructImg=reconstructImg.unsqueeze(1)
        return reconstructImg














 
import torch


def test_masked_autoencoder_vit():
    # Parameters for the model and input
    batch_size = 4
    img_sizeh = 128
    inputL=336
    predictionL=720
    img_sizew = inputL+predictionL
    channels = 1  # RGB images
    patch_size=(4,32)
    # Create a model instance
    model = MaskedAutoencoderViT(img_sizeh=img_sizeh,predictionL=predictionL,inputL=inputL, patch_size=patch_size, in_chans=channels,
                                 embed_dim=1024, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4., norm_layer=torch.nn.LayerNorm)

    # Generate a batch of random images (batch_size, channels, height, width)
    random_images = torch.randn(batch_size, channels, img_sizeh, img_sizew)

    # Forward pass through the model
    loss, pred, mask ,reconstructImg= model(random_images)

    # Print the loss and output shape
    print(f"Loss: {loss.item()}")
    print(f"Predicted output shape: {reconstructImg.shape}")
    print(f"Mask shape: {mask.shape}")


# Run the test function
if __name__ == "__main__":
    test_masked_autoencoder_vit()
