# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import numpy as np

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None, **kwargs):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None, **kwargs):
        images = images[:,:,:,32:352]
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


#----------------------------------------------------------------------------
# EDMLoss for Ambient Diffusion

@persistence.persistent_class
class AmbientLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, norm=2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.norm = norm

    # Centered, orthogonal fft in torch >= 1.7
    def fft(self, x):
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        return x

    # Centered, orthogonal ifft in torch >= 1.7
    def ifft(self, x):
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        return x
    
    def forward(self, image, maps, mask):
        coil_imgs = maps*image
        coil_ksp = self.fft(coil_imgs)
        sampled_ksp = mask*coil_ksp
        return sampled_ksp

    def adjoint(self, ksp, maps, mask):
        sampled_ksp = mask*ksp
        coil_imgs = self.ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(maps)*coil_imgs,dim=1)[:,None,...] #sum over coil dimension
        return img_out

    def __call__(self, net, images, corruption_matrix, hat_corruption_matrix, maps=None, labels=None, augment_pipe=None):        
        images = images[:,:,:,32:352]
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        y_noisy = y + n
        y_noisy_cplx = y_noisy[:,0] + 1j*y_noisy[:,1]
        y_noisy_cplx = y_noisy_cplx[:,None,...]
        
        noisy_image = self.adjoint(self.forward(y_noisy_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        noisy_image = torch.cat((noisy_image.real, noisy_image.imag), dim=1)
        
        hat_corruption_matrix_new = torch.ones_like(noisy_image).cuda()
        hat_corruption_matrix_new[:,0,:,:,] = hat_corruption_matrix[:,0]

        cat_input = torch.cat([noisy_image, hat_corruption_matrix_new], axis=1)
        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]

        D_yn_cplx = D_yn[:,0] + 1j*D_yn[:,1]
        D_yn_cplx = D_yn_cplx[:,None,...]
        masked_D_yn = self.adjoint(self.forward(D_yn_cplx, maps, corruption_matrix), maps, corruption_matrix)
        masked_D_yn = torch.cat((masked_D_yn.real, masked_D_yn.imag), dim=1)
        masked_D_yn_hat = self.adjoint(self.forward(D_yn_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        masked_D_yn_hat = torch.cat((masked_D_yn_hat.real, masked_D_yn_hat.imag), dim=1)

        y_cplx = y[:,0] + 1j*y[:,1]
        y_cplx = y_cplx[:,None,...]
        masked_y = self.adjoint(self.forward(y_cplx, maps, corruption_matrix), maps, corruption_matrix)
        masked_y = torch.cat((masked_y.real, masked_y.imag), dim=1)
        masked_y_hat = self.adjoint(self.forward(y_cplx, maps, hat_corruption_matrix), maps, hat_corruption_matrix)
        masked_y_hat = torch.cat((masked_y_hat.real, masked_y_hat.imag), dim=1)
        
        if self.norm == 2:
            train_loss = weight * ((masked_D_yn_hat - masked_y_hat) ** 2)
            val_loss = weight * ((masked_D_yn - masked_y) ** 2)
            test_loss = weight * ((D_yn - y) ** 2)
        elif self.norm == 1:
            # l1 loss
            train_loss = weight * (hat_corruption_matrix * torch.abs(D_yn - y))
            val_loss = weight * (corruption_matrix * torch.abs(D_yn - y))
            test_loss = weight * torch.abs(D_yn - y)
        else:
            # raise exception
            raise ValueError("Wrong norm type. Use 1 or 2.")
        return train_loss, val_loss, test_loss
#----------------------------------------------------------------------------
# VPLoss for Ambient Diffusion
@persistence.persistent_class
class AmbientVPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5, norm=2):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.norm = norm

    def __call__(self, net, images, corruption_matrix, hat_corruption_matrix, labels, augment_pipe=None, **kwargs):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        cat_input = torch.cat([hat_corruption_matrix * (y + n), hat_corruption_matrix], axis=1)
        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :y.shape[1]]

        if self.norm == 2:
            train_loss = weight * ((hat_corruption_matrix * (D_yn - y)) ** 2)
            val_loss = weight * ((corruption_matrix * (D_yn - y)) ** 2)
            test_loss = weight * ((D_yn - y) ** 2)
        elif self.norm == 1:
            # l1 loss
            train_loss = weight * (hat_corruption_matrix * torch.abs(D_yn - y))
            val_loss = weight * (corruption_matrix * torch.abs(D_yn - y))
            test_loss = weight * torch.abs(D_yn - y)
        else:
            # raise exception
            raise ValueError("Wrong norm type. Use 1 or 2.")
        return train_loss, val_loss, test_loss


    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()


#----------------------------------------------------------------------------
# Cross-view Ambient loss for multi-acquisition low-field MRI.
# One noisy repetition builds the (further-corrupted) denoiser input; a second,
# independent repetition supervises the prediction in measurement space.  No
# clean/high-field target is used.

@persistence.persistent_class
class CrossViewAmbientLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 cross_view_weight=1.0, input_heldout_weight=0.1, min_noise_variance=1e-8):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.cross_view_weight = cross_view_weight
        self.input_heldout_weight = input_heldout_weight
        self.min_noise_variance = min_noise_variance

    @staticmethod
    def _fftmod(x):
        x = x.clone()
        x[..., ::2, :] *= -1
        x[..., :, ::2] *= -1
        return x

    @staticmethod
    def _fft(x):
        return torch.fft.fft2(x, dim=(-2, -1), norm="ortho")

    @staticmethod
    def _ifft(x):
        return torch.fft.ifft2(x, dim=(-2, -1), norm="ortho")

    def _forward(self, x_cplx, maps, mask):
        return mask * self._fft(maps * x_cplx)

    def _adjoint(self, ksp, maps, mask):
        return torch.sum(torch.conj(maps) * self._ifft(mask * ksp), dim=1, keepdim=True)

    def _norm_residual(self, x_cplx, maps, mask, y, noise_std):
        """||M(F S x - y)||^2 / (sigma^2 N) per sample, N = C * nnz(mask)."""
        pred = self._forward(x_cplx, maps, mask)                       # [B,C,H,W]
        resid = pred - y
        sse = (resid.real ** 2 + resid.imag ** 2).sum(dim=(1, 2, 3))   # [B]
        n_coils = maps.shape[1]
        n_obs = (mask.abs() > 0).float().sum(dim=(1, 2, 3)) * n_coils   # [B]
        var = noise_std.reshape(-1) ** 2
        var = var.clamp_min(self.min_noise_variance)
        return sse / (var * n_obs.clamp_min(1.0))

    def __call__(self, net, batch, labels=None, augment_pipe=None):
        dev = batch["input_ksp"].device
        in_maps = self._fftmod(batch["input_maps"])                    # [B,C,H,W]
        in_ksp = self._fftmod(batch["input_ksp"])
        in_mask = batch["input_mask"]                                  # [B,1,H,W]
        in_delta = batch["input_delta_mask"]
        tgt_maps = self._fftmod(batch["target_maps"])
        tgt_ksp = self._fftmod(batch["target_ksp"])
        tgt_mask = batch["target_mask"]
        in_noise = batch["input_noise_std"].to(dev)
        tgt_noise = batch["target_noise_std"].to(dev)
        B = in_ksp.shape[0]

        # --- build the further-corrupted, diffusion-noised network input ---
        y_in_delta = in_delta * in_ksp
        in_adj = self._adjoint(y_in_delta, in_maps, in_delta)          # [B,1,H,W] cplx
        in_ch = torch.cat((in_adj.real, in_adj.imag), dim=1)           # [B,2,H,W]

        rnd = torch.randn([B, 1, 1, 1], device=dev)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        noisy = in_ch + sigma * torch.randn_like(in_ch)
        mask_ch = in_delta.repeat(1, 2, 1, 1)                          # [B,2,H,W]
        net_in = torch.cat([noisy, mask_ch], dim=1)                    # [B,4,H,W]
        pred = net(net_in, sigma, labels)[:, :2]                       # [B,2,H,W]
        x_theta = (pred[:, 0:1] + 1j * pred[:, 1:2])                   # [B,1,H,W] cplx

        # --- cross-view measurement loss on the INDEPENDENT target view ---
        y_tgt = tgt_mask * tgt_ksp
        cross = self._norm_residual(x_theta, tgt_maps, tgt_mask, y_tgt, tgt_noise)  # [B]

        # --- optional held-out input-consistency loss (input_mask minus delta) ---
        heldout_mask = (in_mask - in_delta).clamp(min=0.0)
        y_in_full = in_mask * in_ksp
        if (heldout_mask.abs() > 0).any():
            heldout = self._norm_residual(x_theta, in_maps, heldout_mask,
                                          heldout_mask * y_in_full, in_noise)       # [B]
        else:
            heldout = torch.zeros_like(cross)

        total = self.cross_view_weight * cross + self.input_heldout_weight * heldout
        return {
            "loss": total,
            "cross_view_loss": cross.detach(),
            "input_heldout_loss": heldout.detach(),
            "per_view_residual": cross.detach().sqrt(),
            "prediction_norm": x_theta.detach().abs().flatten(1).norm(dim=1),
            "sigma": sigma.detach().flatten(),
        }
