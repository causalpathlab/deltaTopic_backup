# -*- coding: utf-8 -*-
"""Main module."""
from typing import List, Optional, Tuple, Optional
import numpy as np
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import _CONSTANTS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data

from nn.base_components import MultiMaskedEncoder, DeltaETMDecoder

torch.backends.cudnn.benchmark = True

def etm_llik(xx, pr, eps=1e-8):
    return torch.sum(xx * torch.log(pr+eps),dim=1)

class DeltaETM_module(BaseModuleClass):
    """
    scCLR 

    Parameters
    ----------
    dim_input_list
        List of number of input genes for each dataset. If
            the datasets have different sizes, the dataloader will loop on the
            smallest until it reaches the size of the longest one
    total_genes
        Total number of different genes
    n_latent
        dimension of latent space
    n_layers_encoder_individual
        number of individual layers in the encoder
    n_layers_encoder_shared
        number of shared layers in the encoder
    dim_hidden_encoder
        dimension of the hidden layers in the encoder
    n_layers_decoder_individual
        number of layers that are conditionally batchnormed in the encoder
    n_layers_decoder_shared
        number of shared layers in the decoder
    dim_hidden_decoder_individual
        dimension of the individual hidden layers in the decoder
    dim_hidden_decoder_shared
        dimension of the shared hidden layers in the decoder
    dropout_rate_encoder
        dropout encoder
    dropout_rate_decoder
        dropout decoder
    n_batch
        total number of batches
    n_labels
        total number of labels
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        mask: torch.Tensor = None, 
        n_latent: int = 10,
        n_layers_encoder_individual: int = 1,
        n_layers_encoder_shared: int = 2,
        dim_hidden_encoder: int = 32,
        n_layers_decoder: int = 1, # by default, the decoder has no hidden layers
        dim_hidden_decoder: int = 32, # not in effect when n_layers_decoder = 1
        dropout_rate_encoder: float = 0.1,
        dropout_rate_decoder: float = 0.1,
        n_batch: int = 0,
        n_labels: int = 0,
        log_variational: bool = True,
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.mask = mask
        self.n_latent = n_latent

        self.n_batch = n_batch
        self.n_labels = n_labels

        self.log_variational = log_variational

        self.z_encoder = MultiMaskedEncoder(
            n_heads=len(dim_input_list),
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            mask = self.mask,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
        )

        # TODO: use self.total_genes is dangerous, if we have dfferent sets of genes in spliced and un unspliced
        self.decoder = DeltaETMDecoder(self.n_latent , self.total_genes)


    def sample_from_posterior_z(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample tensor of latent values from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``

        """
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode when having multiple datasets")
        outputs = self.inference(x, mode)

        qz_m = outputs["qz_m"]
        z = outputs["z"]
        if deterministic:
            z = qz_m
        #if output_z_raw:
        #    z = z_raw
        return dict(z=z)
    
    def get_latent_parameter_z_shared(
        self, x: torch.Tensor, mode: int = None) -> torch.Tensor:
        """
        Sample tensor of latent values from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        mode
            head id to use in the encoder

        Returns
        -------
        type
            qz_m qz_v
            dictionary of tensors of shape ``(batch_size, n_latent)``
    
        """
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode when having multiple datasets")
        outputs = self.inference(x, mode)
        qz_m = outputs["qz_m"] 
        qz_v = outputs["qz_v"]
        return dict(qz_m = qz_m, qz_v = qz_v)
    
    def reconstruction_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:

        reconstruction_loss = None
        reconstruction_loss = -etm_llik(x,recon)
        
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        #batch_index = tensors[_CONSTANTS.BATCH_KEY]
        #y = tensors[_CONSTANTS.LABELS_KEY]
        return dict(z=z)

    @auto_move_data
    def inference(self, x: torch.Tensor, mode: Optional[int] = None) -> dict:
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_, mode)
        
        return dict(qz_m=qz_m, qz_v=qz_v, z=z)


    @auto_move_data
    def generative(self,z: torch.Tensor, mode: int) -> dict:

        recon, hh, log_softmax_rho, log_softmax_delta  = self.decoder(z, mode)

        return dict(recon=recon, hh=hh, log_softmax_rho = log_softmax_rho, log_softmax_delta = log_softmax_delta)
    # this is for the purpose of computing the integrated gradient 
    # output source specifc or shared LV based no the interests 
    def get_latent_representation(
        self, 
        tensors: torch.Tensor,
        mode: int,
        deterministic: bool = False,
        output_z_ind: bool = True,  
    ):
        inference_out = self.inference(tensors, mode)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        
        return z

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        mode: int,
        deterministic: bool = False,
        decode_mode: int = None,
    ) -> torch.Tensor:
        """
        Returns the tensor of scaled frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        mode
            int encode mode (which input head to use in the model)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        deterministic
            bool - whether to sample or not
        decode_mode
            int use to a decode mode different from encoding mode

        Returns
        -------
        type
            tensor of means of the scaled frequencies

        """
        if decode_mode is None:
            decode_mode = mode
        inference_out = self.inference(x, mode)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
    
        gen_out = self.generative(z, decode_mode)
        
        recon = gen_out['recon']
       

        reconstruction_loss = self.reconstruction_loss(x, recon)
        return reconstruction_loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        mode: Optional[int] = None,
        kl_weight=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Return the reconstruction loss and the Kullback divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        local_l_mean
            tensor of means of the prior distribution of latent variable l
            with shape (batch_size, 1)
        local_l_var
            tensor of variances of the prior distribution of latent variable l
            with shape (batch_size, 1)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape (batch_size, n_labels)
        mode
            indicates which head/tail to use in the joint network


        Returns
        -------
        the reconstruction loss and the Kullback divergences

        """
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode")
        x = tensors[_CONSTANTS.X_KEY]
       
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        
        reconstruction_loss = self.get_reconstruction_loss(x, mode)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        kl_local = kl_divergence_z
 
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossRecorder(loss, reconstruction_loss, kl_local)

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        w = self.decoder.pathway_decoders[0].fc_layers[0][0].weight
        bn = self.decoder.pathway_decoders[0].fc_layers[0][1]
        sigma = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        b = gamma / sigma
        b_identity = torch.diag(b)
        loadings = torch.matmul(b_identity, w)
        loadings_domain1 = loadings.detach().cpu().numpy()   
 
        w = self.decoder.pathway_decoders[1].fc_layers[0][0].weight
        bn = self.decoder.pathway_decoders[1].fc_layers[0][1]
        sigma = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        b = gamma / sigma
        b_identity = torch.diag(b)
        loadings = torch.matmul(b_identity, w)
        loadings_domain2 = loadings.detach().cpu().numpy()   

        w = self.decoder.pathway_decoder_shared.fc_layers[0][0].weight
        bn = self.decoder.pathway_decoder_shared.fc_layers[0][1]
        sigma = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        b = gamma / sigma
        b_identity = torch.diag(b)
        loadings = torch.matmul(b_identity, w)
        loadings_domain_shared = loadings.detach().cpu().numpy() 
        
        return loadings_domain1, loadings_domain2, loadings_domain_shared



'''
import torch
from module.DeltaETM_module import DeltaETM_module
test_module = DeltaETM_module([20,20],20)
x = torch.rand(2, 20)
inference_out = test_module.inference(x, mode = 0)
z = inference_out["z"]
z.shape
generative_out = test_module.generative(z, mode = 0)
pr = generative_out["recon"]

def etm_llik(xx,pr, eps=1e-8):
    return torch.sum(xx * torch.log(pr+eps),dim=1)

reconstruction_loss = etm_llik(x,pr)
'''