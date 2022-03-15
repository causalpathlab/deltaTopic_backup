# -*- coding: utf-8 -*-
"""Main module."""
from typing import List, Optional, Tuple, Union, Callable, Iterable, Optional
from scvi._compat import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from torch.nn import ModuleList

from scvi import _CONSTANTS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, MultiDecoder, one_hot
from scvi.module import VAE

from nn.base_components import MultiLatentEncoder, MaskedLinearLayers, MaskedResEncoder
from nn.base_components import MaskedLatentEncoder, Decoder_residual, MultiLatentDecoder

torch.backends.cudnn.benchmark = True

class scCLR_phase2_module(VAE):
    """
    The "second-phase" model for learning domain specific LVs with shared LVs (fixed)
    as "residual" in the skip connetion layers. 
    TODO: edit decscriptions
    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer (for encoder)
    n_latent
        Dimensionality of the latent space
    n_layers_encoder
        Number of hidden layers used for encoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    use_batch_norm
        Bool whether to use batch norm in decoder
    bias
        Bool whether to have bias term in linear decoder
    """

    def __init__(
        self,
        n_input: int,
        mask: torch.Tensor, 
        n_latent: int,
        dim_s: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden_l_encoder: int = 128,
        n_layers_l_decoder: int = 1,
        n_layers_masked: int = 1,
        n_hidden_masked: int = 128,
        n_out_masked: int = 128,
        n_layers_FC: int = 1,
        n_layers_decoder: int = 1,
        n_hidden_FC: int = 128,
        n_hidden_decoder: int = 128,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        encode_covariates: bool = False,
        model_library_bool: bool = False,
        deeply_inject_covariates: bool = True,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        n_continuous_cov: int = 0,
        latent_distribution: str = "normal",
        **vae_kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden_l_encoder,
            n_layers=n_layers_l_decoder,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            n_continuous_cov= n_continuous_cov,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=False,
            **vae_kwargs,
        )
        self.use_batch_norm = use_batch_norm
        self.mask = mask
        self.model_library_bool = model_library_bool

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        self.z_encoder = MaskedLatentEncoder(
                    n_input = n_input, 
                    n_output = n_latent, 
                    dim_s = dim_s, 
                    mask = mask,
                    n_layers_masked = n_layers_masked,
                    n_hidden_masked = n_hidden_masked,
                    n_out_masked = n_out_masked,
                    n_layers_FC = n_layers_FC,
                    n_hidden_FC = n_hidden_FC,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = Decoder_residual(
                n_input = n_input_decoder, 
                n_output = n_input, 
                n_layers = n_layers_decoder,
                n_hidden = n_hidden_decoder, 
                dim_s = dim_s,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder
            )
    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        residual_key = 's'
        s = tensors[residual_key] 
        
        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs, s=s
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        
        residual_key = "s"
        s = tensors[residual_key] 

        library = inference_outputs["library"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        y = tensors[_CONSTANTS.LABELS_KEY]

        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None
        input_dict = {
            "z": z,
            "s": s,
            "library": library,
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, s, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        qz_m, qz_v, z = self.z_encoder(encoder_input, s, batch_index, *categorical_input)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded
        # TO-DO: fix multiple sample case
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                library = Normal(ql_m, ql_v.sqrt()).sample()
        
        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs
    
    @auto_move_data
    def generative(
        self,
        z,
        s,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        y=None,
        transform_batch=None,
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, decoder_input, s, library, batch_index, *categorical_input, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )
    
    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        penalty_weight: float = 1.0,
    ):
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY]
        local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY]
		
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]

        mean_key = "mu"
        s_m = tensors[mean_key] if mean_key in tensors.keys() else None
        
        sd_key = "sigma_square"
        s_v = tensors[sd_key] if sd_key in tensors.keys() else None
        
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, qz_v.sqrt()), Normal(mean, scale)).sum(dim=1)
        
        if self.model_library_bool:
            kl_divergence_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_l_mean, torch.sqrt(local_l_var)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        
        kl_divergence_z_s = kl(Normal(qz_m, qz_v.sqrt()), Normal(s_m, s_v.sqrt())).sum(dim=1)

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local - penalty_weight * kl_divergence_z_s)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global, kl_divergence_z_s = kl_divergence_z_s)

class scCLR_Res_module(VAE):
    """
    The "second-phase" model for learning domain specific LVs with shared LVs (fixed)
    as "residual" in the skip connetion layers. 

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer (for encoder)
    n_latent
        Dimensionality of the latent space
    n_layers_encoder
        Number of hidden layers used for encoder NNs
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    use_batch_norm
        Bool whether to use batch norm in decoder
    bias
        Bool whether to have bias term in linear decoder
    """

    def __init__(
        self,
        n_input: int,
        mask: torch.Tensor, 
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_FC: int = 1, 
        n_layers_skip: int = 1, 
        n_layers_decoder: int =2,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        use_batch_norm: bool = True,
        bias: bool = False,
        latent_distribution: str = "normal",
        **vae_kwargs,
    ):
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers_decoder,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=False,
            **vae_kwargs,
        )
        self.use_batch_norm = use_batch_norm
        self.mask = mask

        self.z_encoder = MaskedResEncoder(
            n_input = n_input,
            n_output = n_latent,
            mask = mask,
            n_layers_FC = n_layers_FC,
            n_layers_skip = n_layers_skip,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
        )

    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]

        residuals = tensors["residuals"]
        
        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs, residuals=residuals
        )
        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, residuals, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        qz_m, qz_v, z = self.z_encoder(encoder_input, residuals, batch_index, *categorical_input)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded
        # TO-DO: fix multiple sample case
        '''if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
                ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
                library = Normal(ql_m, ql_v.sqrt()).sample()
        '''
        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs

class scCLR_module_mask_decoder_no_softmax(BaseModuleClass):
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
    indices_mappings
        list of mapping the model inputs to the model output
        Eg: ``[[0,2], [0,1,3,2]]`` means the first dataset has 2 genes that will be reconstructed at location ``[0,2]``
        the second dataset has 4 genes that will be reconstructed at ``[0,1,3,2]``
    gene_likelihoods
        list of distributions to use in the generative process 'zinb', 'nb', 'poisson'
    model_library_bools bool list
        model or not library size with a latent variable or use observed values
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
    dispersion
        See ``vae.py``
    combine_latent
        The way to combine z_individual and z_shared, either 'cat' or 'add'. default is 'cat'
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        indices_mappings: List[Union[np.ndarray, slice]],
        gene_likelihoods: List[str],
        model_library_bools: List[bool],
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
        dispersion: str = "gene-batch",
        log_variational: bool = True,
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.indices_mappings = indices_mappings
        self.gene_likelihoods = gene_likelihoods
        self.model_library_bools = model_library_bools
        self.mask = mask
        self.n_latent = n_latent

        self.n_batch = n_batch
        self.n_labels = n_labels

        self.dispersion = dispersion
        self.log_variational = log_variational


        self.z_encoder = MultiLatentEncoder(
            n_heads=len(dim_input_list),
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            mask = self.mask,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
        )

        self.l_encoders = ModuleList(
            [
                Encoder(
                    self.n_input_list[i],
                    1,
                    n_layers=1,
                    dropout_rate=dropout_rate_encoder,
                )
                if self.model_library_bools[i]
                else None
                for i in range(len(self.n_input_list))
            ]
        )
        
        dim_decoder_input = self.n_latent 
        
        self.decoder = MultiLatentDecoder(
            len(dim_input_list),
            len(dim_input_list) * [self.n_latent],
            self.total_genes,
            mask = self.mask,
            n_hidden = dim_hidden_decoder,
            n_layers=n_layers_decoder,
            #n_cat_list=[self.n_batch],
            dropout_rate=dropout_rate_decoder,
        )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_labels))
        else:  # gene-cell
            pass

    def sample_from_posterior_z(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False, output_z_raw: bool = False
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
        qz_ind_m = outputs["qz_ind_m"]
        z_raw = outputs["z_raw"]
        z_ind_raw = outputs["z_ind_raw"]
        z = outputs["z"]
        z_ind = outputs["z_ind"]
        if deterministic:
            z = qz_m
            z_ind = qz_ind_m
        if output_z_raw:
            z = z_raw
            z_ind = z_ind_raw    
        return dict(z_ind=z_ind, z=z)
    
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

    ## Not changed
    def sample_from_posterior_l(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample the tensor of library sizes from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, 1)``

        """
        _, _, _, _, _, _, ql_m, _, library = self.encode(x, mode)
        if deterministic and ql_m is not None:
            library = ql_m
        return library


    def sample_scale(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Return the tensor of predicted frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            int encode mode (which input head to use in the model)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        deterministic
            bool - whether to sample or not
        decode_mode
            int use to a decode mode different from encoding mode

        Returns
        -------
        type
            tensor of predicted expression

        """
        if decode_mode is None:
            decode_mode = mode
        inference_out = self.inference(x, mode)
        if deterministic:
            z = inference_out["qz_m"]
            z_ind = inference_out["qz_ind_m"]
            if inference_out["ql_m"] is not None:
                library = inference_out["ql_m"]
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
            library = inference_out["library"]
                 
        gen_out = self.generative(z_ind, z, library, batch_index, y, decode_mode)

        return gen_out["px_scale"]


    # This is a potential wrapper for a vae like get_sample_rate
    def get_sample_rate(self, x, batch_index, *_, **__):
        return self.sample_rate(x, 0, batch_index)

    def sample_rate(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
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
        qz_ind_m, qz_ind_v, z_ind, qz_m, qz_v, z, ql_m, ql_v, library = self.encode(x, mode)
        if deterministic:
            z = qz_m
            z_ind = qz_ind_m
            if ql_m is not None:
                library = ql_m
        
        if self.combine_latent == "cat":
            z_decode_input = torch.cat([z_ind,z], dim = -1)
        elif self.combine_latent == "add":
            z_decode_input = z_ind + z 

        px_scale, px_r, px_rate, px_dropout = self.decode(
            z_decode_input, decode_mode, library, batch_index, y
        )

        return px_rate


    def reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        reconstruction_loss = None
        if self.gene_likelihoods[mode] == "zinb":
            reconstruction_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "nb":
            reconstruction_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "poisson":
            reconstruction_loss = -Poisson(px_rate).log_prob(x).sum(dim=1)
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        z_ind = inference_outputs["z_ind"]
        library = inference_outputs["library"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        y = tensors[_CONSTANTS.LABELS_KEY]
        return dict(z_ind=z_ind, z=z, library=library, batch_index=batch_index, y=y)

    @auto_move_data
    def inference(self, x: torch.Tensor, mode: Optional[int] = None) -> dict:
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz_ind_m, qz_ind_v, z_ind, qz_m, qz_v, z = self.z_encoder(x_, mode)
        z_ind_raw = z_ind
        z_raw = z

        ql_m, ql_v, library = None, None, None
        if self.model_library_bools[mode]:
            ql_m, ql_v, library = self.l_encoders[mode](x_)
        else:
            library = torch.log(torch.sum(x, dim=1)).view(-1, 1)

        return dict(z_ind_raw = z_ind_raw, z_raw = z_raw, qz_ind_m=qz_ind_m, qz_ind_v=qz_ind_v, z_ind=z_ind, qz_m=qz_m, qz_v=qz_v, z=z, ql_m=ql_m, ql_v=ql_v, library=library)


    @auto_move_data
    def generative(
        self,
        z_ind: torch.Tensor,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: Optional[int] = None,
    ) -> dict:


        px_scale, px_r, px_rate, px_dropout, path_ind, path_s  = self.decoder(
            z_ind, z, mode, library, self.dispersion, batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r.view(1, self.px_r.size(0))
        px_r = torch.exp(px_r)

        px_scale = px_scale / torch.sum(
            px_scale[:, self.indices_mappings[mode]], dim=1
        ).view(-1, 1)
        px_rate = px_scale * torch.exp(library)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout, path_ind=path_ind, path_s=path_s
        )
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
            z_ind = inference_out["qz_ind_m"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
        
        if output_z_ind:
            z_out = z_ind
        else:
            z_out = z
        
        return z_out

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
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
            z_ind = inference_out["qz_ind_m"]
            if inference_out["ql_m"] is not None:
                library = inference_out["ql_m"]
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
            library = inference_out["library"]
    
        gen_out = self.generative(z_ind, z, library, batch_index, y, decode_mode)
        
        px_rate = gen_out['px_rate']
        px_r = gen_out['px_r']
        px_dropout = gen_out['px_dropout']

        # mask loss to observed genes
        mapping_indices = self.indices_mappings[mode]
        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )
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
        local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY]
        local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qz_ind_m = inference_outputs["qz_ind_m"]
        qz_ind_v = inference_outputs["qz_ind_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # mask loss to observed genes
        mapping_indices = self.indices_mappings[mode]
        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        mean_ind = torch.zeros_like(qz_ind_m)
        scale_ind = torch.ones_like(qz_ind_v)
        kl_divergence_z_ind = kl(Normal(qz_ind_m, torch.sqrt(qz_ind_v)), Normal(mean_ind, scale_ind)).sum(
            dim=1
        )
    
        if self.model_library_bools[mode]:
            kl_divergence_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_l_mean, torch.sqrt(local_l_var)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        kl_local = kl_divergence_l + kl_divergence_z + kl_divergence_z_ind
        kl_global = 0.0
 
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossRecorder(loss, reconstruction_loss, kl_local, kl_global)

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



class scCLR_module_mask_decoder(BaseModuleClass):
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
    indices_mappings
        list of mapping the model inputs to the model output
        Eg: ``[[0,2], [0,1,3,2]]`` means the first dataset has 2 genes that will be reconstructed at location ``[0,2]``
        the second dataset has 4 genes that will be reconstructed at ``[0,1,3,2]``
    gene_likelihoods
        list of distributions to use in the generative process 'zinb', 'nb', 'poisson'
    model_library_bools bool list
        model or not library size with a latent variable or use observed values
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
    dispersion
        See ``vae.py``
    combine_latent
        The way to combine z_individual and z_shared, either 'cat' or 'add'. default is 'cat'
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        indices_mappings: List[Union[np.ndarray, slice]],
        gene_likelihoods: List[str],
        model_library_bools: List[bool],
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
        dispersion: str = "gene-batch",
        log_variational: bool = True,
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.indices_mappings = indices_mappings
        self.gene_likelihoods = gene_likelihoods
        self.model_library_bools = model_library_bools
        self.mask = mask
        self.n_latent = n_latent

        self.n_batch = n_batch
        self.n_labels = n_labels

        self.dispersion = dispersion
        self.log_variational = log_variational


        self.z_encoder = MultiLatentEncoder(
            n_heads=len(dim_input_list),
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            mask = self.mask,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
        )

        self.l_encoders = ModuleList(
            [
                Encoder(
                    self.n_input_list[i],
                    1,
                    n_layers=1,
                    dropout_rate=dropout_rate_encoder,
                )
                if self.model_library_bools[i]
                else None
                for i in range(len(self.n_input_list))
            ]
        )
        
        dim_decoder_input = self.n_latent 
        
        self.decoder = MultiLatentDecoder(
            len(dim_input_list),
            dim_decoder_input,
            self.total_genes,
            mask = self.mask,
            n_hidden = dim_hidden_decoder,
            n_layers=n_layers_decoder,
            #n_cat_list=[self.n_batch],
            dropout_rate=dropout_rate_decoder,
        )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_labels))
        else:  # gene-cell
            pass

    def sample_from_posterior_z(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False, output_z_raw: bool = False
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
        qz_ind_m = outputs["qz_ind_m"]
        z_raw = outputs["z_raw"]
        z_ind_raw = outputs["z_ind_raw"]
        z = outputs["z"]
        z_ind = outputs["z_ind"]
        if deterministic:
            z = qz_m
            z_ind = qz_ind_m
        if output_z_raw:
            z = z_raw
            z_ind = z_ind_raw    
        return dict(z_ind=z_ind, z=z)
    
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

    ## Not changed
    def sample_from_posterior_l(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample the tensor of library sizes from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, 1)``

        """
        _, _, _, _, _, _, ql_m, _, library = self.encode(x, mode)
        if deterministic and ql_m is not None:
            library = ql_m
        return library


    def sample_scale(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Return the tensor of predicted frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            int encode mode (which input head to use in the model)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        deterministic
            bool - whether to sample or not
        decode_mode
            int use to a decode mode different from encoding mode

        Returns
        -------
        type
            tensor of predicted expression

        """
        if decode_mode is None:
            decode_mode = mode
        inference_out = self.inference(x, mode)
        if deterministic:
            z = inference_out["qz_m"]
            z_ind = inference_out["qz_ind_m"]
            if inference_out["ql_m"] is not None:
                library = inference_out["ql_m"]
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
            library = inference_out["library"]
                 
        gen_out = self.generative(z_ind, z, library, batch_index, y, decode_mode)

        return gen_out["px_scale"]


    # This is a potential wrapper for a vae like get_sample_rate
    def get_sample_rate(self, x, batch_index, *_, **__):
        return self.sample_rate(x, 0, batch_index)

    def sample_rate(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
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
        qz_ind_m, qz_ind_v, z_ind, qz_m, qz_v, z, ql_m, ql_v, library = self.encode(x, mode)
        if deterministic:
            z = qz_m
            z_ind = qz_ind_m
            if ql_m is not None:
                library = ql_m
        
        if self.combine_latent == "cat":
            z_decode_input = torch.cat([z_ind,z], dim = -1)
        elif self.combine_latent == "add":
            z_decode_input = z_ind + z 

        px_scale, px_r, px_rate, px_dropout = self.decode(
            z_decode_input, decode_mode, library, batch_index, y
        )

        return px_rate


    def reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        reconstruction_loss = None
        if self.gene_likelihoods[mode] == "zinb":
            reconstruction_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "nb":
            reconstruction_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "poisson":
            reconstruction_loss = -Poisson(px_rate).log_prob(x).sum(dim=1)
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        z_ind = inference_outputs["z_ind"]
        library = inference_outputs["library"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        y = tensors[_CONSTANTS.LABELS_KEY]
        return dict(z_ind=z_ind, z=z, library=library, batch_index=batch_index, y=y)

    @auto_move_data
    def inference(self, x: torch.Tensor, mode: Optional[int] = None) -> dict:
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz_ind_m, qz_ind_v, z_ind, qz_m, qz_v, z = self.z_encoder(x_, mode)
        z_ind_raw = z_ind
        z_raw = z
        # take the softmax to normalize z_ind and z
        concat_z = torch.concat([z_ind, z], dim=-1)
        concat_z = nn.Softmax(dim=-1)(concat_z)
        z_ind, z = torch.split(concat_z, self.n_latent, dim=-1)

        ql_m, ql_v, library = None, None, None
        if self.model_library_bools[mode]:
            ql_m, ql_v, library = self.l_encoders[mode](x_)
        else:
            library = torch.log(torch.sum(x, dim=1)).view(-1, 1)

        return dict(z_ind_raw = z_ind_raw, z_raw = z_raw, qz_ind_m=qz_ind_m, qz_ind_v=qz_ind_v, z_ind=z_ind, qz_m=qz_m, qz_v=qz_v, z=z, ql_m=ql_m, ql_v=ql_v, library=library)


    @auto_move_data
    def generative(
        self,
        z_ind: torch.Tensor,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: Optional[int] = None,
    ) -> dict:


        px_scale, px_r, px_rate, px_dropout, path = self.decoder(
            z_ind, z, mode, library, self.dispersion, batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r.view(1, self.px_r.size(0))
        px_r = torch.exp(px_r)

        px_scale = px_scale / torch.sum(
            px_scale[:, self.indices_mappings[mode]], dim=1
        ).view(-1, 1)
        px_rate = px_scale * torch.exp(library)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout, path=path
        )
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
            z_ind = inference_out["qz_ind_m"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
        
        if output_z_ind:
            z_out = z_ind
        else:
            z_out = z
        
        return z_out

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
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
            z_ind = inference_out["qz_ind_m"]
            if inference_out["ql_m"] is not None:
                library = inference_out["ql_m"]
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
            library = inference_out["library"]
    
        gen_out = self.generative(z_ind, z, library, batch_index, y, decode_mode)
        
        px_rate = gen_out['px_rate']
        px_r = gen_out['px_r']
        px_dropout = gen_out['px_dropout']

        # mask loss to observed genes
        mapping_indices = self.indices_mappings[mode]
        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )
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
        local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY]
        local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qz_ind_m = inference_outputs["qz_ind_m"]
        qz_ind_v = inference_outputs["qz_ind_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # mask loss to observed genes
        mapping_indices = self.indices_mappings[mode]
        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        mean_ind = torch.zeros_like(qz_ind_m)
        scale_ind = torch.ones_like(qz_ind_v)
        kl_divergence_z_ind = kl(Normal(qz_ind_m, torch.sqrt(qz_ind_v)), Normal(mean_ind, scale_ind)).sum(
            dim=1
        )
    
        if self.model_library_bools[mode]:
            kl_divergence_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_l_mean, torch.sqrt(local_l_var)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        kl_local = kl_divergence_l + kl_divergence_z + kl_divergence_z_ind
        kl_global = 0.0
 
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossRecorder(loss, reconstruction_loss, kl_local, kl_global)

class scCLR_module(BaseModuleClass):
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
    indices_mappings
        list of mapping the model inputs to the model output
        Eg: ``[[0,2], [0,1,3,2]]`` means the first dataset has 2 genes that will be reconstructed at location ``[0,2]``
        the second dataset has 4 genes that will be reconstructed at ``[0,1,3,2]``
    gene_likelihoods
        list of distributions to use in the generative process 'zinb', 'nb', 'poisson'
    model_library_bools bool list
        model or not library size with a latent variable or use observed values
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
    dispersion
        See ``vae.py``
    combine_latent
        The way to combine z_individual and z_shared, either 'cat' or 'add'. default is 'cat'
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        indices_mappings: List[Union[np.ndarray, slice]],
        gene_likelihoods: List[str],
        model_library_bools: List[bool],
        mask: torch.Tensor = None, 
        n_latent: int = 10,
        n_layers_encoder_individual: int = 1,
        n_layers_encoder_shared: int = 2,
        dim_hidden_encoder: int = 64,
        n_layers_decoder_individual: int = 0,
        n_layers_decoder_shared: int = 1,
        dim_hidden_decoder_individual: int = 64,
        dim_hidden_decoder_shared: int = 64,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_decoder: float = 0.1,
        n_batch: int = 0,
        n_labels: int = 0,
        dispersion: str = "gene-batch",
        combine_latent: str = 'cat',  # add
        log_variational: bool = True,
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.indices_mappings = indices_mappings
        self.gene_likelihoods = gene_likelihoods
        self.model_library_bools = model_library_bools
        self.mask = mask
        self.n_latent = n_latent

        self.n_batch = n_batch
        self.n_labels = n_labels

        self.dispersion = dispersion
        self.combine_latent = combine_latent
        self.log_variational = log_variational


    

        self.z_encoder = MultiLatentEncoder(
            n_heads=len(dim_input_list),
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            mask = self.mask,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
        )

        self.l_encoders = ModuleList(
            [
                Encoder(
                    self.n_input_list[i],
                    1,
                    n_layers=1,
                    dropout_rate=dropout_rate_encoder,
                )
                if self.model_library_bools[i]
                else None
                for i in range(len(self.n_input_list))
            ]
        )
        
        if self.combine_latent == "cat":
            dim_decoder_input = self.n_latent + self.n_latent 
        elif self.combine_latent == "add":
            dim_decoder_input = self.n_latent 
        
        self.decoder = MultiDecoder(
            dim_decoder_input,
            self.total_genes,
            n_hidden_conditioned=dim_hidden_decoder_individual,
            n_hidden_shared=dim_hidden_decoder_shared,
            n_layers_conditioned=n_layers_decoder_individual,
            n_layers_shared=n_layers_decoder_shared,
            n_cat_list=[self.n_batch],
            dropout_rate=dropout_rate_decoder,
        )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_labels))
        else:  # gene-cell
            pass

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
        qz_ind_m = outputs["qz_ind_m"]

        z = outputs["z"]
        z_ind = outputs["z_ind"]
        if deterministic:
            z = qz_m
            z_ind = qz_ind_m
        return dict(z_ind=z_ind, z=z)
    
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

    ## Not changed
    def sample_from_posterior_l(
        self, x: torch.Tensor, mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample the tensor of library sizes from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, 1)``

        """
        _, _, _, _, _, _, ql_m, _, library = self.encode(x, mode)
        if deterministic and ql_m is not None:
            library = ql_m
        return library


    def sample_scale(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Return the tensor of predicted frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            int encode mode (which input head to use in the model)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        deterministic
            bool - whether to sample or not
        decode_mode
            int use to a decode mode different from encoding mode

        Returns
        -------
        type
            tensor of predicted expression

        """
        if decode_mode is None:
            decode_mode = mode
        inference_out = self.inference(x, mode)
        if deterministic:
            z = inference_out["qz_m"]
            z_ind = inference_out["qz_ind_m"]
            if inference_out["ql_m"] is not None:
                library = inference_out["ql_m"]
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
            library = inference_out["library"]
                 
        gen_out = self.generative(z_ind, z, library, batch_index, y, decode_mode)

        return gen_out["px_scale"]


    # This is a potential wrapper for a vae like get_sample_rate
    def get_sample_rate(self, x, batch_index, *_, **__):
        return self.sample_rate(x, 0, batch_index)

    def sample_rate(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
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
        qz_ind_m, qz_ind_v, z_ind, qz_m, qz_v, z, ql_m, ql_v, library = self.encode(x, mode)
        if deterministic:
            z = qz_m
            z_ind = qz_ind_m
            if ql_m is not None:
                library = ql_m
        
        if self.combine_latent == "cat":
            z_decode_input = torch.cat([z_ind,z], dim = -1)
        elif self.combine_latent == "add":
            z_decode_input = z_ind + z 

        px_scale, px_r, px_rate, px_dropout = self.decode(
            z_decode_input, decode_mode, library, batch_index, y
        )

        return px_rate


    def reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        reconstruction_loss = None
        if self.gene_likelihoods[mode] == "zinb":
            reconstruction_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "nb":
            reconstruction_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "poisson":
            reconstruction_loss = -Poisson(px_rate).log_prob(x).sum(dim=1)
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        z_ind = inference_outputs["z_ind"]
        library = inference_outputs["library"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        y = tensors[_CONSTANTS.LABELS_KEY]
        return dict(z_ind=z_ind, z=z, library=library, batch_index=batch_index, y=y)

    @auto_move_data
    def inference(self, x: torch.Tensor, mode: Optional[int] = None) -> dict:
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz_ind_m, qz_ind_v, z_ind, qz_m, qz_v, z = self.z_encoder(x_, mode)
        #qz_m, qz_v, z = self.z_encoder(x_, mode)
        ql_m, ql_v, library = None, None, None
        if self.model_library_bools[mode]:
            ql_m, ql_v, library = self.l_encoders[mode](x_)
        else:
            library = torch.log(torch.sum(x, dim=1)).view(-1, 1)

        return dict(qz_ind_m=qz_ind_m, qz_ind_v=qz_ind_v, z_ind=z_ind, qz_m=qz_m, qz_v=qz_v, z=z, ql_m=ql_m, ql_v=ql_v, library=library)


    @auto_move_data
    def generative(
        self,
        z_ind: torch.Tensor,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: Optional[int] = None,
    ) -> dict:

        if self.combine_latent == "cat":
            z_decoder_input = torch.cat([z_ind,z], dim = -1)
        elif self.combine_latent == "add":
            z_decoder_input = z_ind + z

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            z_decoder_input, mode, library, self.dispersion, batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r.view(1, self.px_r.size(0))
        px_r = torch.exp(px_r)

        px_scale = px_scale / torch.sum(
            px_scale[:, self.indices_mappings[mode]], dim=1
        ).view(-1, 1)
        px_rate = px_scale * torch.exp(library)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )
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
            z_ind = inference_out["qz_ind_m"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
        
        if output_z_ind:
            z_out = z_ind
        else:
            z_out = z
        
        return z_out

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
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
            z_ind = inference_out["qz_ind_m"]
            if inference_out["ql_m"] is not None:
                library = inference_out["ql_m"]
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            z_ind = inference_out["z_ind"]
            library = inference_out["library"]
    
        gen_out = self.generative(z_ind, z, library, batch_index, y, decode_mode)
        
        px_rate = gen_out['px_rate']
        px_r = gen_out['px_r']
        px_dropout = gen_out['px_dropout']

        # mask loss to observed genes
        mapping_indices = self.indices_mappings[mode]
        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )
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
        local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY]
        local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qz_ind_m = inference_outputs["qz_ind_m"]
        qz_ind_v = inference_outputs["qz_ind_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # mask loss to observed genes
        mapping_indices = self.indices_mappings[mode]
        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        mean_ind = torch.zeros_like(qz_ind_m)
        scale_ind = torch.ones_like(qz_ind_v)
        kl_divergence_z_ind = kl(Normal(qz_ind_m, torch.sqrt(qz_ind_v)), Normal(mean_ind, scale_ind)).sum(
            dim=1
        )
    
        if self.model_library_bools[mode]:
            kl_divergence_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_l_mean, torch.sqrt(local_l_var)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        kl_local = kl_divergence_l + kl_divergence_z + kl_divergence_z_ind
        kl_global = 0.0
 
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossRecorder(loss, reconstruction_loss, kl_local, kl_global)


