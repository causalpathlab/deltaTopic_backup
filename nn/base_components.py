# -*- coding: utf-8 -*-
"""base components"""
import collections
from typing import Callable, Iterable, List, Optional
#from scvi.nn import MultiEncoder
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList
from scvi.nn import FCLayers, Encoder
import torch.nn.functional as F
from scvi.nn import one_hot


torch.backends.cudnn.benchmark = True

def identity(x):
    return x

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

class Decoder_residual(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions into ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        dim_s: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        
        parameter_decoder_input = n_hidden + dim_s

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(parameter_decoder_input, n_output),
            nn.Softmax(dim=-1),
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(parameter_decoder_input, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(parameter_decoder_input, n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, s: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        s :
            residual tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        # concat with s
        px = torch.cat([px, s], dim = -1)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

class MaskedLatentEncoder(nn.Module):
    """
    Masked latnet encoder with single head

    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.
    
    Masked encoder Layers: 
        n_input --> mask --> n_hidden_maksed * n_layers_masked --> concat with z_shared --> 
    Fully-connected layers: 
        FCLayers * n_hidden_FC --> mu, signma  --> n_output (z)

    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        dim_s: int,
        mask: torch.Tensor = None,
        mask_first: bool = True,
        n_layers_masked: int = 1,
        n_hidden_masked: int = 128,
        n_out_masked: int = 128,
        n_layers_FC: int = 1,
        n_hidden_FC: int = 128,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.masked_encoder = MaskedLinearLayers(
                    n_in=n_input,
                    n_out=n_out_masked,
                    n_cat_list=n_cat_list,
                    mask=mask,
                    mask_first=mask_first,
                    n_layers=n_layers_masked,
                    n_hidden=n_hidden_masked,
                    dropout_rate=dropout_rate,
                    inject_covariates=inject_covariates,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                )
                
        n_input_FClayer = n_out_masked + dim_s
        
        self.FClayers = FCLayers(
            n_in=n_input_FClayer,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers_FC,
            n_hidden=n_hidden_FC,
            dropout_rate=dropout_rate,
        )
        
        self.mean_encoder = nn.Linear(n_output, n_output)
        self.var_encoder = nn.Linear(n_output, n_output)

    def forward(self, x: torch.Tensor, s: torch.Tensor, *cat_list: int):
        z_ind = self.masked_encoder(x, *cat_list)
        z_cat = torch.cat([z_ind,s], dim = -1)
        q = self.FClayers(z_cat, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent

class MultiLatentEncoder(nn.Module):
    """
    Maksed latnet encoder with two heads
    """
    def __init__(
        self,
        n_heads: int,
        n_input_list: List[int],
        n_output: int,
        mask: torch.Tensor = None,
        mask_first: bool = True,
        n_hidden: int = 128,
        n_layers_individual: int = 1,
        n_layers_shared: int = 2,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
        deeply_inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        

        self.encoders = ModuleList(
            [
                MaskedLinearLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    mask=mask,
                    mask_first=mask_first,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        # to encode source-specific mean and var
        self.mean_encoders = ModuleList(
            [
                nn.Linear(n_hidden, n_output)
                for i in range(n_heads)
            ]
        )

        self.var_encoders = ModuleList(
            [
                nn.Linear(n_hidden, n_output)
                for i in range(n_heads)
            ]
        )

    def forward(self, x: torch.Tensor, head_id: int, *cat_list: int):
        q = self.encoders[head_id](x, *cat_list)
        
        q_m_ind = self.mean_encoders[head_id](q)
        q_v_ind = torch.exp(self.var_encoders[head_id](q))
        latent_ind = reparameterize_gaussian(q_m_ind, q_v_ind)
        
        q = self.encoder_shared(q, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m_ind, q_v_ind, latent_ind, q_m, q_v, latent

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', mask)
        
    def forward(self, input):
        #mask = Variable(self.mask, requires_grad=False)
        if self.bias is None:
            return F.linear(input, self.weight*self.mask)
        else:
            return F.linear(input, self.weight*self.mask, self.bias)

class MaskedResEncoder(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a Mask induced connected and skip connections.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    mask
        mask must be supplied, no mask option has not been tested with ResLayer
    mask_first
        mask first
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers_FC
        The number of fully-connected hidden layers in mask encoder
    n_layers_skip
        The number of fully-connected in skip connections
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        When `None`, defaults to `torch.exp`.
    **kwargs
        Keyword args for :class:`~scvi.module._base.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        mask: torch.Tensor,
        mask_first: bool = True,
        n_cat_list: Iterable[int] = None,
        n_layers_FC: int = 1,
        n_layers_skip: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        deeply_inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        var_activation: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        
        self.encoder_mask = MaskedLinearLayers(
                    n_in=n_input,
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    mask=mask,
                    mask_first=mask_first,
                    n_layers=n_layers_FC,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
        )
        self.encoder_skip = FCResLayer(
            n_cat_list=n_cat_list,
            n_layers=n_layers_skip,
            n_hidden=n_output,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, residual: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample
        residual
            residual
        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder_mask(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        latent_raw = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        # skip connection
        latent = self.encoder_skip(latent_raw, residual, *cat_list)
        return q_m, q_v, latent

class FCResLayer(FCLayers):
    """ Fully connected layers with skip connection 
    ----------
    
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer (default), or just the first.
    inject_residual
        Whether to inject residual with summation in each linear layer (default), or just the first. 
        Note that the dimension of each linear layer should be same as the residual
    activation_fn
        Which activation function to use the
    """

    def __init__(
        self, 
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        inject_residual: bool = True,
        activation_fn: nn.Module = nn.ReLU 
        ):
            
        super().__init__(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers-1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=use_activation,
            bias=bias,
            inject_covariates=inject_covariates,
            activation_fn=activation_fn
            )
        self.inject_residual = inject_residual    
        self.n_hidden = n_hidden
    
        layers_dim =  (n_layers + 1) * [n_hidden] 

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
            
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "ResLayer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def inject_residual_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_residual)
        return user_cond
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        residual
            Tenosors of residuals to be used at Residual layers
            The dimension must match the hidden dimension, e.g., [batch, sample, hidden]
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert x.shape == residual.shape
        
        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        # skip connection  
                        if isinstance(layer, nn.Linear) and self.inject_residual_into_layer(i):
                            x = x + residual
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        
                        x = layer(x)
        return x


class MultiLatentDecoder(nn.Module):
    """ domain specifc decoder and one shared decoder"""
    def __init__(
        self,
        n_heads: int,
        n_input_list: List[int],
        n_output: int,
        mask: torch.Tensor = None, 
        n_hidden: int = 128, # by default, no hidden layers
        n_layers: int = 1, # by default, no hidden layers
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        n_path = mask.shape[0]

        self.pathway_decoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_path,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                )
                for i in range(n_heads)
            ]
        )

        self.pathway_decoder_shared = FCLayers(
            n_in=n_input_list[0],
            n_out=n_path,
            n_cat_list=[],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
        )

        self.masked_decoder = MaskedLinear(n_path, n_output, torch.transpose(mask,0,1))

        n_in = n_output # number of genes

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_in, n_output), nn.Softmax(dim=-1)
        )
        self.px_r_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        z_ind: torch.Tensor,
        z_s: torch.Tensor,
        head_id:int,
        library: torch.Tensor,
        dispersion: str,
        *cat_list: int,
    ):

        # LV --> pathway representation
        path_ind = self.pathway_decoders[head_id](z_ind, *cat_list)
        #path_ind = nn.Softmax(dim=-1)(path_ind)
        path_s = self.pathway_decoder_shared(z_s, *cat_list)
        #path_s = nn.Softmax(dim=-1)(path_s)
        #path = 0.5 * (path_ind + path_s) 

        # pathway representation --> gene
        px_ind = self.masked_decoder(path_ind)
        px_s = self.masked_decoder(path_s)
        px = px_ind + px_s

        # get the parameters of the model
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_r, px_rate, px_dropout, path_ind, path_s

class DeltaDecoder(nn.Module):
    """Break down the differences between spliced and unplisced with a topic-by-gene matrix """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_conditioned: int = 32,
        n_hidden_shared: int = 128,
        n_layers_conditioned: int = 1,
        n_layers_shared: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        n_out = n_hidden_conditioned if n_layers_shared else n_hidden_shared
        

        
        
        if n_layers_conditioned:
            self.px_decoder_conditioned = FCLayers(
                n_in=n_input,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_layers=n_layers_conditioned,
                n_hidden=n_hidden_conditioned,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_out
        else:
            self.px_decoder_conditioned = None
            n_in = n_input

        if n_layers_shared:
            self.px_decoder_final = FCLayers(
                n_in=n_in,
                n_out=n_hidden_shared,
                n_cat_list=[],
                n_layers=n_layers_shared,
                n_hidden=n_hidden_shared,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_hidden_shared
        else:
            self.px_decoder_final = None

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_in, n_output), nn.Softmax(dim=-1)
        )
        self.px_r_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: int,
        library: torch.Tensor,
        dispersion: str,
        *cat_list: int,
    ):

        px = z
        if self.px_decoder_conditioned:
            px = self.px_decoder_conditioned(px, *cat_list)
        if self.px_decoder_final:
            px = self.px_decoder_final(px, *cat_list)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_r, px_rate, px_dropout


class MultiDecoder(nn.Module):
    """This is the multi-decoder in scvi.nn, included here for reference"""
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_conditioned: int = 32,
        n_hidden_shared: int = 128,
        n_layers_conditioned: int = 1,
        n_layers_shared: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        n_out = n_hidden_conditioned if n_layers_shared else n_hidden_shared
        if n_layers_conditioned:
            self.px_decoder_conditioned = FCLayers(
                n_in=n_input,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_layers=n_layers_conditioned,
                n_hidden=n_hidden_conditioned,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_out
        else:
            self.px_decoder_conditioned = None
            n_in = n_input

        if n_layers_shared:
            self.px_decoder_final = FCLayers(
                n_in=n_in,
                n_out=n_hidden_shared,
                n_cat_list=[],
                n_layers=n_layers_shared,
                n_hidden=n_hidden_shared,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_hidden_shared
        else:
            self.px_decoder_final = None

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_in, n_output), nn.Softmax(dim=-1)
        )
        self.px_r_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: int,
        library: torch.Tensor,
        dispersion: str,
        *cat_list: int,
    ):

        px = z
        if self.px_decoder_conditioned:
            px = self.px_decoder_conditioned(px, *cat_list)
        if self.px_decoder_final:
            px = self.px_decoder_final(px, *cat_list)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_r, px_rate, px_dropout



# Yichen: this incorporates the one-hot encoding for for category input.
class MaskedLinearLayers(FCLayers):
    """
    A helper class to build Masked Linear layers compatible with FClayer
    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    mask
        The mask, should be dimension n_out * n_in
    mask_first
        wheather mask linear layer should be before or after fully-connected layers, default is true;
        False is useful to construct an decoder with the oposite strucutre (mask linear after fully connected)
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self, 
        n_in: int,
        n_out: int,
        mask: torch.Tensor = None,
        mask_first: bool = True,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU
        ):
            
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=use_activation,
            bias=bias,
            inject_covariates=inject_covariates,
            activation_fn=activation_fn
            )

        self.mask = mask ## out_features, in_features

        if mask is None:
            print("No mask input, use all fully connected layers")

        
        if mask is not None:
            if mask_first:
                layers_dim = [n_in] + [mask.shape[0]] + (n_layers - 1) * [n_hidden] + [n_out]
            else:
                layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [mask.shape[0]] + [n_out]
        else:    
            layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)

        # concatnat one hot encoding to mask if available
        if cat_dim>0:
            mask_input = torch.cat((self.mask, torch.ones(cat_dim, self.mask.shape[1])), dim=0)
        else:
            mask_input = self.mask        

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )
        if mask is not None:
            if mask_first:
                # change the first layer to be MaskedLinear
                self.fc_layers[0] = nn.Sequential(
                                            MaskedLinear(
                                                layers_dim[0] + cat_dim * self.inject_into_layer(0),
                                                layers_dim[1],
                                                mask_input,
                                                bias=bias,
                                            ),
                                            # non-default params come from defaults in original Tensorflow implementation
                                            nn.BatchNorm1d(layers_dim[1], momentum=0.01, eps=0.001)
                                            if use_batch_norm
                                            else None,
                                            nn.LayerNorm(layers_dim[1], elementwise_affine=False)
                                            if use_layer_norm
                                            else None,
                                            activation_fn() if use_activation else None,
                                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                                            )
            else:
                # change the last layer to be MaskedLinear
                self.fc_layers[-1] = nn.Sequential(
                                            MaskedLinear(
                                                layers_dim[-2] + cat_dim * self.inject_into_layer(0),
                                                layers_dim[-1],
                                                torch.transpose(mask_input,0,1),
                                                bias=bias,
                                            ),
                                            # non-default params come from defaults in original Tensorflow implementation
                                            nn.BatchNorm1d(layers_dim[-1], momentum=0.01, eps=0.001)
                                            if use_batch_norm
                                            else None,
                                            nn.LayerNorm(layers_dim[-1], elementwise_affine=False)
                                            if use_layer_norm
                                            else None,
                                            activation_fn() if use_activation else None,
                                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                                            )


    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if (isinstance(layer, nn.Linear) or isinstance(layer, MaskedLinear)) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x
       


