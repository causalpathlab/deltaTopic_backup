import logging
import os
import pickle
import warnings
from itertools import cycle
from typing import List, Optional, Union
from typing_extensions import Literal
import pandas as pd
import numpy as np
import torch
from anndata import AnnData, read
from torch.utils.data import DataLoader

from scvi import _CONSTANTS
from scvi.data import transfer_anndata_setup
from scvi.dataloaders import DataSplitter
from scvi.model._utils import _get_var_names_from_setup_anndata, parse_use_gpu_arg
from scvi.model.base import RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass
from scvi.train import Trainer

from module.scCLR_module import scCLR_module, scCLR_Res_module, scCLR_phase2_module, scCLR_module_mask_decoder, scCLR_module_mask_decoder_no_softmax
from task.scCLRTrainingPlan import scCLRTrainingPlan, Phase2ModelTrainingPlan
from scvi.train import TrainRunner

from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients, LayerDeepLift, LayerDeepLiftShap
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

logger = logging.getLogger(__name__)



def _unpack_tensors(tensors):
    x = tensors[_CONSTANTS.X_KEY].squeeze_(0)
    local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY].squeeze_(0)
    local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY].squeeze_(0)
    batch_index = tensors[_CONSTANTS.BATCH_KEY].squeeze_(0)
    y = tensors[_CONSTANTS.LABELS_KEY].squeeze_(0)
    return x, local_l_mean, local_l_var, batch_index, y

def set_up_adata_pathway(adata_pathway, minGenes=10):
    # remove bad pathways
    mask = adata_pathway.X.copy()
    pathway_size = np.sum(mask,1)
    bad_pathway = pathway_size < minGenes
    mask[bad_pathway,:] = 0
    adata_pathway.X = mask
    print(f'Removing {np.sum(bad_pathway)} pathways less than {minGenes} genes\n')
    return(adata_pathway)

class scCLR_mask_decoder_no_softmax(VAEMixin, BaseModelClass):
    """
    scCLR

    Parameters
    ----------
    adata_source1
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains source1 data.
    adata_source2
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains source2 data.
    adata_pathway
        Anndata object AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_source1
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    generative_distributions
        List of generative distribution for adata_seq data and adata_spatial data.
    model_library_size
        List of bool of whether to model library size for adata_seq and adata_spatial.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.scCLR_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> adata_spatial = anndata.read_h5ad(path_to_anndata_spatial)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> scvi.data.setup_anndata(adata_spatial)
    >>> vae = scvi.model.GIMVI(adata_seq, adata_spatial)
    >>> vae.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_spatial: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        generative_distributions: List = ["zinb", "zinb"],
        model_library_size: List = [True, True],
        n_latent: int = 10,
        combine_latent: str = 'cat',
        **model_kwargs,
    ):
        super(scCLR_mask_decoder_no_softmax, self).__init__()
        self.n_latent = n_latent
        self.adatas = [adata_seq, adata_spatial]
        self.scvi_setup_dicts_ = {
            "seq": adata_seq.uns["_scvi"],
            "spatial": adata_spatial.uns["_scvi"],
        }

        seq_var_names = _get_var_names_from_setup_anndata(adata_seq)
        spatial_var_names = _get_var_names_from_setup_anndata(adata_spatial)

        if not set(spatial_var_names) <= set(seq_var_names):
            raise ValueError("source2 input genes needs to be subset of source 1 input genes, note this is only for gene imputation purpose")
 
        if adata_pathway is not None:
            # condition check
            pathway_var_names = _get_var_names_from_setup_anndata(adata_pathway)
            if not set(seq_var_names) == set(pathway_var_names):
                raise ValueError("source 1 input genes needs to be equal to pathway genes")
            # get pathway_gene_loc
            pathway_gene_loc = [np.argwhere(seq_var_names == g)[0] for g in pathway_var_names]
            pathway_gene_loc = np.concatenate(pathway_gene_loc)

        spatial_gene_loc = [
            np.argwhere(seq_var_names == g)[0] for g in spatial_var_names
        ]
        spatial_gene_loc = np.concatenate(spatial_gene_loc)
        gene_mappings = [slice(None), spatial_gene_loc]
        sum_stats = [d.uns["_scvi"]["summary_stats"] for d in self.adatas]
        n_inputs = [s["n_vars"] for s in sum_stats]

        total_genes = adata_seq.uns["_scvi"]["summary_stats"]["n_vars"]

        # since we are combining datasets, we need to increment the batch_idx
        # of one of the datasets
        adata_seq_n_batches = adata_seq.uns["_scvi"]["summary_stats"]["n_batch"]
        adata_spatial.obs["_scvi_batch"] += adata_seq_n_batches

        n_batches = sum([s["n_batch"] for s in sum_stats])
        
        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified input\n")
        else:
            self.mask = None
            print("No pathways, use fully-connected layers\n")
            
        self.module = scCLR_module_mask_decoder_no_softmax(
            n_inputs,
            total_genes,
            gene_mappings,
            generative_distributions,
            model_library_size,
            mask=self.mask,
            n_batch=n_batches,
            n_latent=n_latent,
            **model_kwargs,
        )

        self._model_summary_string = (
            "scCLR Model with the following params: \nn_latent: {}, n_inputs: {}, n_genes: {}, "
            + "n_batch: {}, generative distributions: {}, combine latent space: {}"
        ).format(n_latent, n_inputs, total_genes, n_batches, generative_distributions, combine_latent)
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int = 200,
        use_gpu: Optional[Union[str, int, bool]] = None,
        kappa: int = 5,
        gamma: int = 5,  
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        kappa
            Scaling parameter for the discriminator loss, defaut is 5
        gamma
            Scaling parameter for the classification loss, default is 5
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        plan_kwargs
            Keyword args for model-specific Pytorch Lightning task. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        gpus, device = parse_use_gpu_arg(use_gpu)

        self.trainer = Trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            **kwargs,
        )
        self.train_indices_, self.test_indices_, self.validation_indices_ = [], [], []
        train_dls, test_dls, val_dls = [], [], []
        for i, ad in enumerate(self.adatas):
            ds = DataSplitter(
                ad,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
            ds.setup()
            train_dls.append(ds.train_dataloader())
            test_dls.append(ds.test_dataloader())
            val = ds.val_dataloader()
            val_dls.append(val)
            val.mode = i
            self.train_indices_.append(ds.train_idx)
            self.test_indices_.append(ds.test_idx)
            self.validation_indices_.append(ds.val_idx)
        train_dl = TrainDL(train_dls)

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        self._training_plan = scCLRTrainingPlan(
            self.module,
            source_classifier=True,
            scale_classification_loss=gamma,
            adversarial_classifier=True,
            scale_adversarial_loss=kappa,
            **plan_kwargs,
        )

        if train_size == 1.0:
            # circumvent the empty data loader problem if all dataset used for training
            self.trainer.fit(self._training_plan, train_dl)
        else:
            # accepts list of val dataloaders
            self.trainer.fit(self._training_plan, train_dl, val_dls)
        try:
            self.history_ = self.trainer.logger.history
        except AttributeError:
            self.history_ = None
        self.module.eval()

        self.to_device(device)
        self.is_trained_ = True


    def _make_scvi_dls(self, adatas: List[AnnData] = None, batch_size=128):
        if adatas is None:
            adatas = self.adatas
        post_list = [self._make_data_loader(ad) for ad in adatas]
        for i, dl in enumerate(post_list):
            dl.mode = i

        return post_list


    @torch.no_grad()
    def get_parameters_z_shared(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)
        self.module.eval()
        latent_shared_parameters = []
        
        for mode, scdl in enumerate(scdls):
            qz_m = []
            qz_v = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                z_dict  = self.module.get_latent_parameter_z_shared(sample_batch, mode)
                qz_m.append(z_dict["qz_m"])
                qz_v.append(z_dict["qz_v"])                

            latent_m = torch.cat(qz_m).cpu().detach().numpy()
            latent_v = torch.cat(qz_v).cpu().detach().numpy()
            
            latent_shared_parameters.append(dict(latent_m=latent_m, latent_v=latent_v))

        return latent_shared_parameters

    @torch.no_grad()
    def get_latent_representation(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)
        self.module.eval()
        latents = []
        
        for mode, scdl in enumerate(scdls):
            latent_z_ind = []
            latent_z = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                z_dict  = self.module.sample_from_posterior_z(sample_batch, mode, deterministic=deterministic)
                latent_z_ind.append(z_dict["z_ind"])
                latent_z.append(z_dict["z"])                

            latent_z = torch.cat(latent_z).cpu().detach().numpy()
            latent_z_ind = torch.cat(latent_z_ind).cpu().detach().numpy()
            
            latents.append(dict(latent_z_ind=latent_z_ind, latent_z=latent_z))

        return latents

    ##TODO: needs to check
    @torch.no_grad()
    def get_imputed_values(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        normalized: bool = True,
        decode_mode: Optional[int] = None,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return imputed values for all genes for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample for the latent vector.
        normalized
            Return imputed normalized values or not.
        decode_mode
            If a `decode_mode` is given, use the encoder specific to each dataset as usual but use
            the decoder of the dataset of id `decode_mode` to impute values.
        batch_size
            Minibatch size for data loading into model.
        """
        self.module.eval()

        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)

        imputed_values = []
        for mode, scdl in enumerate(scdls):
            imputed_value = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                if normalized:
                    imputed_value.append(
                        self.module.sample_scale(
                            sample_batch,
                            mode,
                            batch_index,
                            label,
                            deterministic=deterministic,
                            decode_mode=decode_mode,
                        )
                    )
                else:
                    imputed_value.append(
                        self.module.sample_rate(
                            sample_batch,
                            mode,
                            batch_index,
                            label,
                            deterministic=deterministic,
                            decode_mode=decode_mode,
                        )
                    )

            imputed_value = torch.cat(imputed_value).cpu().detach().numpy()
            imputed_values.append(imputed_value)

        return imputed_values

    def get_featues_scores_LV(
        self,
        adatas: List[AnnData] = None,
        attribution_layer: str = "pathway", # gene
        deterministic: bool = True,
        batch_size: int = 128,
        n_steps: int = 50,
        output_z_ind: bool= True, 
    ):
        """
        Compute the attribution scores of features to LVs.

        Attributing the changes in each LV (independently) to the change of each feature from its baseline by Integrated Gradient (IG).
        Currenly, only zero baseline are supported. Future version might include median baseline or random baseline.
        Features can be either pathways or genes. LVs can be specifed to be source-specifc or shared.
        Note that LVs may not have the same scales, so merging scores across LVs may be largely biased.

        Parameters
        ----------
        adatas
            list of AnnData, if not specifed, will use adatas in the training
        attribution_layer
            which layer to attributr to, default: pathway, or gene
        deterministic
            whether to to use deterministic z or sampling, default is deterministic
        batch_size
            batch size for the data loader
        n_steps
            number of steps in integreated gradient
        output_z_ind
            wheather to use source-specifc LV or shared LVs, default is True (sorce-specific LVs)
        """
        print(f'Attribution layer: {attribution_layer}\noutput_z_ind: {output_z_ind}')
        self.module.eval()

        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)

        pathway_scores = []
        for mode, scdl in enumerate(scdls):
            pathway_score_dict = {}
            # initiate lists for each LV to store pathway scores
            for key in range(self.n_latent):
                pathway_score_dict[key] = []    
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                
                ## TODO: Consider other baseline,e.g. median, random, in addition to zero baselines
                if attribution_layer == 'pathway':
                    # get the pathway layer for layer attribution,
                    mylayer = self.module.z_encoder.encoders[mode].fc_layers[0][0]
                    lig = LayerIntegratedGradients(self.module.get_latent_representation, mylayer)
                elif attribution_layer == 'gene': 
                    lig = IntegratedGradients(self.module.get_latent_representation)
                for key in range(self.n_latent):
                    pathway_score_dict[key].append(
                        lig.attribute(
                            sample_batch, 
                            additional_forward_args=(mode, deterministic, output_z_ind), 
                            n_steps=n_steps, 
                            target=key,
                        )
                    )              
            for key in pathway_score_dict.keys():         
                pathway_score_dict[key] = torch.cat(pathway_score_dict[key]).cpu().detach().numpy()
            pathway_scores.append(pathway_score_dict)
        return pathway_scores

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            dataset_names = ["seq", "spatial"]
            for i in range(len(self.adatas)):
                save_path = os.path.join(
                    dir_path, "adata_{}.h5ad".format(dataset_names[i])
                )
                self.adatas[i].write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_{}.csv".format(dataset_names[i])
                )

                var_names = self.adatas[i].var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
            # saving pathways
            if self.adata_pathway is not None:
                save_path = os.path.join(
                    dir_path, "adata_pathways.h5ad"
                )
                self.adata_pathway.write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_pathways.csv"
                )
                var_names = self.adata_pathway.var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")

        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)


    @classmethod
    def load(
        cls,
        dir_path: str,
        adata_seq: Optional[AnnData] = None,
        adata_spatial: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Instantiate a model from the saved output.

        Parameters
        ----------
        adata_seq
            AnnData organized in the same way as data used to train model.
            It is not necessary to run :func:`~scvi.data.setup_anndata`,
            as AnnData is validated against the saved `scvi` setup dictionary.
            AnnData must be registered via :func:`~scvi.data.setup_anndata`.
        adata_spatial
            AnnData organized in the same way as data used to train model.
            If None, will check for and load anndata saved with the model.
        dir_path
            Path to saved outputs.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

        Examples
        --------
        >>> vae = GIMVI.load(adata_seq, adata_spatial, save_path)
        >>> vae.get_latent_representation()
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        seq_data_path = os.path.join(dir_path, "adata_seq.h5ad")
        spatial_data_path = os.path.join(dir_path, "adata_spatial.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names_seq.csv")
        spatial_var_names_path = os.path.join(dir_path, "var_names_spatial.csv")
        path_var_names_path = os.path.join(dir_path, "var_names_pathways.csv")

        if adata_seq is None and os.path.exists(seq_data_path):
            adata_seq = read(seq_data_path)
        elif adata_seq is None and not os.path.exists(seq_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        if adata_spatial is None and os.path.exists(spatial_data_path):
            adata_spatial = read(spatial_data_path)
        elif adata_spatial is None and not os.path.exists(spatial_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        if os.path.exists(path_data_path):
            adata_path = read(path_data_path)
        elif not os.path.exists(path_data_path):
            adata_path = None
            print("no pathways saved")

        adatas = [adata_seq, adata_spatial]
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
        spatial_var_names = np.genfromtxt(
            spatial_var_names_path, delimiter=",", dtype=str
        )
        var_names = [seq_var_names, spatial_var_names]
        for i, adata in enumerate(adatas):
            saved_var_names = var_names[i]
            user_var_names = adata.var_names.astype(str)
            if not np.array_equal(saved_var_names, user_var_names):
                warnings.warn(
                    "var_names for adata passed in does not match var_names of "
                    "adata used to train the model. For valid results, the vars "
                    "need to be the same and in the same order as the adata used to train the model."
                )

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        scvi_setup_dicts = attr_dict.pop("scvi_setup_dicts_")
        transfer_anndata_setup(scvi_setup_dicts["seq"], adata_seq)
        transfer_anndata_setup(scvi_setup_dicts["spatial"], adata_spatial)
      
        # get the parameters for the class init signiture
        init_params = attr_dict.pop("init_params_")

        # new saving and loading, enable backwards compatibility
        if "non_kwargs" in init_params.keys():
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = init_params["non_kwargs"]
            kwargs = init_params["kwargs"]

            # expand out kwargs
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        else:
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = {
                k: v for k, v in init_params.items() if not isinstance(v, dict)
            }
            kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        
        # the default init require this way of loading models
        if adata_path is not None:    
            model = cls(adata_seq, adata_spatial, **non_kwargs, adata_pathway=adata_path, **kwargs)
        elif adata_path is None:
            model = cls(adata_seq, adata_spatial, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        _, device = parse_use_gpu_arg(use_gpu)
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.module.eval()
        model.to_device(device)
        return model
    
    def get_loadings(self) -> pd.DataFrame:
        """
        Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        cols_1 = ["Z1_{}".format(i) for i in range(self.n_latent)]
        cols_2 = ["Z2_{}".format(i) for i in range(self.n_latent)]
        cols_s = ["Zshared_{}".format(i) for i in range(self.n_latent)]
        #var_names = _get_var_names_from_setup_anndata(self.adata_path)
        np_loadings_domain1, np_loadings_domain2, np_loadings_domain_shared = self.module.get_loadings()

        loadings_domain1 = pd.DataFrame(
            np_loadings_domain1,index=self.adata_pathway.obs.index, columns=cols_1
        )

        loadings_domain2 = pd.DataFrame(
            np_loadings_domain2,index=self.adata_pathway.obs.index, columns=cols_2
        )

        loadings_domain_shared = pd.DataFrame(
            np_loadings_domain_shared,index=self.adata_pathway.obs.index, columns=cols_s
        )

        return loadings_domain1, loadings_domain2, loadings_domain_shared

class scCLR_mask_decoder(VAEMixin, BaseModelClass):
    """
    scCLR

    Parameters
    ----------
    adata_source1
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains source1 data.
    adata_source2
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains source2 data.
    adata_pathway
        Anndata object AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_source1
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    generative_distributions
        List of generative distribution for adata_seq data and adata_spatial data.
    model_library_size
        List of bool of whether to model library size for adata_seq and adata_spatial.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.scCLR_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> adata_spatial = anndata.read_h5ad(path_to_anndata_spatial)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> scvi.data.setup_anndata(adata_spatial)
    >>> vae = scvi.model.GIMVI(adata_seq, adata_spatial)
    >>> vae.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_spatial: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        generative_distributions: List = ["zinb", "zinb"],
        model_library_size: List = [True, True],
        n_latent: int = 10,
        combine_latent: str = 'cat',
        **model_kwargs,
    ):
        super(scCLR_mask_decoder, self).__init__()
        self.n_latent = n_latent
        self.adatas = [adata_seq, adata_spatial]
        self.scvi_setup_dicts_ = {
            "seq": adata_seq.uns["_scvi"],
            "spatial": adata_spatial.uns["_scvi"],
        }

        seq_var_names = _get_var_names_from_setup_anndata(adata_seq)
        spatial_var_names = _get_var_names_from_setup_anndata(adata_spatial)

        if not set(spatial_var_names) <= set(seq_var_names):
            raise ValueError("source2 input genes needs to be subset of source 1 input genes, note this is only for gene imputation purpose")
 
        if adata_pathway is not None:
            # condition check
            pathway_var_names = _get_var_names_from_setup_anndata(adata_pathway)
            if not set(seq_var_names) == set(pathway_var_names):
                raise ValueError("source 1 input genes needs to be equal to pathway genes")
            # get pathway_gene_loc
            pathway_gene_loc = [np.argwhere(seq_var_names == g)[0] for g in pathway_var_names]
            pathway_gene_loc = np.concatenate(pathway_gene_loc)

        spatial_gene_loc = [
            np.argwhere(seq_var_names == g)[0] for g in spatial_var_names
        ]
        spatial_gene_loc = np.concatenate(spatial_gene_loc)
        gene_mappings = [slice(None), spatial_gene_loc]
        sum_stats = [d.uns["_scvi"]["summary_stats"] for d in self.adatas]
        n_inputs = [s["n_vars"] for s in sum_stats]

        total_genes = adata_seq.uns["_scvi"]["summary_stats"]["n_vars"]

        # since we are combining datasets, we need to increment the batch_idx
        # of one of the datasets
        adata_seq_n_batches = adata_seq.uns["_scvi"]["summary_stats"]["n_batch"]
        adata_spatial.obs["_scvi_batch"] += adata_seq_n_batches

        n_batches = sum([s["n_batch"] for s in sum_stats])
        
        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified input\n")
        else:
            self.mask = None
            print("No pathways, use fully-connected layers\n")
            
        self.module = scCLR_module_mask_decoder(
            n_inputs,
            total_genes,
            gene_mappings,
            generative_distributions,
            model_library_size,
            mask=self.mask,
            n_batch=n_batches,
            n_latent=n_latent,
            **model_kwargs,
        )

        self._model_summary_string = (
            "scCLR Model with the following params: \nn_latent: {}, n_inputs: {}, n_genes: {}, "
            + "n_batch: {}, generative distributions: {}, combine latent space: {}"
        ).format(n_latent, n_inputs, total_genes, n_batches, generative_distributions, combine_latent)
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int = 200,
        use_gpu: Optional[Union[str, int, bool]] = None,
        kappa: int = 5,
        gamma: int = 5,  
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        kappa
            Scaling parameter for the discriminator loss, defaut is 5
        gamma
            Scaling parameter for the classification loss, default is 5
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        plan_kwargs
            Keyword args for model-specific Pytorch Lightning task. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        gpus, device = parse_use_gpu_arg(use_gpu)

        self.trainer = Trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            **kwargs,
        )
        self.train_indices_, self.test_indices_, self.validation_indices_ = [], [], []
        train_dls, test_dls, val_dls = [], [], []
        for i, ad in enumerate(self.adatas):
            ds = DataSplitter(
                ad,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
            ds.setup()
            train_dls.append(ds.train_dataloader())
            test_dls.append(ds.test_dataloader())
            val = ds.val_dataloader()
            val_dls.append(val)
            val.mode = i
            self.train_indices_.append(ds.train_idx)
            self.test_indices_.append(ds.test_idx)
            self.validation_indices_.append(ds.val_idx)
        train_dl = TrainDL(train_dls)

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        self._training_plan = scCLRTrainingPlan(
            self.module,
            source_classifier=True,
            scale_classification_loss=gamma,
            adversarial_classifier=True,
            scale_adversarial_loss=kappa,
            **plan_kwargs,
        )

        if train_size == 1.0:
            # circumvent the empty data loader problem if all dataset used for training
            self.trainer.fit(self._training_plan, train_dl)
        else:
            # accepts list of val dataloaders
            self.trainer.fit(self._training_plan, train_dl, val_dls)
        try:
            self.history_ = self.trainer.logger.history
        except AttributeError:
            self.history_ = None
        self.module.eval()

        self.to_device(device)
        self.is_trained_ = True


    def _make_scvi_dls(self, adatas: List[AnnData] = None, batch_size=128):
        if adatas is None:
            adatas = self.adatas
        post_list = [self._make_data_loader(ad) for ad in adatas]
        for i, dl in enumerate(post_list):
            dl.mode = i

        return post_list


    @torch.no_grad()
    def get_parameters_z_shared(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)
        self.module.eval()
        latent_shared_parameters = []
        
        for mode, scdl in enumerate(scdls):
            qz_m = []
            qz_v = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                z_dict  = self.module.get_latent_parameter_z_shared(sample_batch, mode)
                qz_m.append(z_dict["qz_m"])
                qz_v.append(z_dict["qz_v"])                

            latent_m = torch.cat(qz_m).cpu().detach().numpy()
            latent_v = torch.cat(qz_v).cpu().detach().numpy()
            
            latent_shared_parameters.append(dict(latent_m=latent_m, latent_v=latent_v))

        return latent_shared_parameters

    @torch.no_grad()
    def get_latent_representation(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)
        self.module.eval()
        latents = []
        
        for mode, scdl in enumerate(scdls):
            latent_z_ind = []
            latent_z = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                z_dict  = self.module.sample_from_posterior_z(sample_batch, mode, deterministic=deterministic)
                latent_z_ind.append(z_dict["z_ind"])
                latent_z.append(z_dict["z"])                

            latent_z = torch.cat(latent_z).cpu().detach().numpy()
            latent_z_ind = torch.cat(latent_z_ind).cpu().detach().numpy()
            
            latents.append(dict(latent_z_ind=latent_z_ind, latent_z=latent_z))

        return latents

    ##TODO: needs to check
    @torch.no_grad()
    def get_imputed_values(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        normalized: bool = True,
        decode_mode: Optional[int] = None,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return imputed values for all genes for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample for the latent vector.
        normalized
            Return imputed normalized values or not.
        decode_mode
            If a `decode_mode` is given, use the encoder specific to each dataset as usual but use
            the decoder of the dataset of id `decode_mode` to impute values.
        batch_size
            Minibatch size for data loading into model.
        """
        self.module.eval()

        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)

        imputed_values = []
        for mode, scdl in enumerate(scdls):
            imputed_value = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                if normalized:
                    imputed_value.append(
                        self.module.sample_scale(
                            sample_batch,
                            mode,
                            batch_index,
                            label,
                            deterministic=deterministic,
                            decode_mode=decode_mode,
                        )
                    )
                else:
                    imputed_value.append(
                        self.module.sample_rate(
                            sample_batch,
                            mode,
                            batch_index,
                            label,
                            deterministic=deterministic,
                            decode_mode=decode_mode,
                        )
                    )

            imputed_value = torch.cat(imputed_value).cpu().detach().numpy()
            imputed_values.append(imputed_value)

        return imputed_values

    def get_featues_scores_LV(
        self,
        adatas: List[AnnData] = None,
        attribution_layer: str = "pathway", # gene
        deterministic: bool = True,
        batch_size: int = 128,
        n_steps: int = 50,
        output_z_ind: bool= True, 
    ):
        """
        Compute the attribution scores of features to LVs.

        Attributing the changes in each LV (independently) to the change of each feature from its baseline by Integrated Gradient (IG).
        Currenly, only zero baseline are supported. Future version might include median baseline or random baseline.
        Features can be either pathways or genes. LVs can be specifed to be source-specifc or shared.
        Note that LVs may not have the same scales, so merging scores across LVs may be largely biased.

        Parameters
        ----------
        adatas
            list of AnnData, if not specifed, will use adatas in the training
        attribution_layer
            which layer to attributr to, default: pathway, or gene
        deterministic
            whether to to use deterministic z or sampling, default is deterministic
        batch_size
            batch size for the data loader
        n_steps
            number of steps in integreated gradient
        output_z_ind
            wheather to use source-specifc LV or shared LVs, default is True (sorce-specific LVs)
        """
        print(f'Attribution layer: {attribution_layer}\noutput_z_ind: {output_z_ind}')
        self.module.eval()

        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)

        pathway_scores = []
        for mode, scdl in enumerate(scdls):
            pathway_score_dict = {}
            # initiate lists for each LV to store pathway scores
            for key in range(self.n_latent):
                pathway_score_dict[key] = []    
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                
                ## TODO: Consider other baseline,e.g. median, random, in addition to zero baselines
                if attribution_layer == 'pathway':
                    # get the pathway layer for layer attribution,
                    mylayer = self.module.z_encoder.encoders[mode].fc_layers[0][0]
                    lig = LayerIntegratedGradients(self.module.get_latent_representation, mylayer)
                elif attribution_layer == 'gene': 
                    lig = IntegratedGradients(self.module.get_latent_representation)
                for key in range(self.n_latent):
                    pathway_score_dict[key].append(
                        lig.attribute(
                            sample_batch, 
                            additional_forward_args=(mode, deterministic, output_z_ind), 
                            n_steps=n_steps, 
                            target=key,
                        )
                    )              
            for key in pathway_score_dict.keys():         
                pathway_score_dict[key] = torch.cat(pathway_score_dict[key]).cpu().detach().numpy()
            pathway_scores.append(pathway_score_dict)
        return pathway_scores

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            dataset_names = ["seq", "spatial"]
            for i in range(len(self.adatas)):
                save_path = os.path.join(
                    dir_path, "adata_{}.h5ad".format(dataset_names[i])
                )
                self.adatas[i].write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_{}.csv".format(dataset_names[i])
                )

                var_names = self.adatas[i].var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
            # saving pathways
            if self.adata_pathway is not None:
                save_path = os.path.join(
                    dir_path, "adata_pathways.h5ad"
                )
                self.adata_pathway.write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_pathways.csv"
                )
                var_names = self.adata_pathway.var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")

        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)


    @classmethod
    def load(
        cls,
        dir_path: str,
        adata_seq: Optional[AnnData] = None,
        adata_spatial: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Instantiate a model from the saved output.

        Parameters
        ----------
        adata_seq
            AnnData organized in the same way as data used to train model.
            It is not necessary to run :func:`~scvi.data.setup_anndata`,
            as AnnData is validated against the saved `scvi` setup dictionary.
            AnnData must be registered via :func:`~scvi.data.setup_anndata`.
        adata_spatial
            AnnData organized in the same way as data used to train model.
            If None, will check for and load anndata saved with the model.
        dir_path
            Path to saved outputs.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

        Examples
        --------
        >>> vae = GIMVI.load(adata_seq, adata_spatial, save_path)
        >>> vae.get_latent_representation()
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        seq_data_path = os.path.join(dir_path, "adata_seq.h5ad")
        spatial_data_path = os.path.join(dir_path, "adata_spatial.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names_seq.csv")
        spatial_var_names_path = os.path.join(dir_path, "var_names_spatial.csv")
        path_var_names_path = os.path.join(dir_path, "var_names_pathways.csv")

        if adata_seq is None and os.path.exists(seq_data_path):
            adata_seq = read(seq_data_path)
        elif adata_seq is None and not os.path.exists(seq_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        if adata_spatial is None and os.path.exists(spatial_data_path):
            adata_spatial = read(spatial_data_path)
        elif adata_spatial is None and not os.path.exists(spatial_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        if os.path.exists(path_data_path):
            adata_path = read(path_data_path)
        elif not os.path.exists(path_data_path):
            adata_path = None
            print("no pathways saved")

        adatas = [adata_seq, adata_spatial]
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
        spatial_var_names = np.genfromtxt(
            spatial_var_names_path, delimiter=",", dtype=str
        )
        var_names = [seq_var_names, spatial_var_names]
        for i, adata in enumerate(adatas):
            saved_var_names = var_names[i]
            user_var_names = adata.var_names.astype(str)
            if not np.array_equal(saved_var_names, user_var_names):
                warnings.warn(
                    "var_names for adata passed in does not match var_names of "
                    "adata used to train the model. For valid results, the vars "
                    "need to be the same and in the same order as the adata used to train the model."
                )

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        scvi_setup_dicts = attr_dict.pop("scvi_setup_dicts_")
        transfer_anndata_setup(scvi_setup_dicts["seq"], adata_seq)
        transfer_anndata_setup(scvi_setup_dicts["spatial"], adata_spatial)
      
        # get the parameters for the class init signiture
        init_params = attr_dict.pop("init_params_")

        # new saving and loading, enable backwards compatibility
        if "non_kwargs" in init_params.keys():
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = init_params["non_kwargs"]
            kwargs = init_params["kwargs"]

            # expand out kwargs
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        else:
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = {
                k: v for k, v in init_params.items() if not isinstance(v, dict)
            }
            kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        
        # the default init require this way of loading models
        if adata_path is not None:    
            model = cls(adata_seq, adata_spatial, **non_kwargs, adata_pathway=adata_path, **kwargs)
        elif adata_path is None:
            model = cls(adata_seq, adata_spatial, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        _, device = parse_use_gpu_arg(use_gpu)
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.module.eval()
        model.to_device(device)
        return model

class scCLR(VAEMixin, BaseModelClass):
    """
    scCLR

    Parameters
    ----------
    adata_source1
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains source1 data.
    adata_source2
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains source2 data.
    adata_pathway
        Anndata object AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_source1
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    generative_distributions
        List of generative distribution for adata_seq data and adata_spatial data.
    model_library_size
        List of bool of whether to model library size for adata_seq and adata_spatial.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.scCLR_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> adata_spatial = anndata.read_h5ad(path_to_anndata_spatial)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> scvi.data.setup_anndata(adata_spatial)
    >>> vae = scvi.model.GIMVI(adata_seq, adata_spatial)
    >>> vae.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_spatial: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        generative_distributions: List = ["zinb", "zinb"],
        model_library_size: List = [True, True],
        n_latent: int = 10,
        combine_latent: str = 'cat',
        **model_kwargs,
    ):
        super(scCLR, self).__init__()
        self.n_latent = n_latent
        self.adatas = [adata_seq, adata_spatial]
        self.scvi_setup_dicts_ = {
            "seq": adata_seq.uns["_scvi"],
            "spatial": adata_spatial.uns["_scvi"],
        }

        seq_var_names = _get_var_names_from_setup_anndata(adata_seq)
        spatial_var_names = _get_var_names_from_setup_anndata(adata_spatial)

        if not set(spatial_var_names) <= set(seq_var_names):
            raise ValueError("source2 input genes needs to be subset of source 1 input genes, note this is only for gene imputation purpose")
 
        if adata_pathway is not None:
            # condition check
            pathway_var_names = _get_var_names_from_setup_anndata(adata_pathway)
            if not set(seq_var_names) == set(pathway_var_names):
                raise ValueError("source 1 input genes needs to be equal to pathway genes")
            # get pathway_gene_loc
            pathway_gene_loc = [np.argwhere(seq_var_names == g)[0] for g in pathway_var_names]
            pathway_gene_loc = np.concatenate(pathway_gene_loc)

        spatial_gene_loc = [
            np.argwhere(seq_var_names == g)[0] for g in spatial_var_names
        ]
        spatial_gene_loc = np.concatenate(spatial_gene_loc)
        gene_mappings = [slice(None), spatial_gene_loc]
        sum_stats = [d.uns["_scvi"]["summary_stats"] for d in self.adatas]
        n_inputs = [s["n_vars"] for s in sum_stats]

        total_genes = adata_seq.uns["_scvi"]["summary_stats"]["n_vars"]

        # since we are combining datasets, we need to increment the batch_idx
        # of one of the datasets
        adata_seq_n_batches = adata_seq.uns["_scvi"]["summary_stats"]["n_batch"]
        adata_spatial.obs["_scvi_batch"] += adata_seq_n_batches

        n_batches = sum([s["n_batch"] for s in sum_stats])
        
        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified input\n")
        else:
            self.mask = None
            print("No pathways, use fully-connected layers\n")
            
        self.module = scCLR(
            n_inputs,
            total_genes,
            gene_mappings,
            generative_distributions,
            model_library_size,
            mask=self.mask,
            n_batch=n_batches,
            n_latent=n_latent,
            combine_latent=combine_latent,
            **model_kwargs,
        )

        self._model_summary_string = (
            "scCLR Model with the following params: \nn_latent: {}, n_inputs: {}, n_genes: {}, "
            + "n_batch: {}, generative distributions: {}, combine latent space: {}"
        ).format(n_latent, n_inputs, total_genes, n_batches, generative_distributions, combine_latent)
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int = 200,
        use_gpu: Optional[Union[str, int, bool]] = None,
        kappa: int = 5,
        gamma: int = 5,  
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        kappa
            Scaling parameter for the discriminator loss, defaut is 5
        gamma
            Scaling parameter for the classification loss, default is 5
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        plan_kwargs
            Keyword args for model-specific Pytorch Lightning task. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        gpus, device = parse_use_gpu_arg(use_gpu)

        self.trainer = Trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            **kwargs,
        )
        self.train_indices_, self.test_indices_, self.validation_indices_ = [], [], []
        train_dls, test_dls, val_dls = [], [], []
        for i, ad in enumerate(self.adatas):
            ds = DataSplitter(
                ad,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
            ds.setup()
            train_dls.append(ds.train_dataloader())
            test_dls.append(ds.test_dataloader())
            val = ds.val_dataloader()
            val_dls.append(val)
            val.mode = i
            self.train_indices_.append(ds.train_idx)
            self.test_indices_.append(ds.test_idx)
            self.validation_indices_.append(ds.val_idx)
        train_dl = TrainDL(train_dls)

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        self._training_plan = scCLRTrainingPlan(
            self.module,
            source_classifier=True,
            scale_classification_loss=gamma,
            adversarial_classifier=True,
            scale_adversarial_loss=kappa,
            **plan_kwargs,
        )

        if train_size == 1.0:
            # circumvent the empty data loader problem if all dataset used for training
            self.trainer.fit(self._training_plan, train_dl)
        else:
            # accepts list of val dataloaders
            self.trainer.fit(self._training_plan, train_dl, val_dls)
        try:
            self.history_ = self.trainer.logger.history
        except AttributeError:
            self.history_ = None
        self.module.eval()

        self.to_device(device)
        self.is_trained_ = True


    def _make_scvi_dls(self, adatas: List[AnnData] = None, batch_size=128):
        if adatas is None:
            adatas = self.adatas
        post_list = [self._make_data_loader(ad) for ad in adatas]
        for i, dl in enumerate(post_list):
            dl.mode = i

        return post_list


    @torch.no_grad()
    def get_parameters_z_shared(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)
        self.module.eval()
        latent_shared_parameters = []
        
        for mode, scdl in enumerate(scdls):
            qz_m = []
            qz_v = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                z_dict  = self.module.get_latent_parameter_z_shared(sample_batch, mode)
                qz_m.append(z_dict["qz_m"])
                qz_v.append(z_dict["qz_v"])                

            latent_m = torch.cat(qz_m).cpu().detach().numpy()
            latent_v = torch.cat(qz_v).cpu().detach().numpy()
            
            latent_shared_parameters.append(dict(latent_m=latent_m, latent_v=latent_v))

        return latent_shared_parameters

    @torch.no_grad()
    def get_latent_representation(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 128,
        output_z_raw: bool = False,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)
        self.module.eval()
        latents = []
        
        for mode, scdl in enumerate(scdls):
            latent_z_ind = []
            latent_z = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                z_dict  = self.module.sample_from_posterior_z(sample_batch, mode, deterministic=deterministic, output_z_raw= output_z_raw)
                latent_z_ind.append(z_dict["z_ind"])
                latent_z.append(z_dict["z"])                

            latent_z = torch.cat(latent_z).cpu().detach().numpy()
            latent_z_ind = torch.cat(latent_z_ind).cpu().detach().numpy()
            
            latents.append(dict(latent_z_ind=latent_z_ind, latent_z=latent_z))

        return latents

    ##TODO: needs to check
    @torch.no_grad()
    def get_imputed_values(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        normalized: bool = True,
        decode_mode: Optional[int] = None,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return imputed values for all genes for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample for the latent vector.
        normalized
            Return imputed normalized values or not.
        decode_mode
            If a `decode_mode` is given, use the encoder specific to each dataset as usual but use
            the decoder of the dataset of id `decode_mode` to impute values.
        batch_size
            Minibatch size for data loading into model.
        """
        self.module.eval()

        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)

        imputed_values = []
        for mode, scdl in enumerate(scdls):
            imputed_value = []
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                if normalized:
                    imputed_value.append(
                        self.module.sample_scale(
                            sample_batch,
                            mode,
                            batch_index,
                            label,
                            deterministic=deterministic,
                            decode_mode=decode_mode,
                        )
                    )
                else:
                    imputed_value.append(
                        self.module.sample_rate(
                            sample_batch,
                            mode,
                            batch_index,
                            label,
                            deterministic=deterministic,
                            decode_mode=decode_mode,
                        )
                    )

            imputed_value = torch.cat(imputed_value).cpu().detach().numpy()
            imputed_values.append(imputed_value)

        return imputed_values

    def get_featues_scores_LV(
        self,
        adatas: List[AnnData] = None,
        attribution_layer: str = "pathway", # gene
        deterministic: bool = True,
        batch_size: int = 128,
        n_steps: int = 50,
        output_z_ind: bool= True, 
    ):
        """
        Compute the attribution scores of features to LVs.

        Attributing the changes in each LV (independently) to the change of each feature from its baseline by Integrated Gradient (IG).
        Currenly, only zero baseline are supported. Future version might include median baseline or random baseline.
        Features can be either pathways or genes. LVs can be specifed to be source-specifc or shared.
        Note that LVs may not have the same scales, so merging scores across LVs may be largely biased.

        Parameters
        ----------
        adatas
            list of AnnData, if not specifed, will use adatas in the training
        attribution_layer
            which layer to attributr to, default: pathway, or gene
        deterministic
            whether to to use deterministic z or sampling, default is deterministic
        batch_size
            batch size for the data loader
        n_steps
            number of steps in integreated gradient
        output_z_ind
            wheather to use source-specifc LV or shared LVs, default is True (sorce-specific LVs)
        """
        print(f'Attribution layer: {attribution_layer}\noutput_z_ind: {output_z_ind}')
        self.module.eval()

        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)

        pathway_scores = []
        for mode, scdl in enumerate(scdls):
            pathway_score_dict = {}
            # initiate lists for each LV to store pathway scores
            for key in range(self.n_latent):
                pathway_score_dict[key] = []    
            for tensors in scdl:
                (
                    sample_batch,
                    local_l_mean,
                    local_l_var,
                    batch_index,
                    label,
                    *_,
                ) = _unpack_tensors(tensors)
                
                ## TODO: Consider other baseline,e.g. median, random, in addition to zero baselines
                if attribution_layer == 'pathway':
                    # get the pathway layer for layer attribution,
                    mylayer = self.module.z_encoder.encoders[mode].fc_layers[0][0]
                    lig = LayerIntegratedGradients(self.module.get_latent_representation, mylayer)
                elif attribution_layer == 'gene': 
                    lig = IntegratedGradients(self.module.get_latent_representation)
                for key in range(self.n_latent):
                    pathway_score_dict[key].append(
                        lig.attribute(
                            sample_batch, 
                            additional_forward_args=(mode, deterministic, output_z_ind), 
                            n_steps=n_steps, 
                            target=key,
                        )
                    )              
            for key in pathway_score_dict.keys():         
                pathway_score_dict[key] = torch.cat(pathway_score_dict[key]).cpu().detach().numpy()
            pathway_scores.append(pathway_score_dict)
        return pathway_scores

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            dataset_names = ["seq", "spatial"]
            for i in range(len(self.adatas)):
                save_path = os.path.join(
                    dir_path, "adata_{}.h5ad".format(dataset_names[i])
                )
                self.adatas[i].write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_{}.csv".format(dataset_names[i])
                )

                var_names = self.adatas[i].var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
            # saving pathways
            if self.adata_pathway is not None:
                save_path = os.path.join(
                    dir_path, "adata_pathways.h5ad"
                )
                self.adata_pathway.write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_pathways.csv"
                )
                var_names = self.adata_pathway.var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")

        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)


    @classmethod
    def load(
        cls,
        dir_path: str,
        adata_seq: Optional[AnnData] = None,
        adata_spatial: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Instantiate a model from the saved output.

        Parameters
        ----------
        adata_seq
            AnnData organized in the same way as data used to train model.
            It is not necessary to run :func:`~scvi.data.setup_anndata`,
            as AnnData is validated against the saved `scvi` setup dictionary.
            AnnData must be registered via :func:`~scvi.data.setup_anndata`.
        adata_spatial
            AnnData organized in the same way as data used to train model.
            If None, will check for and load anndata saved with the model.
        dir_path
            Path to saved outputs.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

        Examples
        --------
        >>> vae = GIMVI.load(adata_seq, adata_spatial, save_path)
        >>> vae.get_latent_representation()
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        seq_data_path = os.path.join(dir_path, "adata_seq.h5ad")
        spatial_data_path = os.path.join(dir_path, "adata_spatial.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names_seq.csv")
        spatial_var_names_path = os.path.join(dir_path, "var_names_spatial.csv")
        path_var_names_path = os.path.join(dir_path, "var_names_pathways.csv")

        if adata_seq is None and os.path.exists(seq_data_path):
            adata_seq = read(seq_data_path)
        elif adata_seq is None and not os.path.exists(seq_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        if adata_spatial is None and os.path.exists(spatial_data_path):
            adata_spatial = read(spatial_data_path)
        elif adata_spatial is None and not os.path.exists(spatial_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        if os.path.exists(path_data_path):
            adata_path = read(path_data_path)
        elif not os.path.exists(path_data_path):
            adata_path = None
            print("no pathways saved")

        adatas = [adata_seq, adata_spatial]
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
        spatial_var_names = np.genfromtxt(
            spatial_var_names_path, delimiter=",", dtype=str
        )
        var_names = [seq_var_names, spatial_var_names]
        for i, adata in enumerate(adatas):
            saved_var_names = var_names[i]
            user_var_names = adata.var_names.astype(str)
            if not np.array_equal(saved_var_names, user_var_names):
                warnings.warn(
                    "var_names for adata passed in does not match var_names of "
                    "adata used to train the model. For valid results, the vars "
                    "need to be the same and in the same order as the adata used to train the model."
                )

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        scvi_setup_dicts = attr_dict.pop("scvi_setup_dicts_")
        transfer_anndata_setup(scvi_setup_dicts["seq"], adata_seq)
        transfer_anndata_setup(scvi_setup_dicts["spatial"], adata_spatial)
      
        # get the parameters for the class init signiture
        init_params = attr_dict.pop("init_params_")

        # new saving and loading, enable backwards compatibility
        if "non_kwargs" in init_params.keys():
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = init_params["non_kwargs"]
            kwargs = init_params["kwargs"]

            # expand out kwargs
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        else:
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = {
                k: v for k, v in init_params.items() if not isinstance(v, dict)
            }
            kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        
        # the default init require this way of loading models
        if adata_path is not None:    
            model = cls(adata_seq, adata_spatial, **non_kwargs, adata_pathway=adata_path, **kwargs)
        elif adata_path is None:
            model = cls(adata_seq, adata_spatial, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        _, device = parse_use_gpu_arg(use_gpu)
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.module.eval()
        model.to_device(device)
        return model

class scCLR_Res(RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    mask 
        User-specifed binary tensor for pathways, [n_pathway, n_genes]
    adata_pathway
        Pathways registered in adata object, will exlucde pathways with too few genes.     
    n_latent
        Dimensionality of the latent space.
    n_layers_FC
        Numer of fully connected layers in Masked Encoder
    n_layers_skip 
        Number of ResNet layer in skip connection Encoder   
    n_layers_decoder
        Number of hidden layers decoder NNs.    
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
    """

    def __init__(
        self,
        adata: AnnData,
        mask: torch.Tensor = None,
        adata_pathway: AnnData = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_FC: int = 1, 
        n_layers_skip: int = 1, 
        n_layers_decoder: int =2,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super(scCLR_Res, self).__init__(adata)

        n_cats_per_cov = (
            self.scvi_setup_dict_["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in self.scvi_setup_dict_
            else None
        )

        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified tensor\n")
        else:
            raise ValueError("Either mask or adata_pathway must be supplied")
            
            
        self.module = scCLR_Res_module(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_labels=self.summary_stats["n_labels"],
            mask = self.mask, 
            n_continuous_cov=self.summary_stats["n_continuous_covs"],
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_FC=n_layers_FC,
            n_layers_skip=n_layers_skip,
            n_layers_decoder=n_layers_decoder,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self._model_summary_string = (
            "scCLR_Res Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers_FC: {}, n_layers_skip: {}, n_layers_decoder: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers_FC,
            n_layers_skip,
            n_layers_decoder,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

class scCLR_phase2(RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Parameters
    ----------
    ##TODO
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    mask 
        User-specifed binary tensor for pathways, [n_pathway, n_genes]
    adata_pathway
        Pathways registered in adata object, will exlucde pathways with too few genes.     
    n_latent
        Dimensionality of the latent space.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int,
        mask: torch.Tensor = None,
        adata_pathway: AnnData = None,
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
        model_library_size: bool = False,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super(scCLR_phase2, self).__init__(adata)

        n_cats_per_cov = (
            self.scvi_setup_dict_["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in self.scvi_setup_dict_
            else None
        )

        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified tensor\n")
        else:
            raise ValueError("Either mask or adata_pathway must be supplied")
            
            
        self.module = scCLR_phase2_module(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_labels=self.summary_stats["n_labels"],
            mask = self.mask, 
            n_latent = n_latent,
            model_library_bool = model_library_size,
            n_continuous_cov=self.summary_stats["n_continuous_covs"],
            n_cats_per_cov=n_cats_per_cov,
            n_hidden_l_encoder = n_hidden_l_encoder,
            n_layers_l_decoder = n_layers_l_decoder,
            n_layers_masked = n_layers_masked,
            n_hidden_masked = n_hidden_masked,
            n_out_masked = n_out_masked,
            n_layers_FC = n_layers_FC,
            n_layers_decoder = n_layers_decoder,
            n_hidden_FC = n_hidden_FC,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self._model_summary_string = (
            "scCLR_Res Model with the following params: \n, n_latent: {}, "
            "n_hidden_l_encoder: {}, n_layers_l_decoder: {}, n_layers_masked: {}, "
            "n_hidden_masked: {}, n_out_masked: {}, n_layers_FC: {}, n_layers_decoder: {}, n_hidden_FC: {}, "
            "dropout_rate: {}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_latent,
            n_hidden_l_encoder,
            n_layers_l_decoder,
            n_layers_masked,
            n_hidden_masked,
            n_out_masked,
            n_layers_FC,
            n_layers_decoder,
            n_hidden_FC,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = Phase2ModelTrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

class TrainDL(DataLoader):
    def __init__(self, data_loader_list, **kwargs):
        self.data_loader_list = data_loader_list
        self.largest_train_dl_idx = np.argmax(
            [len(dl.indices) for dl in data_loader_list]
        )
        self.largest_dl = self.data_loader_list[self.largest_train_dl_idx]
        super().__init__(self.largest_dl, **kwargs)

    def __len__(self):
        return len(self.largest_dl)

    def __iter__(self):
        train_dls = [
            dl if i == self.largest_train_dl_idx else cycle(dl)
            for i, dl in enumerate(self.data_loader_list)
        ]
        return zip(*train_dls)