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
from scvi.model.base import RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass
from scvi.train import Trainer, AdversarialTrainingPlan, TrainRunner
from task.TotalDeltaETM_TrainingPlan import TrainingPlan

from module.DeltaETM_module import DeltaETM_module, TotalDeltaETM_module
from task.DeltaETM_TrainingPlan import DeltaETMTrainingPlan
from scvi.train import TrainRunner

from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients, LayerDeepLift, LayerDeepLiftShap
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

import wandb
# start a new experiment
wandb.init(project="TotalDeltaETM")

logger = logging.getLogger(__name__)

def _unpack_tensors(tensors):
    x = tensors[_CONSTANTS.X_KEY].squeeze_(0)
    unspliced = tensors[_CONSTANTS.PROTEIN_EXP_KEY].squeeze_(0)
    local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY].squeeze_(0)
    local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY].squeeze_(0)
    batch_index = tensors[_CONSTANTS.BATCH_KEY].squeeze_(0)
    y = tensors[_CONSTANTS.LABELS_KEY].squeeze_(0)
    return x, unspliced, local_l_mean, local_l_var, batch_index, y

def set_up_adata_pathway(adata_pathway, minGenes=10):
    """
    remove pathways with less than "minGenes" genes
    """
    # remove bad pathways
    mask = adata_pathway.X.copy()
    pathway_size = np.sum(mask,1)
    bad_pathway = pathway_size < minGenes
    mask[bad_pathway,:] = 0
    adata_pathway.X = mask
    print(f'Removing {np.sum(bad_pathway)} pathways less than {minGenes} genes\n')
    return(adata_pathway)

class DeltaETM(VAEMixin, BaseModelClass):
    """
    DeltaETM

    Parameters
    ----------
    adata_seq
        Spliced count AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains source1 data.
    adata_spatial
        Unscplied count AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains source2 data.
    adata_pathway
        Pathway AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_source1
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.DeltaETM_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> adata_spatial = anndata.read_h5ad(path_to_anndata_spatial)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> scvi.data.setup_anndata(adata_spatial)
    >>> DeltaETM = DeltaEMT(adata_seq, adata_spatial, adata_pathway)
    >>> DeltaETM.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_spatial: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        n_latent: int = 4,
        combine_latent: str = 'cat',
        **model_kwargs,
    ):
        super(DeltaETM, self).__init__()
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
            
        self.module = DeltaETM_module(
            n_inputs,
            total_genes,
            mask=self.mask,
            n_batch=n_batches,
            n_latent=n_latent,
            **model_kwargs,
        )

        self._model_summary_string = (
            "DelETM Model with the following params: \nn_latent: {}, n_inputs: {}, n_genes: {}, "
            + "n_batch: {}"
        ).format(n_latent, n_inputs, total_genes, n_batches)
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int = 500,
        use_gpu: Optional[Union[str, int, bool]] = None,
        kappa: int = 5,
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
        self._training_plan = DeltaETMTrainingPlan(
            self.module,
            adversarial_classifier=True,
            scale_adversarial_loss=kappa,
            **plan_kwargs,
        )

        wandb.watch(self.module.decoder, log_freq=10, log="all")
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
    def get_latent_representation(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 128,
        output_softmax_z: bool = True,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        output_softmax_z
            If true, return the softmax of the latent space embedding.
        """
        if adatas is None:
            adatas = self.adatas
        scdls = self._make_scvi_dls(adatas, batch_size=batch_size)
        self.module.eval()
        latents = []
        
        for mode, scdl in enumerate(scdls):
            latent_z = []
            for tensors in scdl:
                (
                    sample_batch,
                    *_,
                ) = _unpack_tensors(tensors)
                z_dict  = self.module.sample_from_posterior_z(sample_batch, mode, deterministic=deterministic, output_softmax_z = output_softmax_z)
                latent_z.append(z_dict["z"])                

            latent_z = torch.cat(latent_z).cpu().detach().numpy()
            latents.append(dict(latent_z=latent_z))
        print(f'Deterministic: {deterministic}\nOutput_softmax_z: {output_softmax_z}')
        return latents

    def get_featues_scores_LV(
        self,
        adatas: List[AnnData] = None,
        attribution_layer: str = "pathway", # gene
        deterministic: bool = True,
        batch_size: int = 128,
        n_steps: int = 50,
        output_softmax_z: bool = True,
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
        output_softmax_z
            wheather to output the softmax of the latent space embedding
        """
        print(f'Attribution layer: {attribution_layer}, output_softmax_z: {output_softmax_z}')
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
                            additional_forward_args=(mode, deterministic, output_softmax_z), 
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
    
    def get_delta(self) -> pd.DataFrame:
        """
        get_delta
        """
        cols = ["topic_{}".format(i) for i in range(self.n_latent)]
        #var_names = _get_var_names_from_setup_anndata(self.adata_path)
        np_loadings = self.module.get_loadings()

        loadings = pd.DataFrame(
            np_loadings,index=self.adata_pathway.obs.index, columns=cols
        )

        return loadings

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
    
class TotalDeltaETM(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    """
    DeltaETM

    Parameters
    ----------
    adata_seq
        Spliced and unspliced count AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains data.
    adata_pathway
        Pathway AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_source1
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.DeltaETM_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> model = TotalDeltaEMT(adata_seq adata_pathway)
    >>> model.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        n_latent: int = 4,
        combine_latent: str = 'concat',
        **model_kwargs,
    ):
        super(TotalDeltaETM, self).__init__()
        self.n_latent = n_latent
        self.adata = adata_seq
        self.scvi_setup_dicts_ = self.adata.uns["_scvi"]
        if adata_pathway is not None:
            # condition check
            var_names = _get_var_names_from_setup_anndata(adata_seq)
            pathway_var_names = _get_var_names_from_setup_anndata(adata_pathway)
            if not set(var_names) == set(pathway_var_names):
                raise ValueError("expression genes needs to be equal to pathway genes")
        
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
        
        self.summary_stats = adata_seq.uns['_scvi']["summary_stats"]    
        self.module = TotalDeltaETM_module(
            dim_input_list = [self.summary_stats["n_vars"],self.summary_stats["n_vars"]],
            total_genes = self.summary_stats["n_vars"],
            mask=self.mask,
            n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            combine_method = combine_latent,
            **model_kwargs,
        )
        
        self._model_summary_string = (
            "TotalDelETM Model with the following params: \nn_latent: {},  n_genes: {}, "
            + "n_batch: {}, combine_latent: {}"
        ).format(n_latent, self.summary_stats["n_vars"], self.summary_stats["n_batch"], combine_latent)
        self.init_params_ = self._get_init_params(locals())    
        wandb.watch(self.module.decoder, log_freq=10, log="all")
    def train(
        self,
        max_epochs: Optional[int] = 400,
        lr: float = 4e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        reduce_lr_on_plateau: bool = True,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        #adversarial_classifier: Optional[bool] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
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
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        '''if adversarial_classifier is None:
            imputation = (
                True if "totalvi_batch_mask" in self.scvi_setup_dict_.keys() else False
            )
            adversarial_classifier = True if imputation else False
        '''
        #adversarial_classifier = False    
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1

        update_dict = {
            "lr": lr,
            #"adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            #"check_val_every_n_epoch": check_val_every_n_epoch,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

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
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            **kwargs,
        )
        return runner()
    
    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        batch_size: int = 128,
        output_softmax_z: bool = True,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        output_softmax_z
            If true, return the softmax of the latent space embedding.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch,
                sample_batch_unspliced,
                *_,
            ) = _unpack_tensors(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, sample_batch_unspliced, deterministic=deterministic, output_softmax_z = output_softmax_z)
            latent_z.append(z_dict["z"])                

        latent_z = torch.cat(latent_z).cpu().detach().numpy()
        
        print(f'Deterministic: {deterministic}\nOutput_softmax_z: {output_softmax_z}')
        return latent_z
    
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
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)
            varnames_save_path = os.path.join(
                dir_path, "var_names.csv"
            )

            var_names = self.adata.var_names.astype(str)
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
        seq_data_path = os.path.join(dir_path, "adata.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names.csv")
        path_var_names_path = os.path.join(dir_path, "var_names_pathways.csv")

        if adata_seq is None and os.path.exists(seq_data_path):
            adata_seq = read(seq_data_path)
        elif adata_seq is None and not os.path.exists(seq_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        
        if os.path.exists(path_data_path):
            adata_path = read(path_data_path)
        elif not os.path.exists(path_data_path):
            adata_path = None
            print("no pathways saved")

        #adatas = [adata_seq, adata_spatial]
        adata = adata_seq
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
        #spatial_var_names = np.genfromtxt(
        #    spatial_var_names_path, delimiter=",", dtype=str
        #)
        var_names = seq_var_names
        #for i, adata in enumerate(adatas):
        saved_var_names = var_names
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
        transfer_anndata_setup(scvi_setup_dicts, adata_seq)
      
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
            model = cls(adata_seq, **non_kwargs, adata_pathway=adata_path, **kwargs)
        elif adata_path is None:
            model = cls(adata_seq, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        _, device = parse_use_gpu_arg(use_gpu)
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.module.eval()
        model.to_device(device)
        return model
