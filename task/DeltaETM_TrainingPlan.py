import torch

from scvi import _CONSTANTS
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass
from scvi.train import AdversarialTrainingPlan, TrainingPlan
from typing import Callable, Optional, Union
from scvi._compat import Literal

# custome training plan to log extra infomation
class Phase2ModelTrainingPlan(TrainingPlan):
    """
    Train phase 2 model with custome logging.
    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    lr
        Learning rate used for optimization :class:`~torch.optim.Adam`.
    weight_decay
        Weight decay used in :class:`~torch.optim.Adam`.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss
        # pytorch lightning automatically backprops on "loss"
        self.log("train_loss", scvi_loss.loss, on_epoch=True)
        return {
            "loss": scvi_loss.loss,
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            "kl_global": scvi_loss.kl_global,
            "n_obs": reconstruction_loss.shape[0],
            "kl_divergence_z_s": scvi_loss.kl_divergence_z_s.sum(),
        }

    def training_epoch_end(self, outputs):
        n_obs, elbo, rec_loss, kl_local, kl_divergence_z_s = 0, 0, 0, 0, 0
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            kl_local += tensors["kl_local_sum"]
            n_obs += tensors["n_obs"]
            kl_divergence_z_s += tensors["kl_divergence_z_s"]
        # kl global same for each minibatch
        kl_global = outputs[0]["kl_global"]
        elbo += kl_global
        self.log("elbo_train", elbo / n_obs)
        self.log("reconstruction_loss_train", rec_loss / n_obs)
        self.log("kl_local_train", kl_local / n_obs)
        self.log("kl_global_train", kl_global)
        self.log("kl_divergence_z_s_train", kl_divergence_z_s / n_obs)

    def validation_step(self, batch, batch_idx):
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss
        self.log("validation_loss", scvi_loss.loss, on_epoch=True)
        return {
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            "kl_global": scvi_loss.kl_global,
            "n_obs": reconstruction_loss.shape[0],
            "kl_divergence_z_s": scvi_loss.kl_divergence_z_s.sum(),
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        n_obs, elbo, rec_loss, kl_local, kl_divergence_z_s = 0, 0, 0, 0, 0 
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            kl_local += tensors["kl_local_sum"]
            n_obs += tensors["n_obs"]
            kl_divergence_z_s = tensors["kl_divergence_z_s"]
        # kl global same for each minibatch
        kl_global = outputs[0]["kl_global"]
        elbo += kl_global
        self.log("elbo_validation", elbo / n_obs)
        self.log("reconstruction_loss_validation", rec_loss / n_obs)
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_global_validation", kl_global)
        self.log("kl_divergence_z_s_validation", kl_divergence_z_s / n_obs)


class DeltaETMTrainingPlan(AdversarialTrainingPlan):
    def __init__(
        self, 
        *args, 
        source_classifier: Union[bool, Classifier] = True,
        scale_classification_loss: Union[float, Literal["auto"]] = "auto", 
        **kwargs):
        super().__init__(*args, **kwargs)
        self.source_classifier = source_classifier
        self.scale_classification_loss = scale_classification_loss
        if kwargs["adversarial_classifier"] is True:
            self.n_output_classifier = 2
            self.adversarial_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=3,
                logits=True,
            )
        else:
            self.adversarial_classifier = kwargs["adversarial_classifier"]
        
        # source classifier
        if self.source_classifier is True:
            self.n_output_classifier = 2
            self.source_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=3,
                logits=True,
            )
        else:
            self.source_classifier = source_classifier

            
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # tuning parameter for adversarial 
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        # tunung parameter for source classification loss
        gamma = (
            1 - self.kl_weight
            if self.scale_classification_loss == "auto"
            else self.scale_classification_loss
        )
    
        if optimizer_idx == 0:
            # batch contains both data loader outputs
            scvi_loss_objs = []
            n_obs = 0
            zs = []
            #z_inds = []
            for (i, tensors) in enumerate(batch):
                n_obs += tensors[_CONSTANTS.X_KEY].shape[0]
                self.loss_kwargs.update(dict(kl_weight=self.kl_weight, mode=i))
                inference_kwargs = dict(mode=i)
                generative_kwargs = dict(mode=i)
                inference_outputs, _, scvi_loss = self.forward(
                    tensors,
                    loss_kwargs=self.loss_kwargs,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                )
                zs.append(inference_outputs["z"])
                #z_inds.append(inference_outputs["z_ind"])
                scvi_loss_objs.append(scvi_loss)

            loss = sum([scl.loss for scl in scvi_loss_objs])
            loss /= n_obs
            rec_loss = sum([scl.reconstruction_loss.sum() for scl in scvi_loss_objs])
            kl = sum([scl.kl_local.sum() for scl in scvi_loss_objs])

            
            batch_tensor = [
                torch.zeros((z.shape[0], 1), device=z.device) + i
                for i, z in enumerate(zs)
            ]
            # fool classifier if doing adversarial training
            if kappa > 0 and self.adversarial_classifier is not False:
                fool_loss = self.loss_adversarial_classifier(
                    torch.cat(zs), torch.cat(batch_tensor), False
                )
                loss += fool_loss * kappa

            # source classifier if using source-specific claddifier
            if gamma > 0 and self.source_classifier is not False:
                classification_loss = self.loss_adversarial_classifier(
                    torch.cat(zs), torch.cat(batch_tensor), True
                )
                loss += classification_loss * gamma

            return {
                "loss": loss,
                "reconstruction_loss_sum": rec_loss,
                "kl_local_sum": kl,
                "kl_global": 0.0,
                "n_obs": n_obs,
            }

        # train adversarial classifier
        # this condition will not be met, keep for record only 
        # this condition will not be met unless self.adversarial_classifier is not False
        if optimizer_idx == 1:
            zs = []
            for (i, tensors) in enumerate(batch):
                inference_inputs = self.module._get_inference_input(tensors)
                inference_inputs.update({"mode": i})
                outputs = self.module.inference(**inference_inputs)
                zs.append(outputs["z"])

            batch_tensor = [
                torch.zeros((z.shape[0], 1), device=z.device) + i
                for i, z in enumerate(zs)
            ]
            loss = self.loss_adversarial_classifier(
                torch.cat(zs).detach(), torch.cat(batch_tensor), True
            )
            loss *= kappa

            return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.loss_kwargs.update(dict(kl_weight=self.kl_weight, mode=dataloader_idx))
        inference_kwargs = dict(mode=dataloader_idx)
        generative_kwargs = dict(mode=dataloader_idx)
        _, _, scvi_loss = self.forward(
            batch,
            loss_kwargs=self.loss_kwargs,
            inference_kwargs=inference_kwargs,
            generative_kwargs=generative_kwargs,
        )
        reconstruction_loss = scvi_loss.reconstruction_loss
        return {
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            "kl_global": scvi_loss.kl_global,
            "n_obs": reconstruction_loss.shape[0],
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        n_obs, elbo, rec_loss, kl_local = 0, 0, 0, 0
        for dl_out in outputs:
            for tensors in dl_out:
                elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
                rec_loss += tensors["reconstruction_loss_sum"]
                kl_local += tensors["kl_local_sum"]
                n_obs += tensors["n_obs"]
        # kl global same for each minibatch
        self.log("elbo_validation", elbo / n_obs)
        self.log("reconstruction_loss_validation", rec_loss / n_obs)
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_global_validation", 0.0)