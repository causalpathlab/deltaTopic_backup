import torch

from scvi import _CONSTANTS
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass
from scvi.train import AdversarialTrainingPlan, TrainingPlan
from typing import Callable, Optional, Union
from scvi._compat import Literal

class DeltaETMTrainingPlan(AdversarialTrainingPlan):
    def __init__(
        self, 
        *args, 
        **kwargs):
        super().__init__(*args, **kwargs)
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
                
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # tuning parameter for adversarial 
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
       
    
        if optimizer_idx == 0:
            # batch contains both data loader outputs
            scvi_loss_objs = []
            n_obs = 0
            zs = []
            # reconstruction loss for both datasets
            rec_losses = []
            kls = []
            for (i, tensors) in enumerate(batch):
                # the number of observations in the bathch for each dataset
                n_obs_each = tensors[_CONSTANTS.X_KEY].shape[0]
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
                scvi_loss_objs.append(scvi_loss)
                rec_losses.append(scvi_loss.reconstruction_loss.sum()/n_obs_each)
                kls.append(scvi_loss.kl_local.sum()/n_obs_each)
                
            loss = sum([scl.loss for scl in scvi_loss_objs])
            loss /= n_obs
            rec_loss = sum([scl.reconstruction_loss.sum() for scl in scvi_loss_objs])
            kl = sum([scl.kl_local.sum() for scl in scvi_loss_objs])

            # fool classifier if doing adversarial training
            batch_tensor = [
                torch.zeros((z.shape[0], 1), device=z.device) + i
                for i, z in enumerate(zs)
            ]
            if kappa > 0 and self.adversarial_classifier is not False:
                fool_loss = self.loss_adversarial_classifier(
                    torch.cat(zs), torch.cat(batch_tensor), False
                )
                loss += fool_loss * kappa
            self.log("kappa", kappa, on_epoch=True)
            self.log("rec_loss_0", rec_losses[0], on_epoch=True)
            self.log("rec_loss_1", rec_losses[1], on_epoch=True)
            self.log("kl_0", kls[0], on_epoch=True)
            self.log("kl_1", kls[1], on_epoch=True)
            self.log("fool_loss", fool_loss, on_epoch=True)
            #self.log("rho", self.decoder.rho, on_epoch=True)
            
            
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