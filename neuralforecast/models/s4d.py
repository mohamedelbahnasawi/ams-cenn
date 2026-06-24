"""S4D — diagonal state-space model baseline (pure-torch, self-contained). Included as a "family
coverage" baseline in this work; NeuralForecast ships no SSM model and we avoid the mamba-ssm CUDA dependency.
Wrapper mirrors the CeNN BaseModel plumbing so it runs under the identical L=512 protocol."""
from typing import Optional
from copy import deepcopy
import torch
import torch.nn as nn  # noqa: F401

from ..losses.pytorch import MAE
from ..common._base_model import BaseModel
from ..s4d.model import S4DModel1D


class S4D(BaseModel):
    """S4D (diagonal state-space) forecaster compatible with NeuralForecast's BaseModel.

    Parameters:
    - h: int, forecast horizon.
    - input_size: int, input window length.
    - n_series: int, number of time-series (multivariate).
    - d_model: int=128, model width.
    - d_state: int=64, SSM state size N per channel.
    - n_layers: int=2, number of S4D layers.
    - dropout: float=0.1.
    - loss / valid_loss: PyTorch losses (default MAE()).
    - plus all standard BaseModel trainer/data args.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True
    RECURRENT = False

    def __init__(
        self,
        h,
        input_size,
        n_series,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y: bool = False,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        training_data_availability_threshold: float = 0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss if valid_loss is not None else loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )
        self.c_out = self.loss.outputsize_multiplier
        self.model = S4DModel1D(
            n_features=n_series,
            seq_length=input_size,
            pred_length=h,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            c_out=self.c_out,
        )

    def forward(self, windows_batch):
        x = windows_batch["insample_y"]  # [B, L, n_series]
        return self.model(x)             # [B, h, n_series * c_out]

    def configure_optimizers(self):
        """S4D recipe (Gu et al.): the SSM kernel parameters (log_dt, log_A_real, A_imag, C) must
        NOT be weight-decayed — the official code tags them `param._optim={'weight_decay':0.0}`;
        decaying them shrinks the learned timescales/poles and degrades long-range modeling. NF's
        default builds one flat AdamW group, so we override to split the wd=0 SSM params from the
        rest. Otherwise identical to BaseModel.configure_optimizers (same optimizer + lr + scheduler).
        All OTHER baselines remain under the uniform wd=0.01 protocol; only S4D's kernel params are
        excepted, per the authors' published recipe (documented in the paper)."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if getattr(p, "_optim", {}).get("weight_decay", None) == 0.0:
                no_decay.append(p)
            else:
                decay.append(p)
        okw = deepcopy(self.optimizer_kwargs) if self.optimizer_kwargs else {}
        okw.pop("lr", None)
        wd = okw.pop("weight_decay", 0.0)
        groups = [{"params": decay, "weight_decay": wd},
                  {"params": no_decay, "weight_decay": 0.0}]
        optimizer = (self.optimizer(groups, lr=self.learning_rate, **okw)
                     if self.optimizer else torch.optim.AdamW(groups, lr=self.learning_rate))

        lr_scheduler = {"frequency": 1, "interval": "step"}
        if self.lr_scheduler:
            lkw = deepcopy(self.lr_scheduler_kwargs) if self.lr_scheduler_kwargs else {}
            lkw.pop("optimizer", None)
            lr_scheduler["scheduler"] = self.lr_scheduler(optimizer=optimizer, **lkw)
        else:
            lr_scheduler["scheduler"] = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=self.lr_decay_steps, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
