from src.utils.typing_utils import *

import os
from omegaconf import OmegaConf

from torch import optim
from torch.optim import lr_scheduler
from diffusers.training_utils import *
from diffusers.optimization import get_scheduler


import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, euler_angles_to_matrix, rotation_6d_to_matrix

def normalize_center_unit(points: torch.Tensor, eps=1e-8):
    """
    points: (B, N, 3)
    return:
      normalized_points: (B, N, 3)
      center: (B, 1, 3)
      scale: (B, 1, 1)
    """
    # 1) center to zero
    center = points.mean(dim=1, keepdim=True)        # (B,1,3)
    centered = points - center                       # (B,N,3)

    # 2) scale so that max distance = 1
    dist = torch.norm(centered, dim=-1, keepdim=True)  # (B,N,1)
    scale = dist.max(dim=1, keepdim=True)[0]           # (B,1,1)

    normalized = centered / (scale + eps)
    return normalized

def transform_points(points, pose6d, scale, translation):
    B, N, _ = points.shape
    R = rotation_6d_to_matrix(pose6d)  # (B,3,3)

    # scale -> (B,1,1)
    if scale.dim() == 1:              # (B,)
        scale_ = scale.view(B, 1, 1)
    else:                             # (B,1) or (B,k)
        scale_ = scale.view(B, 1, -1).mean(-1).unsqueeze(-1)

    t = translation.view(B, 1, 3)     # (B,1,3)

    # rotate: (B,N,3) @ (B,3,3) -> (B,N,3)
    p_rot = torch.matmul(points, R)

    p_out = scale_ * p_rot + t
    return p_out

    
    return normalize_center_unit(p_out)
   
   
# https://github.com/huggingface/diffusers/pull/9812: fix `self.use_ema_warmup`
class MyEMAModel(EMAModel):
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        foreach: bool = False,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            foreach (bool): Use torch._foreach functions for updating shadow parameters. Should be faster.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """

        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            # use_ema_warmup = True

        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        self.model_cls = model_cls
        self.model_config = model_config

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            # cur_decay_value = (1 + step) / (10 + step)
            cur_decay_value = self.decay

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        """
        Update EMA weights.

        Diffusers' default implementation assumes ``shadow_params`` live on the
        same device as the live model params. For large multi-GPU training runs
        that can be fragile, so we support keeping EMA weights on CPU and copy
        each source param to the shadow param's device on demand.
        """
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)

        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for s_param, param in zip(self.shadow_params, parameters):
            src = param.detach()
            if src.device != s_param.device or src.dtype != s_param.dtype:
                if src.is_floating_point():
                    src = src.to(device=s_param.device, dtype=s_param.dtype, non_blocking=True)
                else:
                    src = src.to(device=s_param.device, non_blocking=True)

            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - src))
            else:
                s_param.copy_(src)

def get_configs(yaml_path: str, cli_configs: List[str]=[], **kwargs) -> DictConfig:
    yaml_configs = OmegaConf.load(yaml_path)
    cli_configs = OmegaConf.from_cli(cli_configs)

    configs = OmegaConf.merge(yaml_configs, cli_configs, kwargs)
    OmegaConf.resolve(configs)  # resolve ${...} placeholders
    return configs

def get_optimizer(name: str, params: Parameter, **kwargs) -> Optimizer:
    if name == "adamw":
        return optim.AdamW(params=params, **kwargs)
    else:
        raise NotImplementedError(f"Not implemented optimizer: {name}")

def get_lr_scheduler(name: str, optimizer: Optimizer, **kwargs) -> LRScheduler:
    if name == "one_cycle":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs["max_lr"],
            total_steps=kwargs["total_steps"],
            pct_start=kwargs["pct_start"],
        )
    elif name == "cosine_warmup":
        return get_scheduler(
            "cosine", optimizer,
            num_warmup_steps=kwargs["num_warmup_steps"],
            num_training_steps=kwargs["total_steps"],
        )
    elif name == "constant_warmup":
        return get_scheduler(
            "constant_with_warmup", optimizer,
            num_warmup_steps=kwargs["num_warmup_steps"],
            num_training_steps=kwargs["total_steps"],
        )
    elif name == "constant":
        return lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda _: 1)
    elif name == "linear_decay":
        return lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: max(0., 1. - epoch / kwargs["total_epochs"]),
        )
    else:
        raise NotImplementedError(f"Not implemented lr scheduler: {name}")

def save_experiment_params(
    args: Namespace, 
    configs: DictConfig, 
    save_dir: str
) -> Dict[str, Any]:
    params = OmegaConf.merge(configs, {"args": {k: str(v) for k, v in vars(args).items()}})
    OmegaConf.save(params, os.path.join(save_dir, "params.yaml"))
    return dict(params)


def save_model_architecture(model: Module, save_dir: str) -> None:
    num_buffers = sum(b.numel() for b in model.buffers())
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f"Number of buffers: {num_buffers}\n" +\
        f"Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n" +\
        f"Model architecture:\n{model}"

    with open(os.path.join(save_dir, "model.txt"), "w") as f:
        f.write(message)


import torch
from icecream import ic
def dice_loss(gt_masks, pred_masks, is_logits=True):
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_masks.to(pred_masks.dtype))
    pred_masks = pred_masks.sigmoid()
    
    intersection = torch.sum(pred_masks * gt_masks, dim=[1,2])  # (B)
    union = torch.sum(pred_masks, dim=[1,2]) + torch.sum(gt_masks, dim=[1,2])  # (B)
    
    dice_coef = (2 * intersection + 1e-8) / (union + 1e-8)  # (N, C)
    dice_loss = 1 - torch.mean(dice_coef)  # 1

    return dice_loss + bce_loss*0.1
