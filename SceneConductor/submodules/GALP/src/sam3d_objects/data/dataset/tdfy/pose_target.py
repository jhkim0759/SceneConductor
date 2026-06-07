"""Minimal ScaleShiftInvariant implementation for GALP."""
import torch
from pytorch3d.transforms import Transform3d


class ScaleShiftInvariant:
    """Scale-Shift Invariant normalization for pointmaps (MoGe / Midas eq. 6)."""

    @classmethod
    def get_scale_and_shift(cls, pointmap: torch.Tensor):
        """Return (scale, shift) tensors of shape (3,) each."""
        shift_z = pointmap[..., -1].nanmedian().unsqueeze(0)
        shift = torch.zeros_like(shift_z.expand(1, 3))
        shift[..., -1] = shift_z
        shifted_pointmap = pointmap - shift
        scale = shifted_pointmap.abs().nanmean().to(shift.device)
        shift = shift.reshape(3)
        scale = scale.expand(3)
        return scale, shift

    @staticmethod
    def ssi_to_metric(scale: torch.Tensor, shift: torch.Tensor) -> Transform3d:
        """Return a Transform3d T such that T.transform_points(p) = p * scale + shift."""
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        if shift.ndim == 1:
            shift = shift.unsqueeze(0)
        return Transform3d(device=shift.device).scale(scale.float()).translate(shift.float())
