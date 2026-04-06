"""Core rectified-flow class assembly."""

import torch as th

from src.models.flow.flow_sampling import RectifiedFlowSamplingMixin
from src.models.flow.flow_training import RectifiedFlowTrainingMixin


class RectifiedFlow(
    RectifiedFlowSamplingMixin,
    RectifiedFlowTrainingMixin,
):
    """Utilities for rectified-flow training and sampling."""

    def __init__(self, model_cfg):
        """
        Initialize the class instance.

        :param model_cfg: Model configuration object.
        :return: None.
        """
        self.model_cfg = model_cfg
        self.num_time_steps = int(getattr(model_cfg, "steps", 1000))

    def _scale_timesteps(self, t):
        """Map continuous flow time into the backbone timestep embedding range."""
        return t.float() * float(self.num_time_steps)

    @staticmethod
    def _expand_time(t, reference):
        """Broadcast a batch-shaped time tensor to match a reference tensor."""
        expanded = t.float()
        while expanded.ndim < reference.ndim:
            expanded = expanded.unsqueeze(-1)
        return expanded

    @staticmethod
    def _clone_self_condition(self_condition):
        """Create a shallow copy of the conditioning dictionary."""
        if self_condition is None:
            return None
        return dict(self_condition)

    def _prepare_model_inputs(self, x_t, control_input_t, endpoint_estimate=None):
        """Format model inputs based on whether self-conditioning is enabled."""
        if not getattr(self.model_cfg, "enable_self_condition", False):
            return x_t, control_input_t

        if endpoint_estimate is None:
            endpoint_estimate = th.zeros_like(x_t)
        control_self_condition = th.zeros_like(control_input_t)
        return (
            th.cat([x_t, endpoint_estimate], dim=-1),
            th.cat([control_input_t, control_self_condition], dim=-1),
        )

    @staticmethod
    def sample_base_state(reference):
        """Sample the Gaussian flow start state x0 with the same shape as `reference`."""
        return th.randn_like(reference)

    def velocity_to_endpoint(self, x_t, velocity, t):
        """Convert a velocity prediction into a terminal point estimate."""
        return x_t + (1.0 - self._expand_time(t, x_t)) * velocity

    def clip_terminal_sample(self, sample, clip_denoised=True):
        """Clip only the final generated counts, not the velocity field itself."""
        if not clip_denoised:
            return sample
        return sample.masked_fill(sample < getattr(self.model_cfg, "cutoff", 0.0), 0)


def create_flow(model_cfg):
    """Create a rectified-flow process."""
    return RectifiedFlow(model_cfg)


__all__ = ["RectifiedFlow", "create_flow"]
