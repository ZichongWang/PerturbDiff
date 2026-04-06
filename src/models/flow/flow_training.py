"""Training methods for rectified flow."""

import torch as th
from geomloss import SamplesLoss


def mean_flat(tensor):
    """
    Compute the mean across all non-batch dimensions.

    :param tensor: Input tensor with batch dimension at index 0.
    :return: Tensor reduced to per-sample means.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class RectifiedFlowTrainingMixin:
    """Rectified-flow training mixin."""

    def get_model_output(
        self,
        model,
        x_t,
        control_input_t,
        t,
        self_condition=None,
        model_kwargs=None,
        endpoint_estimate=None,
    ):
        """
        Run Cross_DiT once at continuous time `t` and return branch outputs.

        :param model: Velocity model.
        :param x_t: Interpolated state tensor.
        :param control_input_t: Control branch input tensor.
        :param t: Continuous time tensor in [0, 1).
        :param self_condition: Conditioning inputs for the model.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param endpoint_estimate: Optional self-conditioned endpoint estimate.
        :return: Model output dictionary.
        """
        if model_kwargs is None:
            model_kwargs = {}
        assert model.model_name == "Cross_DiT"
        x_input, control_input = self._prepare_model_inputs(x_t, control_input_t, endpoint_estimate=endpoint_estimate)
        return model(
            x_input,
            control_input,
            self._scale_timesteps(t).unsqueeze(1),
            self_condition=self_condition,
            **model_kwargs,
        )

    @staticmethod
    def _compute_mmd_weight(t, alpha, gamma):
        """Compute the per-sample weighted-MMD factor alpha * t^gamma / (1 - t)."""
        assert alpha >= 0.0, "optimization.mmd_weight_alpha must be non-negative."
        assert gamma >= 0.0, "optimization.mmd_weight_gamma must be non-negative."
        if alpha == 0.0:
            return th.zeros_like(t)
        denominator = (1.0 - t).clamp_min(1e-6)
        return alpha * th.pow(t, gamma) / denominator

    def training_losses(
        self,
        model,
        x_start,
        control_input_start,
        self_condition=None,
        model_kwargs=None,
        p_drop_cond=0.0,
        MMD_loss_fn=None,
        mmd_weight_alpha=0.0,
        mmd_weight_gamma=0.0,
        return_model_output=False,
    ):
        """
        Prepare flow-matching inputs and compute losses for one batch.

        :param model: Velocity model.
        :param x_start: Ground-truth perturbed cells.
        :param control_input_start: Ground-truth control cells.
        :param self_condition: Conditioning inputs for the model.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :param p_drop_cond: Probability of dropping conditional batch embeddings.
        :param MMD_loss_fn: Optional MMD loss function override.
        :param mmd_weight_alpha: Non-negative alpha in the weighted-MMD term.
        :param mmd_weight_gamma: Non-negative gamma in the weighted-MMD term.
        :param return_model_output: Whether to also return raw model outputs.
        :return: Loss term dictionary, and optional model output dict.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if MMD_loss_fn is None:
            MMD_loss_fn = SamplesLoss(loss="energy", blur=0.05)

        t = th.rand(x_start.shape[0], device=x_start.device, dtype=x_start.dtype).clamp(max=1.0 - 1e-6)
        t_expanded = self._expand_time(t, x_start)
        x0 = self.sample_base_state(x_start)
        x_t = (1.0 - t_expanded) * x0 + t_expanded * x_start
        target_velocity = x_start - x0

        control_input_t = control_input_start
        if model.model_cfg.p_drop_control > 0.0 and th.rand(1, device=x_start.device) < model.model_cfg.p_drop_control:
            control_input_t = th.zeros_like(control_input_t)

        working_self_condition = self._clone_self_condition(self_condition)
        if working_self_condition is not None:
            working_self_condition["cont_emb"] = control_input_start
            if p_drop_cond > 0.0 and th.rand(1, device=x_start.device) < p_drop_cond:
                working_self_condition["batch_emb"] = None

        endpoint_estimate = None
        use_self_condition_now = bool(getattr(model.model_cfg, "enable_self_condition", False)) and bool(
            (th.rand(1, device=x_start.device) > 0.5).item()
        )
        if use_self_condition_now:
            with th.no_grad():
                first_pass = self.get_model_output(
                    model=model,
                    x_t=x_t,
                    control_input_t=control_input_t,
                    t=t,
                    self_condition=working_self_condition,
                    model_kwargs=model_kwargs,
                    endpoint_estimate=None,
                )
            endpoint_estimate = self.velocity_to_endpoint(x_t, first_pass["x"], t)

        model_output = self.get_model_output(
            model=model,
            x_t=x_t,
            control_input_t=control_input_t,
            t=t,
            self_condition=working_self_condition,
            model_kwargs=model_kwargs,
            endpoint_estimate=endpoint_estimate,
        )
        velocity_pred = model_output["x"]
        x1_hat = self.velocity_to_endpoint(x_t, velocity_pred, t)

        terms = {}
        mse = mean_flat((target_velocity - velocity_pred) ** 2)
        if model.model_cfg.no_mse_loss:
            mse = th.zeros_like(mse)
        terms["mse"] = mse

        if mmd_weight_alpha == 0.0:
            raw_mmd = th.zeros_like(mse)
            weighted_mmd = th.zeros_like(mse)
        else:
            raw_mmd = MMD_loss_fn(x_start.type_as(x1_hat), x1_hat)
            weighted_mmd = raw_mmd * self._compute_mmd_weight(t, mmd_weight_alpha, mmd_weight_gamma).type_as(raw_mmd)

        terms["mmd_weighted"] = weighted_mmd
        terms["mmd_raw"] = raw_mmd

        if return_model_output:
            return terms, model_output
        return terms


__all__ = ["RectifiedFlowTrainingMixin"]
