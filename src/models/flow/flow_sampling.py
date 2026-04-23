"""Sampling methods for endpoint-predicting rectified flow."""

import torch as th
from tqdm.auto import tqdm


class RectifiedFlowSamplingMixin:
    """Rectified-flow sampling mixin."""

    @staticmethod
    def _build_unconditional_self_condition(self_condition):
        """Drop batch conditioning while preserving dataset and gene context."""
        if self_condition is None:
            return None
        ret = dict(self_condition)
        ret["batch_emb"] = None
        return ret

    def predict_endpoint(
        self,
        model,
        x_t,
        control_input_t,
        t,
        self_condition=None,
        guidance_strength=0.0,
        endpoint_estimate=None,
        model_kwargs=None,
    ):
        """
        Predict the terminal endpoint at the current state and time.

        :param model: Endpoint model.
        :param x_t: Current state tensor.
        :param control_input_t: Control branch input tensor.
        :param t: Continuous time tensor in [0, 1).
        :param self_condition: Conditioning inputs for the model.
        :param guidance_strength: Classifier-free guidance scale in endpoint space.
        :param endpoint_estimate: Optional self-conditioned endpoint estimate.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :return: Endpoint tensor.
        """
        if model_kwargs is None:
            model_kwargs = {}

        output = self.get_model_output(
            model=model,
            x_t=x_t,
            control_input_t=control_input_t,
            t=t,
            self_condition=self_condition,
            model_kwargs=model_kwargs,
            endpoint_estimate=endpoint_estimate,
        )
        endpoint = output["x"]

        if self_condition is not None and guidance_strength != 0.0:
            uncond_output = self.get_model_output(
                model=model,
                x_t=x_t,
                control_input_t=control_input_t,
                t=t,
                self_condition=self._build_unconditional_self_condition(self_condition),
                model_kwargs=model_kwargs,
                endpoint_estimate=endpoint_estimate,
            )
            endpoint = (1.0 + guidance_strength) * endpoint - guidance_strength * uncond_output["x"]

        return endpoint

    def sample_euler_loop(
        self,
        model,
        x_start,
        control_input_start,
        self_condition=None,
        flow_steps=8,
        guidance_strength=0.0,
        clip_denoised=True,
        progress=False,
        model_kwargs=None,
    ):
        """
        Integrate the flow ODE with a fixed-step Euler solver.

        :param model: Endpoint model.
        :param x_start: Initial state x(0), sampled from Gaussian noise.
        :param control_input_start: Fixed control branch input tensor.
        :param self_condition: Conditioning inputs for the model.
        :param flow_steps: Number of Euler steps from t=0 to t=1.
        :param guidance_strength: Classifier-free guidance scale in endpoint space.
        :param clip_denoised: Whether to clip only the final generated counts.
        :param progress: Whether to show a progress bar.
        :param model_kwargs: Extra kwargs forwarded to the model.
        :return: Final sample tensor and an empty trajectory placeholder.
        """
        if model_kwargs is None:
            model_kwargs = {}
        assert flow_steps > 0, "sampling.flow_steps must be a positive integer."
        assert control_input_start is not None
        control_input_start = x_start if control_input_start is None else control_input_start
        sample = x_start.clone()
        endpoint_estimate = None
        dt = 1.0 / float(flow_steps)
        iterator = range(flow_steps)
        if progress:
            iterator = tqdm(iterator, total=flow_steps, leave=False)

        for step_idx in iterator:
            t_value = step_idx / float(flow_steps)
            t = th.full((sample.shape[0],), t_value, device=sample.device, dtype=sample.dtype)
            x1_pred = self.predict_endpoint(
                model=model,
                x_t=sample,
                control_input_t=control_input_start,
                t=t,
                self_condition=self_condition,
                guidance_strength=guidance_strength,
                endpoint_estimate=endpoint_estimate,
                model_kwargs=model_kwargs,
            )
            denominator = (1.0 - self._expand_time(t, sample)).clamp_min(1e-6)
            velocity = (x1_pred - sample) / denominator
            if getattr(model.model_cfg, "enable_self_condition", False):
                endpoint_estimate = x1_pred
            sample = sample + dt * velocity

        sample = self.clip_terminal_sample(sample + control_input_start, clip_denoised=clip_denoised)
        return sample, []


__all__ = ["RectifiedFlowSamplingMixin"]
