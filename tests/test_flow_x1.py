import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.flow.flow_core import RectifiedFlow
import src.models.flow.flow_training as flow_training_module


class DummyEndpointModel:
    def __init__(self, output_value=0.0, enable_self_condition=False):
        self.model_name = "Cross_DiT"
        self.model_cfg = SimpleNamespace(
            p_drop_control=0.0,
            enable_self_condition=enable_self_condition,
            no_mse_loss=False,
        )
        self.output_value = output_value
        self.recorded_x_inputs = []

    def __call__(self, x_input, x_control_input, t, self_condition=None, **kwargs):
        del x_control_input, t, kwargs
        self.recorded_x_inputs.append(x_input.detach().clone())
        base_shape = x_input.shape
        if self.model_cfg.enable_self_condition:
            base_shape = base_shape[:-1] + (base_shape[-1] // 2,)

        batch_emb = None if self_condition is None else self_condition.get("batch_emb")
        fill_value = self.output_value
        if isinstance(fill_value, tuple):
            fill_value = fill_value[0] if batch_emb is not None else fill_value[1]
        return {"x": th.full(base_shape, fill_value, dtype=x_input.dtype, device=x_input.device)}


def test_endpoint_to_velocity_is_finite_near_terminal_time():
    flow = RectifiedFlow(SimpleNamespace(steps=1000, enable_self_condition=False))
    x_t = th.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=th.float32)
    x1_hat = th.tensor([[1.0, 3.0], [4.0, 7.0]], dtype=th.float32)
    t = th.tensor([0.5, 1.0 - 1e-8], dtype=th.float32)

    velocity = flow.endpoint_to_velocity(x_t, x1_hat, t)

    assert th.isfinite(velocity).all()
    assert th.allclose(velocity[0], th.tensor([2.0, 4.0]))


def test_training_losses_use_x1_target_and_unscaled_mmd_weight(monkeypatch):
    flow = RectifiedFlow(
        SimpleNamespace(
            steps=1000,
            pairing_strategy="within_set_random",
            ot_cost="l2_squared",
            ot_reg=0.05,
            ot_num_iters=200,
            ot_sampling="row_multinomial",
            ot_return_coupling=False,
            enable_self_condition=False,
        )
    )
    model = DummyEndpointModel(output_value=0.0, enable_self_condition=False)
    x_start = th.ones(2, 3, dtype=th.float32)
    control = th.zeros_like(x_start)

    monkeypatch.setattr(
        flow_training_module,
        "pair_control_within_set",
        lambda control_input_start, x_start, **kwargs: (control_input_start, {"stub": True}),
    )

    original_rand = flow_training_module.th.rand

    def fake_rand(shape, *args, **kwargs):
        if isinstance(shape, int):
            shape = (shape,)
        return th.full(shape, 0.25, device=kwargs.get("device"), dtype=kwargs.get("dtype", th.float32))

    monkeypatch.setattr(flow_training_module.th, "rand", fake_rand)
    try:
        losses = flow.training_losses(
            model,
            x_start,
            control,
            self_condition={"batch_emb": None},
            MMD_loss_fn=lambda truth, pred: th.ones(truth.shape[0], device=truth.device, dtype=truth.dtype),
            mmd_weight_alpha=2.0,
            mmd_weight_gamma=1.0,
        )
    finally:
        monkeypatch.setattr(flow_training_module.th, "rand", original_rand)

    assert th.allclose(losses["mse"], th.ones(2))
    assert th.allclose(losses["mmd_raw"], th.ones(2))
    assert th.allclose(losses["mmd_weighted"], th.full((2,), 0.5))


def test_sampling_self_condition_uses_guided_x1_prediction():
    flow = RectifiedFlow(SimpleNamespace(steps=1000, enable_self_condition=True, cutoff=0.0))
    model = DummyEndpointModel(output_value=(2.0, 1.0), enable_self_condition=True)
    x_start = th.zeros(1, 2, dtype=th.float32)
    control = th.zeros_like(x_start)
    self_condition = {"batch_emb": th.ones(1, 2, dtype=th.float32), "ds_name": [["fake"]]}

    sample, _ = flow.sample_euler_loop(
        model,
        x_start=x_start,
        control_input_start=control,
        self_condition=self_condition,
        flow_steps=2,
        guidance_strength=0.5,
        clip_denoised=False,
    )

    second_step_cond_input = model.recorded_x_inputs[2]
    guided_x1 = (1.0 + 0.5) * 2.0 - 0.5 * 1.0

    assert second_step_cond_input.shape[-1] == 4
    assert th.allclose(second_step_cond_input[..., 2:], th.full((1, 2), guided_x1))
    assert th.isfinite(sample).all()
