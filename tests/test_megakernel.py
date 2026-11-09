"""The Triton kernel path has to reproduce the reference per-parameter loop, and keep only the state it needs.

The parameter set is a LoRA's (every tensor 2-D, so the second moment is factored, which is what the fast path covers).
The fast path bakes no shape and no tensor count into the compiled code, but it does only support the options the current
training config uses: a case outside that set has to fall back to the reference loop rather than run a kernel that
ignores it.
"""

import io

import pytest
import torch
import triton
import triton.language as tl

import prodigyplus

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# the shapes of a rank-64 Anima LoRA
SHAPES = [((2048, 64), 3), ((64, 2048), 3), ((64, 1024), 2), ((8192, 64), 1), ((64, 8192), 1)]

# options: the first three are inside the fast path's set, the rest are not (and must fall back)
CASES = {
    "default": {},
    "no_decay": {"weight_decay": 0.0},
    "large_steps": {"lr": 100.0, "weight_decay": 0.1},
    "unfactored": {"factored": False},
    "epsilon": {"eps": 1e-8},
    "decoupled_decay": {"weight_decay": 0.1, "weight_decay_by_lr": False},
    "bias_correction": {"use_bias_correction": True},
    "no_schedulefree": {"use_schedulefree": False},
}
SUPPORTED = ("default", "no_decay", "large_steps")


def make_params(seed=0, scale=0.5, dtype=torch.bfloat16):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    params = []
    for shape, count in SHAPES:
        for _ in range(count):
            t = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32) * scale
            params.append(torch.nn.Parameter(t.to(dtype)))
    return params


def set_grads(params, seed, scale=0.1):
    """Large gradients and parameters, so a structural error moves the result far more than rounding does."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for p in params:
        p.grad = (torch.randn(p.shape, generator=gen, device="cuda", dtype=torch.float32) * scale).to(p.dtype)


def build(params, fast=True, **overrides):
    # d0 = 0.1 instead of the default 1e-6: `d` only grows when the Prodigy numerator is positive, so with the default the
    # step size stays at 1e-6 for a synthetic run and the parameters would barely move, which hides structural errors
    kwargs = dict(lr=1.0, betas=(0.95, 0.99), weight_decay=0.01, eps=None, d0=0.1, use_megakernel=fast)
    kwargs.update(overrides)
    return prodigyplus.ProdigyPlusScheduleFree(params, **kwargs)


def train(steps=8, seed=7, fast=True, dtype=torch.bfloat16, **overrides):
    params = make_params(dtype=dtype)
    kwargs = {"stochastic_rounding": False}  # the exact comparison needs the same rounding as the reference
    kwargs.update(overrides)
    opt = build(params, fast, **kwargs)
    torch.manual_seed(seed)
    for step in range(steps):
        set_grads(params, seed=seed + step)
        opt.step()
        for p in params:
            p.grad = None
    return params, opt


def relative_mean_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-12)).item()


def stats(opt):
    group = opt.param_groups[0]
    return {name: group[name] for name in ("d", "d_numerator", "d_denom", "k")}


def state(opt):
    return opt.param_groups[0].get("megakernel_state")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("case", SUPPORTED, ids=list(SUPPORTED))
def test_megakernel_matches_the_per_parameter_loop(case, dtype):
    """Both parameter dtypes the loop takes: bf16 (`--full_bf16`) and fp32 (the default with bf16 mixed precision)."""
    kernel_params, kernel_opt = train(dtype=dtype, **CASES[case])
    ref_params, ref_opt = train(fast=False, dtype=dtype, **CASES[case])
    assert state(kernel_opt) is not None  # the kernel path ran

    for i, (a, b) in enumerate(zip(kernel_params, ref_params)):
        assert relative_mean_difference(a, b) < 5e-3, (case, dtype, i, relative_mean_difference(a, b))

    kernel_stats, ref_stats = stats(kernel_opt), stats(ref_opt)
    assert kernel_stats["k"] == ref_stats["k"]
    assert kernel_stats["d"] == pytest.approx(ref_stats["d"], rel=1e-6)
    assert kernel_stats["d_numerator"] == pytest.approx(ref_stats["d_numerator"], rel=5e-2, abs=1e-13)
    assert kernel_stats["d_denom"] == pytest.approx(ref_stats["d_denom"], rel=5e-2, abs=1e-13)


def test_the_state_dtypes_are_the_reference_ones():
    """`z` follows the parameter, the subsamples and the second moment are bf16/fp32 respectively, in either case."""
    for dtype, z_dtype in ((torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)):
        params, opt = train(steps=2, dtype=dtype)
        assert state(opt) is not None
        per_param = opt.state[params[0]]
        assert per_param["z"].dtype == z_dtype
        assert per_param["p0"].dtype == torch.bfloat16
        assert per_param["s"].dtype == torch.bfloat16
        assert per_param["exp_avg_sq_row"].dtype == torch.float32


@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_the_fast_path_is_taken_exactly_where_it_applies(case):
    """An option outside the fast path's set must fall back, not run a kernel that ignores it."""
    _, opt = train(steps=1, **CASES[case])
    if case in SUPPORTED:
        assert state(opt) is not None, case
    else:
        assert state(opt) is None, case
        assert len(opt.state) == len(opt.param_groups[0]["params"])  # the loop initialised per-parameter state


def test_megakernel_can_be_disabled():
    _, opt = train(steps=1, fast=False)
    assert state(opt) is None


def test_one_dimensional_parameters_fall_back():
    params = [torch.nn.Parameter(torch.randn(2048, device="cuda").bfloat16())]
    params[0].grad = torch.randn(2048, device="cuda").bfloat16() * 0.01
    opt = build(params)
    opt.step()
    assert state(opt) is None


def test_float16_parameters_fall_back():
    params = [torch.nn.Parameter(torch.randn(64, 2048, device="cuda").half())]
    params[0].grad = torch.randn(64, 2048, device="cuda").half() * 0.01
    opt = build(params)
    opt.step()
    assert state(opt) is None  # fp16 is not one of the two dtypes the loop takes


def test_a_missing_gradient_falls_back():
    params = make_params()
    set_grads(params, seed=0)
    opt = build(params)
    opt.step()
    assert state(opt) is not None
    params[0].grad = None  # the loop steps aside for this step instead of treating it as zero
    opt.step()
    assert params[0].detach().abs().sum() > 0


def test_the_grad_table_follows_new_gradient_tensors():
    """`zero_grad(set_to_none=True)` gives every parameter a new gradient tensor each step."""
    params = make_params()
    opt = build(params)
    for step in range(3):
        for p in params:
            p.grad = None
        set_grads(params, seed=step)
        before = [p.detach().clone() for p in params]
        opt.step()
        assert all(not torch.equal(a, b) for a, b in zip(before, params)), step
    assert state(opt) is not None


def test_parameters_actually_move():
    params, start = train(steps=8)[0], make_params()
    moved = max(
        ((a.detach().float() - b.detach().float()).abs().mean() / b.detach().float().abs().mean().clamp_min(1e-12)).item()
        for a, b in zip(params, start)
    )
    assert moved > 0.2, moved  # d0=0.1 and lr=1: the step size is O(0.1), so a bf16 parameter has to move visibly


def test_only_the_step_scratch_is_added_to_the_state():
    """The state is the reference loop's own (`z`, `p0`, `s`, the factored second moment); the kernels add only scratch."""
    params = make_params()
    opt = build(params)
    set_grads(params, seed=11)
    n = sum(p.numel() for p in params)
    base = torch.cuda.memory_allocated()
    opt.step()
    allocated = torch.cuda.memory_allocated() - base

    kernel_state = state(opt)
    scratch = sum(t.nbytes for entry in kernel_state._scratch.values() for t in entry.values())
    assert allocated - scratch < 2.6 * n  # z (2 B/param) + p0/s (0.36) and nothing else
    assert scratch < 0.2 * n  # the row-block partials and the two accumulators per parameter
    # the reference loop's state tensors are the ones the kernels walk
    for p in params[:3]:
        assert {"z", "p0", "s", "exp_avg_sq_row", "exp_avg_sq_col"} <= set(opt.state[p])


def round_trip(opt):
    """A checkpoint the way a trainer saves one: `state_dict()` shares storage with the live state, so it has to be
    serialized before a second optimizer can use it."""
    buffer = io.BytesIO()
    torch.save(opt.state_dict(), buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def test_state_dict_round_trip():
    params, opt_a = train(steps=3)
    checkpoint = round_trip(opt_a)
    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt_b = build(params_b)
    opt_b.load_state_dict(checkpoint)

    set_grads(params, seed=500)
    set_grads(params_b, seed=500)
    opt_a.step()
    opt_b.step()
    for a, b in zip(params, params_b):
        assert (a.detach().float() - b.detach().float()).abs().max() < 2e-2

    # the state dict is the reference one: per-parameter entries, no extension keys
    assert set(opt_b.state_dict()) == {"state", "param_groups"}


def test_a_per_parameter_state_resumes_on_the_kernel_path():
    """The kernel path stores no format of its own, so a state from the loop resumes on it.

    `use_megakernel` is a hyperparameter, so `state_dict` carries the source optimizer's value; switching paths is a
    matter of setting it on the loaded group. A reloaded checkpoint has lost the fp32 dtype of the factored second moment
    (torch casts state tensors to the parameter's dtype), which the kernel path restores.
    """
    params, opt = train(steps=3, fast=False)
    checkpoint = round_trip(opt)
    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt_b = build(params_b)
    opt_b.load_state_dict(checkpoint)
    assert opt_b.param_groups[0]["use_megakernel"] is False
    opt_b.param_groups[0]["use_megakernel"] = True

    set_grads(params_b, seed=900)
    before = [p.detach().clone() for p in params_b]
    opt_b.step()
    assert state(opt_b) is not None
    assert all(not torch.isnan(p).any() for p in params_b)
    assert all(not torch.equal(a, b) for a, b in zip(before, params_b))


# the stochastic rounding primitive itself: repeatedly rounding a residual has to average out, where round-to-nearest
# stalls. That is the property the Philox dither exists for, and it does not need to be PyTorch's stream.


@triton.jit
def _round_kernel(t_in, t_out, n, seed, offset, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(t_in + offs, mask=mask)
    noise, _, _, _ = tl.randint4x(seed, (offset + offs).to(tl.uint32))
    bits = x.to(tl.int32, bitcast=True) + (noise.to(tl.int32) & 0xFFFF)
    tl.store(t_out + offs, (bits & -65536).to(tl.float32, bitcast=True).to(tl.bfloat16), mask=mask)


def test_stochastic_rounding_accumulates_a_residual():
    """Add 1e-4 to 1.0 a thousand times, rounding to bf16 after every addition.

    bf16's ulp at 1.0 is 2**-8, so round-to-nearest returns 1.0 forever while an unbiased dither walks up by ~0.1.
    """
    torch.manual_seed(0)
    delta, steps = 1e-4, 1000
    rounded = torch.empty(1, device="cuda", dtype=torch.bfloat16)

    nearest = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    stochastic = nearest.clone()
    for step in range(steps):
        nearest = (nearest + delta).to(torch.bfloat16).float()  # round to nearest: stalls
        _round_kernel[(1,)](stochastic + delta, rounded, 1, 1234, step * 8, BLOCK=8)
        stochastic = rounded.float()

    assert nearest.item() == 1.0
    drift = stochastic.item() - 1.0  # ~steps * delta, i.e. the residual is not lost
    assert 0.05 < drift < 0.25, drift
