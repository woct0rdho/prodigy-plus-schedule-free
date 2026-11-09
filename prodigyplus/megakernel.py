"""The step as three Triton kernels over address tables, on the state the reference implementation keeps.

The per-parameter loop launches about 47 kernels per parameter tensor - fp32 upcasts of the parameter, of `z` and of the
gradient, the second moment, `atan2_`, the schedule-free lerp, the weight decay and two copy-backs - so a LoRA with a few
hundred tensors spends its time in launch overhead and in re-reading the same data.

The obvious fix, one flat buffer per group, needs a gather/scatter pair at every step (4 B/param of staging that the
algorithm never asked for) and either an fp32 shadow of every intermediate (eager) or a `torch.compile` graph built from
every parameter tensor (which is where the compile time goes). This module takes the third route: the kernels address the
parameters where they are, through tables of `int64` addresses built once per group, and the state is exactly the
reference implementation's - `state[p]['z']`, `['p0']`, `['s']`, `['exp_avg_sq_row']` and `['exp_avg_sq_col']` - so
`state_dict()` is untouched and a checkpoint resumes on either path.

Neither the arity nor the shapes are baked into the compiled code: a table entry is one `tl.load`, the tile sizes are
compile-time constants and the per-parameter sizes are runtime operands. A LoRA with 500 or 5000 tensors, or with 64 or
8192 wide matrices, compiles the same three kernels once (about 0.2 s in total).

Per step:

1. `_stats_kernel` (one program per parameter): the factored second moment of the gradient and the Prodigy statistics on
   the `1-in-slice_p` subsample. The factor reduced over the columns is complete within a row block, so it is updated in
   place; the factor reduced over the rows needs a sum across blocks, so each block writes a partial for the next kernel.
2. `_finalize_kernel` (one program per parameter, plus one for the group): the row-partial sum and the Prodigy
   accumulators.
3. `_update_kernel` (one program per parameter): the Adam-atan2 update, the schedule-free lerp of `y`, the decoupled
   weight decay and the bf16 stochastic rounding.

A 2-D parameter of shape `(R, C)` has two factors: `pc` of length `C` (mean of squares over the `R` rows) and `pr` of
length `R` (mean over the `C` columns). Which of the two the reference code calls `row` and which it calls `col` depends
on which axis is larger, and so does the normalisation: for `R >= C` the per-column factor is divided by itself plus
1e-30, and for `R < C` the per-row factor is divided by its mean. Both end up as `sqrt(nc[c] * nr[r])`, so only the
update kernel's scalar prelude needs to know the orientation.

The numerics follow the reference step: fp32 second moment, the `s` EMA rounded to bf16 after each of its two halves, the
d-statistics summed per parameter first and then across the group in fp32, and the rounding dithering the low 16 mantissa
bits of the fp32 value before truncating to bf16. The dither comes from Triton's Philox (`tl.randint4x`): the seed is the
key, a monotonic per-step counter plus a per-parameter base is the offset, and two of the four streams of one call cover
the parameter and `z` without sharing a number. That is not PyTorch's stream and does not need to be - stochastic
rounding only needs an unbiased dither, which the low-16-bit draw gives: it rounds up with probability equal to the
fractional part, exactly like the eager `copy_stochastic_`.

The parameters may be bf16 or fp32 (`mixed_precision='bf16'` without `--full_bf16` keeps a LoRA in fp32): `z` follows
the parameter's dtype, `p0`/`s` are bf16 either way, and only a bf16 store needs the dither - an fp32 parameter is
written exactly.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# tile shape: a block of rows times 64 columns (the LoRA rank), which keeps the inner run contiguous
BLOCK_R = 32
BLOCK_C = 64

# slots of the per-step scalar tensor; constexpr so the kernels can read them as module globals, and `__index__` keeps
# them usable as list indices on the host side
_S_DLR, _S_D, _S_BETA2, _S_BETA3, _S_CKP1, _S_XY, _S_DECAY, _S_DUPDATE, _S_VALUE2 = (
    tl.constexpr(slot) for slot in range(9)
)
_ATAN2_A = tl.constexpr(1.2732395)  # Adam-atan2, see CoreOptimiser.update_
_SLICE_P = 11


@triton.jit
def _stats_kernel(
    t_grad,
    t_z,
    t_p0,
    t_s,
    t_pc_partial,
    t_pr,
    t_rows,
    t_cols,
    t_dot,
    t_s_abs,
    scalars,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SLICE_P: tl.constexpr,
    DO_STATS: tl.constexpr,
    PARAM_FP32: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.load(t_rows + pid)
    cols = tl.load(t_cols + pid)
    pr = tl.load(t_pr + pid).to(tl.pointer_type(tl.float32))
    pc_partial = tl.load(t_pc_partial + pid).to(tl.pointer_type(tl.float32))
    beta2 = tl.load(scalars + _S_BETA2)
    value2 = tl.load(scalars + _S_VALUE2)
    # the gradient and `z` have the parameter's dtype, the subsamples are always bf16
    if PARAM_FP32:
        grad = tl.load(t_grad + pid).to(tl.pointer_type(tl.float32))
        z = tl.load(t_z + pid).to(tl.pointer_type(tl.float32))
    else:
        grad = tl.load(t_grad + pid).to(tl.pointer_type(tl.bfloat16))
        z = tl.load(t_z + pid).to(tl.pointer_type(tl.bfloat16))
    if DO_STATS:
        p0 = tl.load(t_p0 + pid).to(tl.pointer_type(tl.bfloat16))
        s = tl.load(t_s + pid).to(tl.pointer_type(tl.bfloat16))
        beta3 = tl.load(scalars + _S_BETA3)
        d_update = tl.load(scalars + _S_DUPDATE)
        dot_ptr = tl.load(t_dot + pid).to(tl.pointer_type(tl.float32))
        s_abs_ptr = tl.load(t_s_abs + pid).to(tl.pointer_type(tl.float32))
    dot = 0.0
    s_abs = 0.0

    for r0 in range(0, rows, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < rows
        pr_sum = tl.zeros([BLOCK_R], tl.float32)
        for c0 in range(0, cols, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask_c = offs_c < cols
            mask = mask_r[:, None] & mask_c[None, :]
            index = offs_r[:, None] * cols + offs_c[None, :]
            x = tl.load(grad + index, mask=mask, other=0.0).to(tl.float32)
            square = x * x
            # the factor reduced over the columns: complete for these rows, the mean divides it below
            pr_sum += tl.sum(square, axis=1)
            # the factor reduced over the rows: one partial per row block, summed in the finalize kernel
            tl.store(pc_partial + (r0 // BLOCK_R) * cols + offs_c, tl.sum(square, axis=0), mask=mask_c)

            if DO_STATS:
                # the 1-in-SLICE_P subsample in the flat order of the parameter, as `get_sliced_tensor` takes it
                selected = (index % SLICE_P) == 0
                sub = index // SLICE_P
                p0_value = tl.load(p0 + sub, mask=selected, other=0.0)
                z_value = tl.load(z + index, mask=selected, other=0.0)
                # the reference `p0 - z` runs in the promoted dtype: bf16 for bf16 parameters (which rounds it), fp32
                # for fp32 ones (where `p0` is the only bf16 operand). Triton would keep the extra mantissa bits.
                x0_minus = p0_value - z_value
                if not PARAM_FP32:
                    x0_minus = x0_minus.to(tl.bfloat16)
                x0_minus = x0_minus.to(tl.float32)
                dot += tl.sum(tl.where(selected, x * x0_minus, 0.0))

                s_old = tl.load(s + sub, mask=selected, other=0.0)
                s_new = (s_old.to(tl.float32) * beta3).to(tl.bfloat16)
                s_new = (s_new.to(tl.float32) + d_update * x).to(tl.bfloat16)
                tl.store(s + sub, s_new, mask=selected)
                s_abs += tl.sum(tl.where(selected, tl.abs(s_new.to(tl.float32)), 0.0))

        pr_new = tl.load(pr + offs_r, mask=mask_r, other=0.0) * beta2 + value2 * (pr_sum / cols)
        tl.store(pr + offs_r, pr_new, mask=mask_r)

    if DO_STATS:
        tl.store(dot_ptr, dot)
        tl.store(s_abs_ptr, s_abs)


@triton.jit
def _finalize_kernel(
    t_pc,
    t_pc_partial,
    t_rows,
    t_cols,
    t_dot,
    t_s_abs,
    running_numerator,
    running_denom,
    scalars,
    n_params,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    DO_STATS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == n_params:
        # one program owns the group statistics, so their accumulation order is fixed and reproducible
        if DO_STATS:
            d_update = tl.load(scalars + _S_DUPDATE)
            numerator = 0.0
            denom = 0.0
            for i in range(n_params):
                dot_ptr = tl.load(t_dot + i).to(tl.pointer_type(tl.float32))
                s_abs_ptr = tl.load(t_s_abs + i).to(tl.pointer_type(tl.float32))
                numerator += tl.load(dot_ptr)
                denom += tl.load(s_abs_ptr)
                tl.store(dot_ptr, 0.0)
                tl.store(s_abs_ptr, 0.0)
            tl.store(running_numerator, tl.load(running_numerator) + numerator * d_update)
            tl.store(running_denom, tl.load(running_denom) + denom)
    else:
        rows = tl.load(t_rows + pid)
        cols = tl.load(t_cols + pid)
        pc = tl.load(t_pc + pid).to(tl.pointer_type(tl.float32))
        pc_partial = tl.load(t_pc_partial + pid).to(tl.pointer_type(tl.float32))
        beta2 = tl.load(scalars + _S_BETA2)
        value2 = tl.load(scalars + _S_VALUE2)
        chunks = tl.cdiv(rows, BLOCK_R)

        for c0 in range(0, cols, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask_c = offs_c < cols
            total = tl.zeros([BLOCK_C], tl.float32)
            for chunk in range(chunks):
                total += tl.load(pc_partial + chunk * cols + offs_c, mask=mask_c, other=0.0)
                tl.store(pc_partial + chunk * cols + offs_c, tl.zeros([BLOCK_C], tl.float32), mask=mask_c)
            pc_new = tl.load(pc + offs_c, mask=mask_c, other=0.0) * beta2 + value2 * (total / rows)
            tl.store(pc + offs_c, pc_new, mask=mask_c)


@triton.jit
def _update_kernel(
    t_param,
    t_grad,
    t_z,
    t_pc,
    t_pr,
    t_rows,
    t_cols,
    t_flip,
    t_rng_base,
    scalars,
    seed,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    PARAM_FP32: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.load(t_rows + pid)
    cols = tl.load(t_cols + pid)
    flip = tl.load(t_flip + pid)
    if PARAM_FP32:
        param = tl.load(t_param + pid).to(tl.pointer_type(tl.float32))
        grad = tl.load(t_grad + pid).to(tl.pointer_type(tl.float32))
        z = tl.load(t_z + pid).to(tl.pointer_type(tl.float32))
    else:
        param = tl.load(t_param + pid).to(tl.pointer_type(tl.bfloat16))
        grad = tl.load(t_grad + pid).to(tl.pointer_type(tl.bfloat16))
        z = tl.load(t_z + pid).to(tl.pointer_type(tl.bfloat16))
    pc = tl.load(t_pc + pid).to(tl.pointer_type(tl.float32))
    pr = tl.load(t_pr + pid).to(tl.pointer_type(tl.float32))
    dlr = tl.load(scalars + _S_DLR)
    d = tl.load(scalars + _S_D)
    ckp1 = tl.load(scalars + _S_CKP1)
    xy_step = tl.load(scalars + _S_XY)
    decay = tl.load(scalars + _S_DECAY)

    # the reference divides the per-column factor by itself plus 1e-30 when the parameter is at least as tall as it is
    # wide, and the per-row factor by its mean when it is wider than tall; 1.0 keeps the other case a plain multiply
    pr_total = 0.0
    for r0 in range(0, rows, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        pr_total += tl.sum(tl.load(pr + offs_r, mask=offs_r < rows, other=0.0))
    pr_mean = tl.where(flip != 0, pr_total / rows, 1.0)

    for r0 in range(0, rows, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        mask_r = offs_r < rows
        row_factor = tl.load(pr + offs_r, mask=mask_r, other=0.0) / pr_mean
        for c0 in range(0, cols, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask_c = offs_c < cols
            mask = mask_r[:, None] & mask_c[None, :]
            index = offs_r[:, None] * cols + offs_c[None, :]
            y = tl.load(param + index, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(grad + index, mask=mask, other=0.0).to(tl.float32)
            z_old = tl.load(z + index, mask=mask, other=0.0).to(tl.float32)
            pc_value = tl.load(pc + offs_c, mask=mask_c, other=0.0)
            col_factor = tl.where(flip != 0, pc_value, pc_value / (pc_value + 1e-30))
            denom = tl.sqrt(col_factor[None, :] * row_factor[:, None])
            update = libdevice.atan2(g * d, denom) * _ATAN2_A
            update = update + decay * y  # decoupled weight decay at Y; the learning rate scales it below
            y_new = y + ckp1 * (z_old - y) - (dlr * xy_step) * update
            z_new = z_old - dlr * update
            if STOCHASTIC:
                # the step lives in the seed (the Philox key), the offsets only separate the parameters and elements
                offset = (tl.load(t_rng_base + pid) + index).to(tl.uint32)
                noise_y, noise_z, _, _ = tl.randint4x(seed, offset)
                # `randint4x` hands out uint32: the mask is taken after the cast, so both stay in [0, 65536)
                y_bits = y_new.to(tl.int32, bitcast=True) + (noise_y.to(tl.int32) & 0xFFFF)
                z_bits = z_new.to(tl.int32, bitcast=True) + (noise_z.to(tl.int32) & 0xFFFF)
                y_new = (y_bits & -65536).to(tl.float32, bitcast=True)
                z_new = (z_bits & -65536).to(tl.float32, bitcast=True)
            # the stores take the buffer's own dtype: bf16 after the dither (already representable), fp32 unchanged
            tl.store(param + index, y_new.to(param.dtype.element_ty), mask=mask)
            tl.store(z + index, z_new.to(z.dtype.element_ty), mask=mask)


class KernelState:
    """Per-group bookkeeping: the address tables the kernels walk, and the step's own scratch."""

    def __init__(self, device: torch.device):
        self.device = device
        self.scalars = torch.zeros(16, dtype=torch.float32, device=device)
        self._pinned = torch.empty(16, dtype=torch.float32, pin_memory=True)
        self.rng_step = 0  # the seed carries the step, so no offset ever has to be revisited
        self._scratch: dict[int, dict[str, torch.Tensor]] = {}
        self._tables: dict[str, torch.Tensor] | None = None
        self._grads: list[int] = []

    def scratch(self, index: int, rows: int, cols: int) -> dict[str, torch.Tensor]:
        """One parameter's row-block partials and its two accumulators, allocated on first use."""
        entry = self._scratch.get(index)
        chunks = -(-rows // BLOCK_R)
        if entry is None or entry["pc_partial"].numel() != chunks * cols:
            entry = {
                "pc_partial": torch.zeros(chunks * cols, dtype=torch.float32, device=self.device),
                "dot": torch.zeros(1, dtype=torch.float32, device=self.device),
                "s_abs": torch.zeros(1, dtype=torch.float32, device=self.device),
            }
            self._scratch[index] = entry
        return entry

    def tables(self, optimizer, params: list[torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """The parameter/grad/state addresses, rebuilt when the gradients moved (a new tensor every step).

        Only the addresses are kept: holding the gradient tensors themselves would keep the previous step's gradients
        alive past `zero_grad(set_to_none=True)`, which is 2 B/param of memory the table does not need. `None` means the
        state is not in the dtypes the kernels take (see `step_megakernel`, which repairs the one case that occurs).
        """
        grad_ptrs = [p.grad.data_ptr() for p in params]
        if self._tables is not None and grad_ptrs == self._grads:
            return self._tables
        self._grads = grad_ptrs
        device = self.device
        entries: dict[str, list[int]] = {
            name: []
            for name in (
                "t_param",
                "t_grad",
                "t_z",
                "t_p0",
                "t_s",
                "t_pc",
                "t_pr",
                "t_pc_partial",
                "t_dot",
                "t_s_abs",
                "t_rng_base",
                "t_rows",
                "t_cols",
                "t_flip",
            )
        }
        entries["t_param"] = [p.data_ptr() for p in params]
        entries["t_grad"] = grad_ptrs
        max_numel = max(p.numel() for p in params)
        for index, p in enumerate(params):
            state = optimizer.state[p]
            rows, cols = p.shape
            dr, _ = state["exp_avg_sq_metadata"]
            flip = 1 if int(dr) == 1 else 0
            # `pc` is the factor over the columns; it is `exp_avg_sq_col` when the reference code flips the roles
            row, col = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
            pc, pr = (col, row) if flip else (row, col)
            for name, tensor, dtype in (
                # `z` follows the parameter, the subsamples and the second moment do not
                ("z", state["z"], p.dtype),
                ("p0", state.get("p0"), torch.bfloat16),
                ("s", state.get("s"), torch.bfloat16),
                ("exp_avg_sq_row", row, torch.float32),
                ("exp_avg_sq_col", col, torch.float32),
            ):
                if tensor is not None and tensor.dtype != dtype:
                    return None  # not what the kernels take; the per-parameter loop handles this state
            scratch = self.scratch(index, rows, cols)
            # `p0`/`s` are dropped once `prodigy_steps` is reached; the kernel does not read them then, so any pointer
            # will do, and the accumulator is always allocated
            unused = scratch["dot"].data_ptr()
            entries["t_z"].append(state["z"].data_ptr())
            entries["t_p0"].append(state["p0"].data_ptr() if "p0" in state else unused)
            entries["t_s"].append(state["s"].data_ptr() if "s" in state else unused)
            entries["t_pc"].append(pc.data_ptr())
            entries["t_pr"].append(pr.data_ptr())
            entries["t_pc_partial"].append(scratch["pc_partial"].data_ptr())
            entries["t_dot"].append(scratch["dot"].data_ptr())
            entries["t_s_abs"].append(scratch["s_abs"].data_ptr())
            # a per-parameter offset base, so two parameters never draw the same dither for the same element
            entries["t_rng_base"].append(index * max_numel)
            entries["t_rows"].append(rows)
            entries["t_cols"].append(cols)
            entries["t_flip"].append(flip)
        tables = {name: torch.tensor(values, dtype=torch.int64, device=device) for name, values in entries.items()}
        self._tables = tables
        return tables


def megakernel_supported(group) -> bool:
    """The step of the current training config: schedule-free, Adam-atan2, factored fp32 second moment, no clip."""
    params = group["params"]
    if not group.get("use_megakernel", True):
        return False
    if not (
        group["use_schedulefree"]
        and group["eps"] is None
        and not group["use_stableadamw"]
        and group["factored"]
        and group["factored_fp32"]
        and not group["use_bias_correction"]
        and group["weight_decay_by_lr"]
        and not (group["use_focus"] or group["use_cautious"] or group["use_grams"] or group["use_adopt"])
        and not (group["use_orthograd"] or group["use_speed"] or group["fused_back_pass"])
    ):
        return False
    if not params or params[0].device.type != "cuda" or params[0].dtype not in (torch.bfloat16, torch.float32):
        return False
    dtype = params[0].dtype
    return all(
        len(p.shape) == 2
        and p.dtype == dtype
        and p.is_contiguous()
        and p.device == params[0].device
        and p.grad is not None
        and p.grad.dtype == dtype
        and p.grad.shape == p.shape
        and p.grad.is_contiguous()
        for p in params
    )


@torch.no_grad()
def step_megakernel(optimizer, group) -> bool:
    """One step in three Triton launches. False when the fast path does not apply."""
    if not megakernel_supported(group):
        return False
    params = group["params"]
    if not group["train_mode"]:
        raise Exception("Not in train mode!")

    state: KernelState | None = group.get("megakernel_state")
    if state is None:
        state = KernelState(params[0].device)
        for p in params:
            per_param = optimizer.initialise_state(p, group)  # `z`, `p0`, `s` and the factored second moment
            for name in ("exp_avg_sq_row", "exp_avg_sq_col"):
                # a state dict is reloaded in the parameter's dtype, so `factored_fp32` factors come back bf16; the step
                # works in fp32 and `factored_fp32` asks for fp32, so restore it rather than lose the mantissa
                if per_param[name].dtype != torch.float32:
                    per_param[name] = per_param[name].float()
        group["megakernel_state"] = state

    # the per-step scalars, in one tensor so nothing re-specializes the kernels
    d, d0 = group["d"], group["d0"]
    beta1, beta2, beta3 = optimizer.get_betas(group)
    dlr = optimizer.get_dlr(group)
    weight = dlr**2
    weight_sum = group["running_weight_sum"] = group.get("weight_sum", 0) + weight
    ckp1 = weight / weight_sum if weight_sum else 0
    schedulefree_c = group.get("schedulefree_c", 0)
    if schedulefree_c > 0:
        ckp1 = min(1.0, ckp1 * (1 - beta1) * schedulefree_c)
    xy_step = 1 - beta1 * (1 - ckp1)
    group["effective_lr"] = group["lr"] * xy_step

    values = [0.0] * 16
    values[_S_DLR], values[_S_D] = dlr, d
    values[_S_BETA2], values[_S_BETA3] = beta2, beta3
    values[_S_CKP1], values[_S_XY] = ckp1, xy_step
    values[_S_DECAY] = optimizer.get_weight_decay(group)
    values[_S_DUPDATE] = (d**2) / (d0**0.5)
    values[_S_VALUE2] = (1 - beta2) * d**2
    state._pinned.copy_(torch.tensor(values, dtype=torch.float32))
    state.scalars.copy_(state._pinned, non_blocking=True)

    running_numerator, running_denom = optimizer.get_running_values_for_group(group)
    k, prodigy_steps = group["k"], group["prodigy_steps"]
    do_stats = prodigy_steps <= 0 or k < prodigy_steps
    n_params = len(params)
    tables = state.tables(optimizer, params)
    if tables is None:
        return False
    # the dither is what makes a bf16 store unbiased; an fp32 parameter is written exactly
    param_fp32 = params[0].dtype == torch.float32
    stochastic = group["stochastic_rounding"] and not param_fp32
    seed = torch.cuda.initial_seed()

    _stats_kernel[(n_params,)](
        tables["t_grad"],
        tables["t_z"],
        tables["t_p0"],
        tables["t_s"],
        tables["t_pc_partial"],
        tables["t_pr"],
        tables["t_rows"],
        tables["t_cols"],
        tables["t_dot"],
        tables["t_s_abs"],
        state.scalars,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        SLICE_P=_SLICE_P,
        DO_STATS=do_stats,
        PARAM_FP32=param_fp32,
        num_warps=4,
    )
    _finalize_kernel[(n_params + 1,)](
        tables["t_pc"],
        tables["t_pc_partial"],
        tables["t_rows"],
        tables["t_cols"],
        tables["t_dot"],
        tables["t_s_abs"],
        running_numerator,
        running_denom,
        state.scalars,
        n_params,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        DO_STATS=do_stats,
        num_warps=4,
    )
    _update_kernel[(n_params,)](
        tables["t_param"],
        tables["t_grad"],
        tables["t_z"],
        tables["t_pc"],
        tables["t_pr"],
        tables["t_rows"],
        tables["t_cols"],
        tables["t_flip"],
        tables["t_rng_base"],
        state.scalars,
        seed + state.rng_step,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        STOCHASTIC=stochastic,
        PARAM_FP32=param_fp32,
        num_warps=4,
    )
    if stochastic:
        state.rng_step += 1
    return True
