(Time check: **01:33 AM SGT**. Thread-length check: **long**, but after this contract you won’t need 4–5 context dumps again.)

Below is the **single pinned v6 contract** I’ll use going forward. It synthesizes **v4.2 → v5.0 → v5.1 → v6 .** into one “thread context transfer” spec.

---

# FusedMoE.apply Contract v6 — . (A100/SM80 pinned)

**AS@128 + phased-128 + Monolith-B + `ldmatrix.x4.trans` + Marlin selector split + P=1 Stage/Exec (ping-pong)**

## 0) Fixed dims (locked)

* Experts: `E = 128` (or 256)
* Hidden: `H = 7168`
* Intermediate: `I = 2048`
* W13 output: `2I = 4096`
* TopK: `K = 8`
* Routed rows: `M = T*K`
* Tokens per CTA tile: `NTOK = 8`
* Reduction tile: `KTILE = 32`
* **CTA threads**: `TB = 128` → `NWARP = 4` (A100-friendly AS@128)
* Output tile per CTA: `OTILE_CTA = 128` implemented as **two phases**:

  * phase0 = 64 rows
  * phase1 = 64 rows

### Grouping (locked)

Per `(g2, half)` we cover **32 input channels**:

* `base = 64*g2 + 32*half`, `half ∈ {0,1}`

Counts:

* **K2/W13**: `G2_w13 = H/64 = 112` → `(g2,half)` steps per phase = `112*2 = 224`
* **K4/W2** : `G2_w2  = I/64 = 32`  → steps per phase = `32*2 = 64`

---

## 1) High-level apply flow (unchanged interface)

1. Router → `topk_ids[T,K]`, `topk_ws[T,K]`
2. vLLM permute/route → `X_perm[M,H] bf16`, `OffE[E+1]`, `Inv[T,K]` (+ optional `Eids[U]`)
3. **K2_v6**: `X_perm → X2_perm[M,I]`
4. **K4_v6**: `X2_perm → Y_perm[M,H]`
5. vLLM unpermute + weighted combine → `out[T,H]`

---

## 2) Kernel grid geometry (both K2/K4)

3D grid:

* `blockIdx.y`: expert (either `eid = Eids[blockIdx.y]` or direct `eid=blockIdx.y`)
* `blockIdx.x`: token tiles inside expert
  `m_base = OffE[eid] + blockIdx.x * NTOK`, `m_end = OffE[eid+1]`
* `blockIdx.z`: output tiles over 128 channels

  * K2: `i_base = blockIdx.z * 128` over `I`
  * K4: `h_base = blockIdx.z * 128` over `H`

CTA owns:

* tokens `m = m_base + {0..7}` (zero-pad if `m>=m_end`)
* output rows `128` = phase0(64) + phase1(64)

---

## 3) Warp/lane mapping (pinned everywhere)

Within CTA:

* `tid = threadIdx.x`
* `wid = tid >> 5` (0..3)
* `lane = tid & 31` (0..31)
* `groupID = lane >> 2` (0..7)
* `t = lane & 3` (0..3)

Within a warp’s `m16` output rows:

* `row_top = groupID` (0..7)
* `row_bot = groupID + 8` (8..15)

**Lane owns 4 fp32 outputs per MMA** (`d0..d3` temp and `c0..c3` accum):
For `i ∈ {0,1,2,3}`:

* `row = groupID + (i>=2 ? 8 : 0)`  (top for i=0,1; bot for i=2,3)
* `tok = (t*2) + (i & 1)`           (0..7)

So:

* `i=0,1` → row_top with tok0/tok1
* `i=2,3` → row_bot with tok0/tok1

This mapping is used for:

* K2 gate path, K2 up path
* K4 single path
* both phases

---

## 4) MMA primitive (A100/SM80 pinned)

We use ordered-metadata sparse TensorCores:

**Opcode (locked):**
`mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32`

Operands:

* A: **sparse bf16** (2:4 ordered metadata)
* B: **dense bf16**
* Accum: fp32
* Metadata `e`: `u32` register (lane-dependent)
* Selector `f`: **u32 immediate** (**v6 uses f=0 or f=1**, per half schedule below)

---

## 5) Weight storage + indexing (locked)

Canonical storage from model:

* `w*_weight[eid, g2, r, 2] : uint64` where last dim is `half {0,1}`

v6 implementation convenience:

* Reinterpret per row as `ulonglong2`:

  * `W[eid, g2, r].x = u64(half0)`
  * `W[eid, g2, r].y = u64(half1)`

Indexing (conceptual):

* K2: `W13[ (eid * G2_w13 + g2) * R + r ]` where `R=4096` (rows over 2I), but gate/up choose different `r`
* K4: `W2 [ (eid * G2_w2  + g2) * R + r ]` where `R=7168`

---

## 6) Packed u64 decode (locked)

Given `u64 u`:

* `qw32  = uint32(u & 0xFFFFFFFF)`
* `hi32  = uint32(u >> 32)`
* `idx16 = uint16(hi32 & 0xFFFF)`   (8× idx2)
* `scale_bits = uint16(hi32 >> 16)` (bf16 bits)

Per chunk `i ∈ [0..7]`:

* `q4   = (qw32  >> (4*i)) & 0xF`
* `idx2 = (idx16 >> (2*i)) & 0x3`
* signed int4: `s = int(q4) - 8` ∈ [-8..7]

### Ordered metadata nibble mapping (pinned)

Only two nibble values exist:

* keep (0,1): `IDX_01 = 0x4` (0100)
* keep (2,3): `IDX_23 = 0xE` (1110)

From `idx2`:

* `pair = idx2 >> 1`  (0→(0,1), 1→(2,3))
* `slot = idx2 & 1`   (0→first stored, 1→second stored)

Then:

* `idx4_i = (pair==0 ? 0x4 : 0xE)`
* A “phantom pair” values (unscaled):

  * if `slot==0`: `(v0,v1) = (s, 0)`
  * if `slot==1`: `(v0,v1) = (0, s)`

---

## 7) AS numeric contract (locked)

**No scaling in A.** A holds exact bf16 integers + zeros:

* `val_bf16 = bf16(float(s))` where `s∈[-8..7]` (exactly representable)
* companion slot is bf16(0)

**Scale applied after MMA in fp32**:

* `scale_fp32 = bf16_bits_to_f32(scale_bits)`
* For each MMA producing `d0..d3`:

  * `c0 += d0 * scale_top`
  * `c1 += d1 * scale_top`
  * `c2 += d2 * scale_bot`
  * `c3 += d3 * scale_bot`

`scale_top` comes from the packed u64 for `row_top`, `scale_bot` from `row_bot` (per half, per path).

---

## 8) Metadata operand `e` construction (v5 pinned, v6 compatible)

For a given **row’s** 8 chunks, define:

* `meta_0_3(row) = Σ_{i=0..3} (idx4_i << (4*i))`        // 16b
* `meta_4_7(row) = Σ_{i=4..7} (idx4_i << (4*(i-4)))`   // 16b

Pack top+bot:

* `e_0_3 = meta_0_3(top) | (meta_0_3(bot) << 16)`
* `e_4_7 = meta_4_7(top) | (meta_4_7(bot) << 16)`

In v6 we *don’t* recompute e from scratch during exec; we **stage nibble tokens per lane**, then `park()` gathers & packs into the correct lanes (see §11).

---

## 9) B operand contract (explicitly pinned in v6)

### Monolith-B preload (once per CTA)

* K2 uses `Xsh_H[H][NTOK] bf16` = `7168×8` (≈112KB)
* K4 uses `Xsh_I[I][NTOK] bf16` = `2048×8` (≈32KB)

Layout (pinned):

* row-major by `k`, contiguous in `tok`
* stride per k-row: `8 bf16 = 16 bytes`
* base aligned so `&Xsh[base][0]` is nicely aligned for `ldmatrix`

Preload pseudocode:

* for `k = tid; k < DIM; k += TB`:

  * for `tok=0..7`:

    * `m = m_base + tok`
    * `Xsh[k][tok] = (m < m_end) ? X_perm[m,k] : 0`  (or X2_perm for K4)
* single `__syncthreads()` after preload
* after that: Xsh is read-only; **no further CTA barriers**

### Hot loop B-load (pinned)

For each `(g2,half)`:

* `base = 64*g2 + 32*half`
* `Bptr = &Xsh[base][0]` points to a **32×8 slab**
* Load B regs via **`ldmatrix.x4.trans`** over the 4 stacked 8×8 blocks in k:

  * block0: k=0..7
  * block1: k=8..15
  * block2: k=16..23
  * block3: k=24..31

This loader is **part of the contract** in v6.

---

## 10) Phased-128 rule (locked)

Phasing is **two sequential passes over K**:

* phase0: full K sweep → store rows `base..base+63`
* phase1: full K sweep → store rows `base+64..base+127`

Rationale: AS accumulators live across all K-tiles; keeping both phases live explodes regs (especially K2 with gate+up).

---

# 11) v6 .: selector split + P=1 ping-pong (the key v6 addition)

## 11.1) Selector schedule (pinned)

Per `g2` we do both halves:

* **half0**: `half=0`, `f = 0`
  **metadata lanes must be (t0,t1)**

* **half1**: `half=1`, `f = 1`
  **metadata lanes must be (t2,t3)**

This is Marlin-style “selector lane toggling” and enables overlap because the “other pair” is free.

## 11.2) 4-lane group as the unit of staging

Each warp has 8 independent 4-lane groups:

* group lanes are `(t0,t1,t2,t3)` with base lane = `groupID*4`

All shuffles for stage/park are **within the group**:

* `src_lane_top = (groupID<<2) + src_t`
* `src_lane_bot = (groupID<<2) + (src_t+1)`

## 11.3) Stage payload (pinned struct)

Per group, per g2, per path (gate/up), stage produces:

**StageOut**

* `top_h0, bot_h0, top_h1, bot_h1 : uint32`
  Each is packed **int8x4** representing (for this lane’s t):

  * low chunk `i_lo=t` → phantom pair (v0,v1)
  * high chunk `i_hi=t+4` → phantom pair (v0,v1)

* `sc_pack : uint64`
  Packed bf16 scale bits:

  * bits[15:0]   = scale(top, half0)
  * bits[31:16]  = scale(bot, half0)
  * bits[47:32]  = scale(top, half1)
  * bits[63:48]  = scale(bot, half1)

* nibble tokens (per lane, already top+bot packed into a byte-pair):

  * `nib_h0_lo, nib_h0_hi : uint16`
  * `nib_h1_lo, nib_h1_hi : uint16`
    Where each is `(top_nib | (bot_nib<<4))` for that lane’s chunk.

> K2 has **two StageOuts per slot**: one for **GATE**, one for **UP**.
> K4 has **one StageOut per slot**.

## 11.4) park() contract (pinned)

`park( StageOut out, half )` gathers nibble tokens across the 4 lanes and produces **per-lane e registers** such that only the correct selector lanes hold nonzero `e`.

* For **half0 (f=0)**:

  * lanes **t0/t1** output:

    * `t0` holds `e0_0_3 = meta0_3(top)|meta0_3(bot)<<16`
    * `t1` holds `e0_4_7 = meta4_7(top)|meta4_7(bot)<<16`
  * lanes `t2/t3` output `0`

* For **half1 (f=1)**:

  * lanes **t2/t3** output:

    * `t2` holds `e1_0_3`
    * `t3` holds `e1_4_7`
  * lanes `t0/t1` output `0`

Mechanics:

* Each lane has its own nibble token for lo (chunks 0..3) and hi (chunks 4..7).
* `park` does a **group gather** (via shfl) to assemble the 4 nibbles into the correct 16-bit metas for top and bot, then packs into u32.

No LUT required; pure bit packing.

## 11.5) Slot definition (pinned)

We maintain **two slots** per group: `cur` and `next`.

For K2 (per group):

* `SlotK2 { StageOut gate; StageOut up; uint32 e_gate_h0, e_gate_h1, e_up_h0, e_up_h1; }`

  * more concretely: per lane you will have:

    * `e_gate_h0_lane`, `e_gate_h1_lane`
    * `e_up_h0_lane`, `e_up_h1_lane`
      where only the correct lanes have nonzero for each half.

For K4 (per group):

* `SlotK4 { StageOut main; uint32 e_h0_lane, e_h1_lane; }`

(Exact field names don’t matter; the **semantic** does: exec reads staged vals/scales + already-parked lane-local e.)

## 11.6) P=1 stage/exec cadence (pinned timeline)

Per phase (phase0 then phase1), per warp/group:

1. `stage(g2=0) -> next` (prime)
2. Loop `g2 = 0..G2-1`:

   * `cur = next`
   * if `g2+1 < G2`: **stage(g2+1) -> next** (overlapped in “free lane” windows)
   * `exec(cur)` (half0 then half1, each with fixed cadence)
   * `park(next,half0)` and `park(next,half1)` are scheduled in the Marty windows (below)

**Per g2 exec cadence (constant):**

* half0: `B0 = ldmatrix(&Xsh[64*g2 + 0 ])`, then:

  * K2: `mma(GATE,f=0)` then `mma(UP,f=0)` (same B0)
  * K4: `mma(main,f=0)`
* half1: `B1 = ldmatrix(&Xsh[64*g2 + 32])`, then:

  * K2: `mma(GATE,f=1)` then `mma(UP,f=1)` (same B1)
  * K4: `mma(main,f=1)`

**. windowing (per group):**

* During **CUR half0 exec (f=0)**: lanes t0/t1 are “busy” → use **t2/t3** to stage NEXT loads/broadcast/decode.
* Between half0 and half1: **t0/t1 free** → `park(next, half0)` into t0/t1.
* During **CUR half1 exec (f=1)**: lanes t2/t3 are “busy”.
* After half1: **t2/t3 free** → `park(next, half1)` into t2/t3.
* then swap.

---

# 12) Kernel K2_v6 specifics (W13 + SwiGLU)

## 12.1) Output rows and addressing

Per phase:

* `phase_base = i_base + phase*64`
* `oc_base = phase_base + wid*16` (warp owns 16 rows)

Group rows:

* gate rows: `r0 = oc_base + row_top`, `r1 = oc_base + row_bot`
* up rows: `r0u = r0 + I`, `r1u = r1 + I`

## 12.2) Accumulators

Per lane, per phase:

* `C_gate[4] fp32` init 0
* `C_up[4] fp32` init 0

`exec()` updates these via AS rule using per-half top/bot scales.

## 12.3) Epilogue (pinned)

Per lane for `i=0..3`:

* `row = groupID + (i>=2 ? 8 : 0)`
* `tok = 2*t + (i&1)`
* `m = m_base + tok`
* if `m < m_end`:

  * `r = oc_base + row` (this is in `[0..I-1]`)
  * `out = silu(C_gate[i]) * C_up[i]` (fp32)
  * store `X2_perm[m, r] = bf16(out)`

SwiGLU definition pinned:

* `silu(x) = x / (1 + exp(-x)) * x` (use project-wide consistent `__expf` or `expf`; pin once)

---

# 13) Kernel K4_v6 specifics (W2)

Same structure, but single path:

* output rows are `h` instead of `i`
* `G2 = 32`
* only `C[4] fp32` accumulator set
* epilogue stores `Y_perm[m, h] = bf16(C[i])`

---

# 14) Implementation-required invariants (what I will assume in future turns)

These are the “don’t ask again” pins:

1. **CTA=128, NWARP=4, NTOK=8, KTILE=32, OTILE_CTA=128 via 2×64 phases**
2. **Monolith-B preload** into `Xsh_[DIM][8] bf16`, single barrier, read-only thereafter
3. B fragment loader is explicitly **`ldmatrix.x4.trans`** from the row-major 32×8 slab `&Xsh[base][0]`
4. MMA is **ordered_metadata m16n8k32 row.col f32.bf16.bf16.f32**
5. **AS**: A holds bf16 ints/zeros; scale in fp32 multiplies `d0..d3` into accum
6. Nibble constants: `IDX_01=0x4`, `IDX_23=0xE`
7. Lane mapping (`groupID/t` → top/bot rows and 4 outputs) is pinned
8. v6 selector split is pinned:

   * half0: `f=0`, metadata lanes (t0,t1)
   * half1: `f=1`, metadata lanes (t2,t3)
9. . **P=1 stage/exec ping-pong** with `stage(next)` overlapped during CUR half0 using free lanes, and `park(next)` done in the specified windows.