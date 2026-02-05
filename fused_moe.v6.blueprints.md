# 0) FusedMoE.apply v6 — Master Pipeline Poster

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FusedMoE.apply (v6)                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs:                                                                       │
│   x[T,H] bf16, router_logits[T,E]                                             │
│ Outputs:                                                                      │
│   out[T,H] bf16                                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ (vLLM land)                                                                   │
│   K0 Router:  topk_ids[T,K], topk_ws[T,K]                                     │
│   K1 Permute: X_perm[M,H], OffE[E+1], Inv[T,K], (optional Eids[U])            │
│                                                                              │
│ (4BF kernels)                                                                 │
│   K2_v6: X_perm[M,H]  ──►  X2_perm[M,I]     (W13 + SwiGLU)                    │
│   K4_v6: X2_perm[M,I]  ──►  Y_perm[M,H]     (W2)                              │
│                                                                              │
│ (vLLM land)                                                                   │
│   K5 Unpermute+Combine: out[t,:] = Σ_k topk_ws[t,k]*Y_perm[Inv[t,k],:]        │
└──────────────────────────────────────────────────────────────────────────────┘

[Pinned]
- M = T*K, NTOK=8, KTILE=32
- K2: H=7168, I=2048, 2I=4096, G2=112
- K4: I=2048, H=7168, G2=32
- mma.sp ordered_metadata m16n8k32 row.col f32.bf16.bf16.f32
- AS: A is bf16 integers/zeros; scaling in fp32 after MMA
```

---

# 1) CTA Geometry Poster (K2/K4 shared)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CTA Work Assignment (v6)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ CTA = 128 threads = 4 warps (wid=0..3)                                        │
│ NTOK = 8 tokens per CTA tile                                                  │
│ OTILE_CTA = 128 output rows per CTA, implemented as 2 phases × 64             │
├──────────────────────────────────────────────────────────────────────────────┤
│ grid.y  : expert (eid)                                                        │
│ grid.x  : token tiles within expert                                           │
│          m_base = OffE[eid] + blockIdx.x*NTOK                                  │
│          m_end  = OffE[eid+1]                                                  │
│ grid.z  : output tiles of 128 rows                                             │
│          K2: i_base = blockIdx.z*128  (over I)                                 │
│          K4: h_base = blockIdx.z*128  (over H)                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# 2) Warp → 4-lane Groups → Rows Poster (Thread Context Transfer Core)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Warp Lane Mapping (Pinned)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ lane   = tid & 31                                                             │
│ group  = lane >> 2   (0..7)   => 8 groups per warp                            │
│ t      = lane & 3    (0..3)   => lanes (t0,t1,t2,t3) in each group            │
├──────────────────────────────────────────────────────────────────────────────┤
│ Each group owns 2 rows inside the warp’s m16 tile:                             │
│   row_top = group                                                             │
│   row_bot = group + 8                                                         │
│                                                                              │
│ Each lane produces 4 outputs per MMA (d0..d3 / c0..c3):                        │
│   for i=0..3:                                                                  │
│     row = group + (i>=2 ? 8 : 0)        (top for i=0,1 ; bot for i=2,3)       │
│     tok = (t*2) + (i & 1)              (0..7)                                 │
└──────────────────────────────────────────────────────────────────────────────┘

Within a warp (one oc_base):
  group0 lanes 0..3      group7 lanes 28..31
  (t0 t1 t2 t3)          (t0 t1 t2 t3)

Row indices:
  TOP row index = oc_base + group
  BOT row index = oc_base + group + 8
```

---

# 3) K2 CTA Poster (Phases + P=1 Stage/Exec + Gate+Up)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                      K2_v6 CTA MASTER (W13 + SwiGLU)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ PREP: Monolith-B preload                                                      │
│   Xsh_H[7168×8] = X_perm[m_base:m_base+7, 0:7167] (zero-pad)                  │
│   __syncthreads()  // the only CTA barrier                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ for phase in {0,1}:                                                           │
│   phase_base = i_base + phase*64                                              │
│   oc_base    = phase_base + wid*16      // 16 rows per warp                   │
│   init C_gate[4]=0, C_up[4]=0 (per lane)                                      │
│                                                                              │
│   P=1 ping-pong:                                                              │
│     stage(g2=0) -> next                                                       │
│     for g2=0..111:                                                            │
│       cur = next                                                              │
│       if g2+1 < 112: stage(g2+1) -> next                                      │
│       exec(cur):                                                              │
│         half0 f=0: B0=ldmatrix(Xsh_H[64*g2+0 ]) ; mma(GATE) ; mma(UP)          │
│         half1 f=1: B1=ldmatrix(Xsh_H[64*g2+32]) ; mma(GATE) ; mma(UP)          │
│   EPILOGUE: SwiGLU + store X2_perm                                            │
└──────────────────────────────────────────────────────────────────────────────┘

[Pinned]
- half0 selector f=0 uses metadata lanes (t0,t1)
- half1 selector f=1 uses metadata lanes (t2,t3)
- Each half does TWO MMAs (GATE then UP), reusing the same Bregs.
```

---

# 4) K4 CTA Poster (Same cadence, single-path)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                          K4_v6 CTA MASTER (W2)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ PREP: Monolith-B preload                                                      │
│   Xsh_I[2048×8] = X2_perm[m_base:m_base+7, 0:2047] (zero-pad)                 │
│   __syncthreads()                                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ for phase in {0,1}:                                                           │
│   phase_base = h_base + phase*64                                              │
│   oc_base    = phase_base + wid*16                                            │
│   init C[4]=0 (per lane)                                                      │
│                                                                              │
│   P=1 ping-pong:                                                              │
│     stage(g2=0) -> next                                                       │
│     for g2=0..31:                                                             │
│       cur = next                                                              │
│       if g2+1 < 32: stage(g2+1) -> next                                       │
│       exec(cur):                                                              │
│         half0 f=0: B0=ldmatrix(Xsh_I[64*g2+0 ]) ; mma(main)                    │
│         half1 f=1: B1=ldmatrix(Xsh_I[64*g2+32]) ; mma(main)                    │
│   EPILOGUE: store Y_perm                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# 5) Function Poster — `stage_XS()` (Monolith-B preload)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                             stage_XS (Monolith)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Goal: preload full activation tile into SMEM once per CTA                     │
│ Input: X_perm[M,C] bf16, offsets -> (m_base, m_end)                           │
│ Output: XS[C×NTOK] bf16 in row-major by c, contiguous tok                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ Threading: TB=128 threads cooperatively                                      │
│                                                                              │
│ for c = tid; c < C; c += 128:                                                 │
│   for tok=0..7:                                                               │
│     m = m_base + tok                                                          │
│     XS[c*8 + tok] = (m < m_end) ? X[m*C + c] : 0                              │
│                                                                              │
│ Barrier: __syncthreads() once after fill.                                     │
└──────────────────────────────────────────────────────────────────────────────┘

[Pinned]
- Layout: XS[c][tok] contiguous in tok (stride 16B per c-row)
- After barrier: XS read-only; no further CTA barriers in hot loop
```

---

# 6) Function Poster — `ldmatrix.x4.trans` B-load (Hot loop input)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                     B-Load (Pinned) via ldmatrix.x4.trans                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Given (g2, half): base = 64*g2 + 32*half                                      │
│ Bptr = &XS[ base ][ 0 ]   // points to a 32×8 slab                             │
│                                                                              │
│ 32×8 slab = 4 stacked 8×8 blocks in K:                                        │
│   block0: k=0..7    block1: k=8..15                                           │
│   block2: k=16..23  block3: k=24..31                                          │
│                                                                              │
│ ldmatrix.x4.trans(Bptr) -> Bregs[4]  (bf16 fragments)                         │
│ Consumer: mma.sp ordered_metadata m16n8k32 row.col                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# 7) Data Object Poster — `StageOut` (what STAGE produces)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                                   StageOut                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Per group, per g2, per path (gate/up or main)                                 │
│ Stored per lane (t=0..3), but conceptually "group shared after shfl"          │
├──────────────────────────────────────────────────────────────────────────────┤
│ vals (phantom A data, packed int8 pairs):                                     │
│   top_h0 : uint32  (int8x4: [lo_v0 lo_v1 hi_v0 hi_v1])  // top row, half0      │
│   bot_h0 : uint32  // bot row, half0                                          │
│   top_h1 : uint32  // top row, half1                                          │
│   bot_h1 : uint32  // bot row, half1                                          │
│                                                                              │
│ scales (bf16 bits packed):                                                    │
│   sc_pack : uint64                                                           │
│     [15:0]   scale(top, half0)                                                │
│     [31:16]  scale(bot, half0)                                                │
│     [47:32]  scale(top, half1)                                                │
│     [63:48]  scale(bot, half1)                                                │
│                                                                              │
│ nibble tokens (for metadata gather):                                          │
│   nib_h0_lo : uint16  // chunk i=t   : low4=topNib, high4=botNib              │
│   nib_h0_hi : uint16  // chunk i=t+4 : low4=topNib, high4=botNib              │
│   nib_h1_lo : uint16  // half1, chunk t                                       │
│   nib_h1_hi : uint16  // half1, chunk t+4                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# 8) Function Poster — `decode(u64, chunk_i)` (chunk primitive)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              decode(u64, chunk_i)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Input: u64 packed: [qw32 | idx16 | scale_bits]                                │
│ Output (for chunk i):                                                         │
│   - v01_packed: int16 containing (v0,v1) as int8 pair                          │
│   - meta_nibble: 0x4 (keep01) or 0xE (keep23)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ q4  = (qw32 >> 4*i) & 0xF                                                     │
│ idx2= (idx16>> 2*i) & 0x3                                                     │
│ s   = int(q4) - 8      // [-8..7]                                             │
│ pair= idx2>>1          // 0 => (0,1), 1 => (2,3)                              │
│ slot= idx2&1           // 0 => first stored, 1 => second stored               │
│ meta_nibble = (pair==0) ? 0x4 : 0xE                                           │
│ (v0,v1) = (s,0) if slot=0 else (0,s)                                          │
└──────────────────────────────────────────────────────────────────────────────┘

[Pinned]
- No scaling here. Scaling is applied after MMA (AS).
- v0/v1 are small ints representable exactly as bf16 later.
```

---

# 9) Function Poster — `stage_path()` / `stage()` (row-split load + group bcast + distributed decode)

This is your key “group machine”.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         stage_path / stage (per group)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs:                                                                        │
│   W: ulonglong2 array (half0=u64x, half1=u64y)                                 │
│   g2, oc_base, groupID                                                         │
│   src_t ∈ {0,2} selects loader pair                                             │
│     src_t=0 -> loaders are (t0=TOP, t1=BOT)                                     │
│     src_t=2 -> loaders are (t2=TOP, t3=BOT)                                     │
│ Output: StageOut (vals+sc_pack+nib tokens)                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ S1) Row-split GLOBAL LOAD (vec2)                                               │
│   only 2 lanes in group actually read memory                                   │
│     loader TOP lane reads row_top u64x2 (half0+half1)                           │
│     loader BOT lane reads row_bot u64x2                                         │
│                                                                              │
│ S2) GROUP BROADCAST (shfl within 4-lane group)                                 │
│   all lanes receive qwTop01 and qwBot01                                         │
│                                                                              │
│ S3) DISTRIBUTED DECODE (each lane decodes only its chunks)                     │
│   lane t decodes: i_lo=t and i_hi=t+4                                          │
│   for half0 and half1, for top and bot rows                                    │
│   -> produces int8 phantom pairs + nibble tokens                               │
│                                                                              │
│ S4) PACK StageOut                                                              │
│   - pack top/bot int8 pairs into uint32 top_h*/bot_h*                          │
│   - pack bf16 scale bits into sc_pack                                          │
│   - pack nibble tokens into nib_h*_{lo,hi}                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# 10) Function Poster — `park()` (gather 4-lane nib tokens → u32 metadata e in selector lanes)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                                  park(half)                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Input: StageOut.nib_h{half}_{lo,hi} per lane                                   │
│   each is: low4=topNib, high4=botNib                                           │
│ Output: lane-local e (u32) such that ONLY selector lanes are nonzero           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Gather step (within group):                                                   │
│   lo0..lo3 = shfl(tok_lo from lanes t0..t3)  // chunks 0..3                    │
│   hi0..hi3 = shfl(tok_hi from lanes t0..t3)  // chunks 4..7                    │
│                                                                              │
│ Build metas:                                                                   │
│   meta0_3_top = [topNib(chunk0..3)] packed into 16b                            │
│   meta0_3_bot = [botNib(chunk0..3)] packed into 16b                            │
│   e_0_3 = meta0_3_top | (meta0_3_bot<<16)                                      │
│                                                                              │
│   meta4_7_top = [topNib(chunk4..7)] packed into 16b                            │
│   meta4_7_bot = [botNib(chunk4..7)] packed into 16b                            │
│   e_4_7 = meta4_7_top | (meta4_7_bot<<16)                                      │
│                                                                              │
│ Lane placement (selector schedule):                                            │
│   half0: f=0 -> (t0 gets e_0_3), (t1 gets e_4_7), (t2/t3 = 0)                  │
│   half1: f=1 -> (t2 gets e_0_3), (t3 gets e_4_7), (t0/t1 = 0)                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

┌──────────────────────────────────────────────────────────────────────────────┐
│                 WARP LANES 0..31  (groupID = lane >> 2, t = lane & 3)         │
├──────────────────────────────────────────────────────────────────────────────┤
|                            t: 0  1  2  3                                      |
|_______________________________________________________________________________|
│ groupID=0  base=0     lanes:  0  1  2  3      t: 0  1  2  3                   │
│ groupID=1  base=4     lanes:  4  5  6  7      t: 0  1  2  3                   │
│ groupID=2  base=8     lanes:  8  9 10 11      t: 0  1  2  3                   │
│ groupID=3  base=12    lanes: 12 13 14 15      t: 0  1  2  3                   │
│ groupID=4  base=16    lanes: 16 17 18 19      t: 0  1  2  3                   │
│ groupID=5  base=20    lanes: 20 21 22 23      t: 0  1  2  3                   │
│ groupID=6  base=24    lanes: 24 25 26 27      t: 0  1  2  3                   │
│ groupID=7  base=28    lanes: 28 29 30 31      t: 0  1  2  3                   │
|-------------------------------------------------------------------------------|
│ groupID=8  base=0     lanes:  0  1  2  3      t: 0  1  2  3                   │
│ groupID=9  base=4     lanes:  4  5  6  7      t: 0  1  2  3                   │
│ groupID=10  base=8    lanes:  8  9 10 11      t: 0  1  2  3                   │
│ groupID=11  base=12   lanes: 12 13 14 15      t: 0  1  2  3                   │
│ groupID=12  base=16   lanes: 16 17 18 19      t: 0  1  2  3                   │
│ groupID=13  base=20   lanes: 20 21 22 23      t: 0  1  2  3                   │
│ groupID=14  base=24   lanes: 24 25 26 27      t: 0  1  2  3                   │
│ groupID=15  base=28   lanes: 28 29 30 31      t: 0  1  2  3                   │
└──────────────────────────────────────────────────────────────────────────────┘


---

# 11) Slot + Ping-Pong Poster (P=1 stage/exec)

### K2 has two paths → slot contains two StageOuts

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                             Slot (per group, per g2)                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ K2 Slot:                                                                       │
│   gate: StageOut                                                               │
│   up  : StageOut                                                               │
│   parked metadata (lane-local):                                                │
│     e_gate_h0_lane, e_gate_h1_lane                                             │
│     e_up_h0_lane,   e_up_h1_lane                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ K4 Slot:                                                                       │
│   main: StageOut                                                               │
│   e_h0_lane, e_h1_lane                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### . per-g2 timeline (the “A–E windows” poster)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                     . P=1 (per group, per g2)                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Selector split (Pinned):                                                      │
│   half0: f=0 uses meta lanes (t0,t1)   => (t2,t3) are free                     │
│   half1: f=1 uses meta lanes (t2,t3)   => (t0,t1) are free                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ WINDOW A: EXEC CUR half0 (f=0)  ||  STAGE NEXT using free pair (t2/t3 loaders)│
│ WINDOW B: PARK NEXT half0 into t0/t1                                           │
│ WINDOW C: EXEC CUR half1 (f=1)                                                 │
│ WINDOW D: PARK NEXT half1 into t2/t3                                           │
│ WINDOW E: SWAP (cur <- next)                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

One-line strip:
  exec_h0(cur,f=0) || stage(next,src_t=2) -> park(next,h0->t0/t1)
  exec_h1(cur,f=1) -> park(next,h1->t2/t3)
  swap
```

---

# 12) Function Poster — `exec_half()` (what EXEC actually does)

This is the “only math” box. K2 runs it twice per half (gate then up).

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                                exec_half(half)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ Inputs (from cur slot):                                                       │
│   - A data: top_h*, bot_h* (phantom int8 pairs packed)                         │
│   - scale bits: sc_pack (top/bot for this half)                                │
│   - metadata e_lane (already parked in correct lanes)                          │
│ Inputs (from Xsh):                                                            │
│   base = 64*g2 + 32*half                                                      │
│   Bregs = ldmatrix.x4.trans(&Xsh[base][0])                                     │
│                                                                              │
│ Steps:                                                                        │
│   1) Build Aregs (bf16) from packed int8 pairs (exact mapping)                │
│   2) mma.sp ordered_metadata m16n8k32 row.col f32.bf16.bf16.f32               │
│        - selector f = (half==0 ? 0 : 1)                                       │
│        - metadata e = e_lane (nonzero only in selector lanes)                  │
│        -> yields D[0..3] fp32 per lane                                         │
│   3) AS accumulate:                                                            │
│        scale_top = bf16_bits_to_f32(sc_pack.top_half)                          │
│        scale_bot = bf16_bits_to_f32(sc_pack.bot_half)                          │
│        C0 += D0*scale_top                                                      │
│        C1 += D1*scale_top                                                      │
│        C2 += D2*scale_bot                                                      │
│        C3 += D3*scale_bot                                                      │
└──────────────────────────────────────────────────────────────────────────────┘

K2:
  exec_half0: do GATE then UP using same B0
  exec_half1: do GATE then UP using same B1

K4:
  exec_half0/1: do MAIN once
```

---

# 13) Function Poster — K2 Epilogue (SwiGLU + store mapping)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         K2 Epilogue (per phase, per lane)                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ for i=0..3:                                                                    │
│   row = groupID + (i>=2 ? 8 : 0)                                              │
│   tok = 2*t + (i&1)                                                           │
│   m   = m_base + tok                                                          │
│   if m < m_end:                                                               │
│     r = oc_base + row     // output channel in [0..I-1]                        │
│     out_fp32 = silu(C_gate[i]) * C_up[i]                                      │
│     X2_perm[m,r] = bf16(out_fp32)                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

K4 epilogue is identical minus SwiGLU.



Absolutely — here’s the **B-fragment ownership** in the same “groupID / lanes / t” style.

I’m going to show it as **two panels**:

* Panel A: your warp layout (groupID = tok column)
* Panel B: for each group (tok), what **k-rows** each lane’s `{b0,b1,b2,b3}` pulls from the 32×8 slab.

---

┌──────────────────────────────────────────────────────────────────────────────┐
│                 WARP LANES 0..31  (groupID = lane>>2, t = lane&3)            │
├──────────────────────────────────────────────────────────────────────────────┤
│ groupID (= tok col)     lanes (t0 t1 t2 t3)                                   │
│                                                                              │
│ groupID=0   lanes:  0  1  2  3      t: 0  1  2  3     tok = 0                 │
│ groupID=1   lanes:  4  5  6  7      t: 0  1  2  3     tok = 1                 │
│ groupID=2   lanes:  8  9 10 11      t: 0  1  2  3     tok = 2                 │
│ groupID=3   lanes: 12 13 14 15      t: 0  1  2  3     tok = 3                 │
│ groupID=4   lanes: 16 17 18 19      t: 0  1  2  3     tok = 4                 │
│ groupID=5   lanes: 20 21 22 23      t: 0  1  2  3     tok = 5                 │
│ groupID=6   lanes: 24 25 26 27      t: 0  1  2  3     tok = 6                 │
│ groupID=7   lanes: 28 29 30 31      t: 0  1  2  3     tok = 7                 │
└──────────────────────────────────────────────────────────────────────────────┘

Now the **B fragment mapping** (what each lane holds in `b0..b3`).
Let `base = 64*g2 + 32*half` (your contract), and `tok = groupID`.

Each lane has `t ∈ {0,1,2,3}` and therefore `k0 = 2*t`.

* `b0` pulls k = base + (0  + k0) and (0  + k0 + 1)
* `b1` pulls k = base + (8  + k0) and (8  + k0 + 1)
* `b2` pulls k = base + (16 + k0) and (16 + k0 + 1)
* `b3` pulls k = base + (24 + k0) and (24 + k0 + 1)

---

┌──────────────────────────────────────────────────────────────────────────────┐
│            B FRAGMENTS per GROUP (tok = groupID)  —  b0 b1 b2 b3             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Each entry shows:  b# = { XS[ kA ][tok], XS[ kB ][tok] }                      │
│ where kA,kB are within the 32×8 slab starting at k=base.                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ groupID = tok = g                                                             │
│                                                                              │
│  t=0 (lane = 4g+0):  k0=0                                                     │
│    b0 = { XS[base+ 0][g], XS[base+ 1][g] }                                    │
│    b1 = { XS[base+ 8][g], XS[base+ 9][g] }                                    │
│    b2 = { XS[base+16][g], XS[base+17][g] }                                    │
│    b3 = { XS[base+24][g], XS[base+25][g] }                                    │
│                                                                              │
│  t=1 (lane = 4g+1):  k0=2                                                     │
│    b0 = { XS[base+ 2][g], XS[base+ 3][g] }                                    │
│    b1 = { XS[base+10][g], XS[base+11][g] }                                    │
│    b2 = { XS[base+18][g], XS[base+19][g] }                                    │
│    b3 = { XS[base+26][g], XS[base+27][g] }                                    │
│                                                                              │
│  t=2 (lane = 4g+2):  k0=4                                                     │
│    b0 = { XS[base+ 4][g], XS[base+ 5][g] }                                    │
│    b1 = { XS[base+12][g], XS[base+13][g] }                                    │
│    b2 = { XS[base+20][g], XS[base+21][g] }                                    │
│    b3 = { XS[base+28][g], XS[base+29][g] }                                    │
│                                                                              │
│  t=3 (lane = 4g+3):  k0=6                                                     │
│    b0 = { XS[base+ 6][g], XS[base+ 7][g] }                                    │
│    b1 = { XS[base+14][g], XS[base+15][g] }                                    │
│    b2 = { XS[base+22][g], XS[base+23][g] }                                    │
│    b3 = { XS[base+30][g], XS[base+31][g] }                                    │
└──────────────────────────────────────────────────────────────────────────────┘

If you want the same thing but “expanded” for **all 8 groups at once**, you can read it as:

* **Columns are groupID = tok 0..7**
* **Rows are the lane’s (t=0..3) row-pair selector**
* **b0/b1/b2/b3 move down K in steps of 8**

…but the above is the most faithful ASCII equivalent of Figure 121 while matching your group/t framing.


Yep — here’s the “expanded for all 8 groups at once” view.

**Legend**

* `g = groupID = tok ∈ [0..7]`
* `base = 64*g2 + 32*half`
* Each cell shows what that lane holds for `{b0,b1,b2,b3}` as **pairs**:
  `b# = { XS[kA][g], XS[kB][g] }`

---

┌──────────────────────────────────────────────────────────────────────────────┐
│         B FRAGMENT MAP (all groups at once)  —  columns = tok (=groupID)     │
├──────────────────────────────────────────────────────────────────────────────┤
│ Row = t (lane&3). Each row corresponds to a 2-row pair inside each 8×8 block  │
│ b0/b1/b2/b3 correspond to K-blocks: [0..7], [8..15], [16..23], [24..31].      │
├──────────────────────────────────────────────────────────────────────────────┤
│ t=0  (k-pair: 0,1 in each block)                                              │
│   tok=0: b0{base+ 0,1} b1{base+ 8,9} b2{base+16,17} b3{base+24,25}            │
│   tok=1: b0{base+ 0,1} b1{base+ 8,9} b2{base+16,17} b3{base+24,25}            │
│   tok=2: b0{base+ 0,1} b1{base+ 8,9} b2{base+16,17} b3{base+24,25}            │
│   tok=3: b0{base+ 0,1} b1{base+ 8,9} b2{base+16,17} b3{base+24,25}            │
│   tok=4: b0{base+ 0,1} b1{base+ 8,9} b2{base+16,17} b3{base+24,25}            │
│   tok=5: b0{base+ 0,1} b1{base+ 8,9} b2{base+16,17} b3{base+24,25}            │
│   tok=6: b0{base+ 0,1} b1{base+ 8,9} b2{base+16,17} b3{base+24,25}            │
│   tok=7: b0{base+ 0,1} b1{base+ 8,9} b2{base+16,17} b3{base+24,25}            │
├──────────────────────────────────────────────────────────────────────────────┤
│ t=1  (k-pair: 2,3 in each block)                                              │
│   tok=0: b0{base+ 2,3} b1{base+10,11} b2{base+18,19} b3{base+26,27}           │
│   tok=1: b0{base+ 2,3} b1{base+10,11} b2{base+18,19} b3{base+26,27}           │
│   tok=2: b0{base+ 2,3} b1{base+10,11} b2{base+18,19} b3{base+26,27}           │
│   tok=3: b0{base+ 2,3} b1{base+10,11} b2{base+18,19} b3{base+26,27}           │
│   tok=4: b0{base+ 2,3} b1{base+10,11} b2{base+18,19} b3{base+26,27}           │
│   tok=5: b0{base+ 2,3} b1{base+10,11} b2{base+18,19} b3{base+26,27}           │
│   tok=6: b0{base+ 2,3} b1{base+10,11} b2{base+18,19} b3{base+26,27}           │
│   tok=7: b0{base+ 2,3} b1{base+10,11} b2{base+18,19} b3{base+26,27}           │
├──────────────────────────────────────────────────────────────────────────────┤
│ t=2  (k-pair: 4,5 in each block)                                              │
│   tok=0: b0{base+ 4,5} b1{base+12,13} b2{base+20,21} b3{base+28,29}           │
│   tok=1: b0{base+ 4,5} b1{base+12,13} b2{base+20,21} b3{base+28,29}           │
│   tok=2: b0{base+ 4,5} b1{base+12,13} b2{base+20,21} b3{base+28,29}           │
│   tok=3: b0{base+ 4,5} b1{base+12,13} b2{base+20,21} b3{base+28,29}           │
│   tok=4: b0{base+ 4,5} b1{base+12,13} b2{base+20,21} b3{base+28,29}           │
│   tok=5: b0{base+ 4,5} b1{base+12,13} b2{base+20,21} b3{base+28,29}           │
│   tok=6: b0{base+ 4,5} b1{base+12,13} b2{base+20,21} b3{base+28,29}           │
│   tok=7: b0{base+ 4,5} b1{base+12,13} b2{base+20,21} b3{base+28,29}           │
├──────────────────────────────────────────────────────────────────────────────┤
│ t=3  (k-pair: 6,7 in each block)                                              │
│   tok=0: b0{base+ 6,7} b1{base+14,15} b2{base+22,23} b3{base+30,31}           │
│   tok=1: b0{base+ 6,7} b1{base+14,15} b2{base+22,23} b3{base+30,31}           │
│   tok=2: b0{base+ 6,7} b1{base+14,15} b2{base+22,23} b3{base+30,31}           │
│   tok=3: b0{base+ 6,7} b1{base+14,15} b2{base+22,23} b3{base+30,31}           │
│   tok=4: b0{base+ 6,7} b1{base+14,15} b2{base+22,23} b3{base+30,31}           │
│   tok=5: b0{base+ 6,7} b1{base+14,15} b2{base+22,23} b3{base+30,31}           │
│   tok=6: b0{base+ 6,7} b1{base+14,15} b2{base+22,23} b3{base+30,31}           │
│   tok=7: b0{base+ 6,7} b1{base+14,15} b2{base+22,23} b3{base+30,31}           │
└──────────────────────────────────────────────────────────────────────────────┘

Important nuance: the **k-pairs are the same pattern for every tok**, because each `groupID` is just selecting a different **column** of the same 32×8 slab. What changes across columns is only the `tok` index (`XS[...][tok]`), not which k-rows get picked.

If you want it even more “matrixy” (less repeated text), I can compress each cell to just the row-pair indices per `b#`, like:

* `t=2: b0(4,5) b1(12,13) b2(20,21) b3(28,29)` across tok=0..7

and you mentally append `XS[base+row][tok]`.

