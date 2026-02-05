================================================================================
K2 (W13 + SwiGLU) — CTA-LEVEL POSTER (v6 ., A100/SM80)
CTA → Warps → 4-lane Groups → P=1 Stage/Exec windows → Gate+Up → Phases
================================================================================

Legend / pins:
- CTA = 128 threads = 4 warps (wid=0..3)
- NTOK = 8 tokens per CTA tile
- KTILE = 32  (mma.sp m16n8k32)
- OTILE_CTA = 128 output rows per CTA = 2 phases × 64 rows
- K2 sweep per phase: g2=0..111, half∈{0,1}
- base = 64*g2 + 32*half  (32-wide K slice)
- Selector: half0 -> f=0 (metadata lanes t0/t1), half1 -> f=1 (metadata lanes t2/t3)
- P=1: stage(0)->next; then for each g2: cur=next; stage(g2+1)->next; exec(cur)
- Each half-step does TWO MMAs: GATE then UP (same Bregs, different weights/metadata)

--------------------------------------------------------------------------------
[0] CTA WORK ASSIGNMENT (one block)
--------------------------------------------------------------------------------

blockIdx.y = eid
blockIdx.x = token tile inside expert
  m_base = OffE[eid] + blockIdx.x*NTOK
  m_end  = OffE[eid+1]
blockIdx.z = output tile over I
  i_base = blockIdx.z * 128

CTA owns:
  tokens: m = m_base + {0..7}  (zero-pad if m>=m_end)
  output rows (per phase): 64 rows
  output rows (CTA total): 128 rows = phase0(64) + phase1(64)

--------------------------------------------------------------------------------
[1] CTA → WARPS → ROWS (PHASED)  (visual map)
--------------------------------------------------------------------------------

Within a phase:
  phase_base = i_base + phase*64
  each warp covers 16 rows:
    oc_base = phase_base + wid*16

PHASE 0 (rows i_base + 0..63)
┌──────────────────────────────────────────────────────────────────────────────┐
│  warp0 wid=0 : rows [phase_base+ 0 .. phase_base+15]                         │
│  warp1 wid=1 : rows [phase_base+16 .. phase_base+31]                         │
│  warp2 wid=2 : rows [phase_base+32 .. phase_base+47]                         │
│  warp3 wid=3 : rows [phase_base+48 .. phase_base+63]                         │
└──────────────────────────────────────────────────────────────────────────────┘

PHASE 1 (rows i_base + 64..127)
┌──────────────────────────────────────────────────────────────────────────────┐
│  warp0 wid=0 : rows [phase_base+ 0 .. phase_base+15]   (phase_base=i_base+64)│
│  warp1 wid=1 : rows [phase_base+16 .. phase_base+31]                         │
│  warp2 wid=2 : rows [phase_base+32 .. phase_base+47]                         │
│  warp3 wid=3 : rows [phase_base+48 .. phase_base+63]                         │
└──────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
[2] WARP → 4-LANE GROUPS → TOP/BOT ROWS (within the warp’s 16-row tile)
--------------------------------------------------------------------------------

Warp lanes:
  lane=tid&31
  groupID=lane>>2 (0..7)
  t=lane&3 (0..3)

Each group owns two rows of the warp tile:
  row_top = groupID       (0..7)
  row_bot = groupID + 8   (8..15)

So for this warp’s oc_base:
  gate rows:
    r0  = oc_base + row_top
    r1  = oc_base + row_bot
  up rows:
    r0u = r0 + I
    r1u = r1 + I

Group picture (inside one warp):
┌──────────────────────────────────────────────────────────────────────────────┐
│ groupID 0: rows 0 & 8   groupID 1: rows 1 & 9   ...   groupID 7: rows 7 & 15 │
│ lanes: (t0,t1,t2,t3)    lanes: (t0,t1,t2,t3)              lanes: (t0..t3)     │
└──────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
[3] CTA PREP: MONOLITH-B PRELOAD X_perm -> Xsh_H (one barrier)
--------------------------------------------------------------------------------

Xsh_H[7168][8] bf16  (~112KB), layout: row-major in k, contiguous in tok
stride per k-row = 16B.

PREP pseudocode:
┌──────────────────────────────────────────────────────────────────────────────┐
│ for k = tid; k < 7168; k += 128:                                              │
│   for tok=0..7:                                                               │
│     m = m_base + tok                                                          │
│     Xsh_H[k][tok] = (m < m_end) ? X_perm[m][k] : 0                            │
│ __syncthreads()                                                               │
└──────────────────────────────────────────────────────────────────────────────┘

After PREP: Xsh_H is read-only; no more barriers.

================================================================================
[4] PHASE LOOP — BIG POSTER TIMELINE (P=1 Stage/Exec, per warp/group)
================================================================================

For each phase in {0,1}:
  phase_base = i_base + phase*64
  oc_base = phase_base + wid*16
  init C_gate[4]=0, C_up[4]=0  (per lane)

Then:
  stage(g2=0)->slot_next
  for g2=0..111:
    slot_cur = slot_next
    if g2+1 < 112: stage(g2+1)->slot_next
    exec(slot_cur)

--------------------------------------------------------------------------------
[4A] Per-g2 EXEC cadence (constant) — what happens for ALL warps/groups
--------------------------------------------------------------------------------

Per g2:
  half0 (f=0): B0 + (GATE mma) + (UP mma)
  half1 (f=1): B1 + (GATE mma) + (UP mma)

B load (shared for gate+up):
  base0 = 64*g2 + 0
  base1 = 64*g2 + 32
  B0 = ldmatrix.x4.trans(&Xsh_H[base0][0])
  B1 = ldmatrix.x4.trans(&Xsh_H[base1][0])

ASCII “per g2” strip:
┌──────────────────────────────────────────────────────────────────────────────┐
│ g2 step:                                                                      │
│   [half0 f=0]  B0=ldmatrix  ->  GATE mma.sp  ->  UP mma.sp                   │
│   [half1 f=1]  B1=ldmatrix  ->  GATE mma.sp  ->  UP mma.sp                   │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
[5] STAGE(g2) — CTA-level meaning (it runs per group, 8 groups/warp, 4 warps/CTA)
================================================================================

Think of stage(g2) as 32 independent “micro-stagers” in the CTA:
  4 warps × 8 groups each = 32 groups
Each group stages ONLY its (row_top,row_bot) for the current oc_base rows.

For K2, stage(g2) stages BOTH paths:
  - gate rows (r0,r1)
  - up rows (r0u,r1u)

CTA-level view:
┌──────────────────────────────────────────────────────────────────────────────┐
│ stage(g2)                                                                     │
│   for each warp wid in 0..3:                                                  │
│     for each groupID in 0..7:                                                 │
│       STAGE_PATH(GATE rows r0/r1)                                             │
│       STAGE_PATH(UP   rows r0u/r1u)                                           │
│ (all of the above is warp-synchronous; no block-wide barrier)                 │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
[6] . Ping-Pong Flip — CTA-level timeline (nested)
================================================================================

Below: “what each 4-lane group is doing” while the CTA runs g2 loop.
This is the exact nesting you wanted: CTA → warp → group → windows.

For one group:
  - half0 uses selector f=0 => metadata lanes (t0,t1) are “busy”
  - half1 uses selector f=1 => metadata lanes (t2,t3) are “busy”
So the “free pair” flips per half, enabling staging of NEXT(g2+1) during CUR(g2).

PER GROUP: . windowing over ONE g2
┌──────────────────────────────────────────────────────────────────────────────┐
│ WINDOW A: EXEC CUR half0 (f=0)                                                │
│   - t0/t1 busy feeding metadata for CUR half0                                 │
│   - t2/t3 free: LOAD+BCast NEXT weights (vec2) for BOTH paths (gate+up)       │
│                                                                              │
│ WINDOW B: BUILD NEXT half0 metadata (f=0 lanes t0/t1) for BOTH paths          │
│   - uses NEXT half0 ( .x ) weights already broadcast                          │
│                                                                              │
│ WINDOW C: EXEC CUR half1 (f=1)                                                │
│   - t2/t3 busy feeding metadata for CUR half1                                 │
│   - t0/t1 free (no required work)                                             │
│                                                                              │
│ WINDOW D: BUILD NEXT half1 metadata (f=1 lanes t2/t3) for BOTH paths          │
│   - uses NEXT half1 ( .y ) weights already broadcast                          │
│                                                                              │
│ WINDOW E: SWAP slot_cur <- slot_next                                          │
└──────────────────────────────────────────────────────────────────────────────┘

CTA-level note:
- This windowing happens simultaneously across all 32 groups in the CTA.
- No CTA barrier is required; everything is warp-synchronous + register-resident.

================================================================================
[7] EPILOGUE (per phase): SwiGLU + store X2_perm
================================================================================

Per lane (4 outputs):
  out = silu(C_gate[i]) * C_up[i]   (fp32)
  store bf16 to X2_perm[m][r]

ASCII:
┌──────────────────────────────────────────────────────────────────────────────┐
│ for i in {0,1,2,3}:                                                           │
│   row = groupID + (i>=2 ? 8 : 0)                                              │
│   tok = 2*t + (i&1)                                                           │
│   m   = m_base + tok                                                          │
│   if m < m_end:                                                               │
│     r = oc_base + row                                                         │
│     X2_perm[m][r] = bf16( silu(C_gate[i]) * C_up[i] )                         │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
[8] FULL CTA MASTER PICTURE (everything in one box)
================================================================================

K2 CTA(eid, m_base, i_base)
┌──────────────────────────────────────────────────────────────────────────────┐
│ PREP: X_perm -> Xsh_H (7168×8), 1 barrier                                     │
│                                                                              │
│ for phase in {0,1}:                                                           │
│   oc_base = (i_base + phase*64) + wid*16                                      │
│   C_gate,C_up = 0                                                             │
│   stage(0)->next                                                              │
│   for g2 in 0..111:                                                           │
│     cur=next; stage(g2+1)->next                                               │
│     exec(cur):                                                                │
│       half0 f=0: B0=ldmatrix(Xsh_H[64*g2+0 ]), gate mma, up mma               │
│       half1 f=1: B1=ldmatrix(Xsh_H[64*g2+32]), gate mma, up mma              │
│   store: X2_perm = bf16( silu(C_gate) * C_up )                                │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
End K2 CTA-LEVEL POSTER (v6)
================================================================================

================================================================================
. — PING-PONG FLIP (P=1)  [K2 ASCII Schematic]
(per 4-lane group: t0,t1,t2,t3 ; rows: top=groupID, bot=groupID+8 ; per g2)
K2 has TWO paths: GATE and UP, both use same Bregs per half.
================================================================================

Pinned selector schedule:
  half0 -> f=0 -> metadata lanes: (t0,t1)
  half1 -> f=1 -> metadata lanes: (t2,t3)

Goal:
- While executing CUR(g2), stage NEXT(g2+1) “in the gaps” using the *free* lane pair.
- NEXT weights are vec2-loaded by (t2,t3) during CUR half0.
- NEXT half0 metadata for BOTH paths (gate+up) is built by (t0,t1).
- NEXT half1 metadata for BOTH paths (gate+up) is built by (t2,t3).
- Shuffles “flip” source lanes depending on whether we’re handling CUR or NEXT.

--------------------------------------------------------------------------------
STATE (per group)
--------------------------------------------------------------------------------
Rows owned by the group (within the warp’s m16):
  row_top = groupID
  row_bot = groupID + 8

For a given phase and warp:
  gate rows: r0  = oc_base + row_top
             r1  = oc_base + row_bot
  up rows:   r0u = r0 + I
             r1u = r1 + I

slot_cur  (for g2):
  GATE weights: uTopG_cur01, uBotG_cur01      // {half0,half1} each
  UP   weights: uTopU_cur01, uBotU_cur01
  half0 meta parked in (t0,t1):
    GATE: (t0:e0G_cur_0_3, t1:e0G_cur_4_7)
    UP  : (t0:e0U_cur_0_3, t1:e0U_cur_4_7)
  half1 meta parked in (t2,t3):
    GATE: (t2:e1G_cur_0_3, t3:e1G_cur_4_7)
    UP  : (t2:e1U_cur_0_3, t3:e1U_cur_4_7)

slot_next (for g2+1): same fields with “_next”.

--------------------------------------------------------------------------------
ONE g2 ITERATION — MICRO-TIMELINE (per group)
--------------------------------------------------------------------------------

                 ┌─────────────────────────────────────────────────────────┐
                 │ WINDOW A: EXEC CUR half0  (f=0 uses t0/t1 metadata)      │
                 └─────────────────────────────────────────────────────────┘
  t0,t1 = “busy” (must hold e0*_cur / feed mma)
  t2,t3 = “free”  -> LOAD NEXT weights (vec2) for BOTH paths and broadcast

  ┌───────────────────────┐                           ┌───────────────────────┐
  │ t2 loads TOP (NEXT)    │                           │ t3 loads BOT (NEXT)   │
  │  uTopG_next01 (vec2)   │                           │  uBotG_next01 (vec2)  │
  │  uTopU_next01 (vec2)   │                           │  uBotU_next01 (vec2)  │
  └───────────┬───────────┘                           └───────────┬───────────┘
              │                                                   │
              ├──── shfl-broadcast TOP next weights ──────────────┤  (to t0,t1,t3)
              │                                                   │
              └──── shfl-broadcast BOT next weights ──────────────┘  (to t0,t1,t2)

  After broadcast: all lanes (t0..t3) hold NEXT uTop/uBot for BOTH paths.

                 ┌─────────────────────────────────────────────────────────┐
                 │ WINDOW B: BUILD NEXT half0 metadata (f=0 lanes = t0/t1)  │
                 └─────────────────────────────────────────────────────────┘
  t0,t1 now free (between half0 and half1 setup) -> compute NEXT half0 meta
  for BOTH paths using NEXT half0 data (.x).

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ GATE half0 meta:                                                         │
  │  t0: e0G_next_0_3 = meta0_3(topG_h0_next) | (meta0_3(botG_h0_next)<<16)   │
  │  t1: e0G_next_4_7 = meta4_7(topG_h0_next) | (meta4_7(botG_h0_next)<<16)   │
  │ UP   half0 meta:                                                         │
  │  t0: e0U_next_0_3 = meta0_3(topU_h0_next) | (meta0_3(botU_h0_next)<<16)   │
  │  t1: e0U_next_4_7 = meta4_7(topU_h0_next) | (meta4_7(botU_h0_next)<<16)   │
  │ park all e0*_next in (t0,t1)                                              │
  └─────────────────────────────────────────────────────────────────────────┘

                 ┌─────────────────────────────────────────────────────────┐
                 │ WINDOW C: EXEC CUR half1  (f=1 uses t2/t3 metadata)      │
                 └─────────────────────────────────────────────────────────┘
  t2,t3 = “busy” (must hold e1*_cur / feed mma)
  t0,t1 = “free” (NEXT loads already done; no required staging here)

                 ┌─────────────────────────────────────────────────────────┐
                 │ WINDOW D: BUILD NEXT half1 metadata (f=1 lanes = t2/t3)  │
                 └─────────────────────────────────────────────────────────┘
  After half1 MMAs, t2,t3 become free -> compute NEXT half1 meta for BOTH paths
  using NEXT half1 data (.y).

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ GATE half1 meta:                                                         │
  │  t2: e1G_next_0_3 = meta0_3(topG_h1_next) | (meta0_3(botG_h1_next)<<16)   │
  │  t3: e1G_next_4_7 = meta4_7(topG_h1_next) | (meta4_7(botG_h1_next)<<16)   │
  │ UP   half1 meta:                                                         │
  │  t2: e1U_next_0_3 = meta0_3(topU_h1_next) | (meta0_3(botU_h1_next)<<16)   │
  │  t3: e1U_next_4_7 = meta4_7(topU_h1_next) | (meta4_7(botU_h1_next)<<16)   │
  │ park all e1*_next in (t2,t3)                                              │
  └─────────────────────────────────────────────────────────────────────────┘

                 ┌─────────────────────────────────────────────────────────┐
                 │ WINDOW E: SWAP                                            │
                 └─────────────────────────────────────────────────────────┘
  slot_cur <- slot_next
  (If g2+1 out of range: skip loads/meta; exec just finishes CUR.)

--------------------------------------------------------------------------------
WHERE THE “FLIP” HAPPENS (the key aesthetic)
--------------------------------------------------------------------------------
- NEXT loads (both paths) are sourced from (t2,t3) during CUR half0  [pair NOT used by f=0]
- NEXT half0 meta (both paths) built by (t0,t1)                     [pair used by f=0]
- NEXT half1 meta (both paths) built by (t2,t3)                     [pair used by f=1]
- Exec half0 reads meta from (t0,t1); Exec half1 reads meta from (t2,t3)

================================================================================
Notes (pinned)
--------------------------------------------------------------------------------
- This schematic is per 4-lane group; 8 groups per warp run it in parallel.
- Vec2 is safe: uTop*_01={half0,half1}, uBot*_01={half0,half1}.
- Shuffles are within the 4-lane group only; broadcast sources for NEXT are t2(top) and t3(bot).
================================================================================








================================================================================
K2 — STAGE(g2) ONLY (NO EXEC)  [ASCII Blueprint, gate+up explicit]
(per 4-lane group: t0,t1,t2,t3 ; rows: top=groupID, bot=groupID+8)
================================================================================

Pinned selector schedule (context):
  half0 -> f=0 -> metadata lanes: (t0,t1)
  half1 -> f=1 -> metadata lanes: (t2,t3)

What STAGE(g2) must output (into slot_next), PER GROUP:
  GATE:
    uTopG01 = {uTopG_h0, uTopG_h1}     // TOP row (r0)
    uBotG01 = {uBotG_h0, uBotG_h1}     // BOT row (r1)
    half0 meta parked in (t0,t1):  (t0:e0G_0_3, t1:e0G_4_7)
    half1 meta parked in (t2,t3):  (t2:e1G_0_3, t3:e1G_4_7)

  UP:
    uTopU01 = {uTopU_h0, uTopU_h1}     // TOP row (r0u)
    uBotU01 = {uBotU_h0, uBotU_h1}     // BOT row (r1u)
    half0 meta parked in (t0,t1):  (t0:e0U_0_3, t1:e0U_4_7)
    half1 meta parked in (t2,t3):  (t2:e1U_0_3, t3:e1U_4_7)

All shuffles are WITHIN the 4-lane group only.

--------------------------------------------------------------------------------
STAGE(g2) structure
--------------------------------------------------------------------------------
STAGE(g2) = STAGE_PATH(GATE rows r0/r1)  +  STAGE_PATH(UP rows r0u/r1u)

┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE_PATH(path_rows) does S1–S4:                                             │
│   [S1] row-split vec2 loads (t0 loads TOP, t1 loads BOT)                      │
│   [S2] broadcast weights to t0..t3 (group shfl)                               │
│   [S3] distributed nibble decode (each lane decodes its own i_lo=t, i_hi=t+4)│
│   [S4] ping-pong pack+park metadata into selector lanes (no meta broadcast)   │
└──────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
STAGE_PATH(path) pipeline (S1–S4)
--------------------------------------------------------------------------------

┌──────────────────────────────────────────────────────────────────────────────┐
│ [S1] Row-split GLOBAL LOADS (both halves at once)                             │
│   t0 loads TOP row half-pair:  uTop01 = { W[top,half0], W[top,half1] }       │
│   t1 loads BOT row half-pair:  uBot01 = { W[bot,half0], W[bot,half1] }       │
│   (vec2 = ulonglong2 allowed; scalar fallback = two u64 loads)               │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ [S2] Group BROADCAST of weights (so every lane can decode its own chunks)     │
│   broadcast uTop01 from t0 → {t1,t2,t3}                                       │
│   broadcast uBot01 from t1 → {t0,t2,t3}                                       │
│   After S2: ALL lanes have uTop01 and uBot01.                                 │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ [S3] Distributed nibble decode (each lane decodes its own contributors)       │
│   lane t owns: i_lo=t (0..3) and i_hi=t+4 (4..7)                              │
│   For half0 (uTop01.x/uBot01.x) and half1 (uTop01.y/uBot01.y):                │
│     n_top_lo*, n_bot_lo*, n_top_hi*, n_bot_hi*                                │
│   nibble(idx16,i): idx2=(idx16>>(2*i))&3 ; pair=idx2>>1 ; nib=(pair?0xE:0x4)  │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ [S4] Ping-pong PACK + PARK metadata into selector lanes (NO meta broadcast)   │
│                                                                              │
│   HALF0 (f=0): metadata lanes are (t0,t1)                                     │
│     t0 builds e0_0_3 (chunks 0..3):                                           │
│        gather n_top_lo0 from {t0..t3} → pack → meta0_3(top_h0)                │
│        gather n_bot_lo0 from {t0..t3} → pack → meta0_3(bot_h0)                │
│        e0_0_3 = meta0_3(top_h0) | (meta0_3(bot_h0)<<16)                       │
│     t1 builds e0_4_7 (chunks 4..7):                                           │
│        gather n_top_hi0 from {t0..t3} → pack → meta4_7(top_h0)                │
│        gather n_bot_hi0 from {t0..t3} → pack → meta4_7(bot_h0)                │
│        e0_4_7 = meta4_7(top_h0) | (meta4_7(bot_h0)<<16)                       │
│                                                                              │
│   HALF1 (f=1): metadata lanes are (t2,t3)                                     │
│     t2 builds e1_0_3 (chunks 0..3):                                           │
│        gather n_top_lo1 from {t0..t3} → pack → meta0_3(top_h1)                │
│        gather n_bot_lo1 from {t0..t3} → pack → meta0_3(bot_h1)                │
│        e1_0_3 = meta0_3(top_h1) | (meta0_3(bot_h1)<<16)                       │
│     t3 builds e1_4_7 (chunks 4..7):                                           │
│        gather n_top_hi1 from {t0..t3} → pack → meta4_7(top_h1)                │
│        gather n_bot_hi1 from {t0..t3} → pack → meta4_7(bot_h1)                │
│        e1_4_7 = meta4_7(top_h1) | (meta4_7(bot_h1)<<16)                       │
│                                                                              │
│   Result: half0 meta parked in (t0,t1), half1 meta parked in (t2,t3).         │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
Notes (pinned)
--------------------------------------------------------------------------------
- This is STAGE ONLY; no ldmatrix, no mma, no accumulators.
- K2 STAGE calls STAGE_PATH twice: once for gate rows (r0/r1), once for up rows (r0u/r1u).
================================================================================


================================================================================
K2 — EXEC(g2) ONLY (NO STAGE)  [ASCII Blueprint, constant cadence]
(per 4-lane group uses staged weights + staged metadata)
================================================================================

Inputs available from slot_cur (per group):
  GATE: uTopG01/uBotG01 + (half0 meta in t0/t1) + (half1 meta in t2/t3)
  UP  : uTopU01/uBotU01 + (half0 meta in t0/t1) + (half1 meta in t2/t3)

Exec never global-loads weights. It only:
  - loads B from Xsh_H via ldmatrix
  - builds A regs (per half) from staged u64 weights
  - mma.sp uses selector f and reads metadata from correct lanes
  - AS accumulates into C_gate and C_up

--------------------------------------------------------------------------------
EXEC(g2) cadence (per group, per phase)
--------------------------------------------------------------------------------

[EXEC half0]  f=0
  base0 = 64*g2 + 0
  B0 = ldmatrix.x4.trans(&Xsh_H[base0][0])

  GATE:
    A from uTopG01.x / uBotG01.x   (chunks t and t+4 for top/bot rows)
    e from lanes (t0:e0G_0_3, t1:e0G_4_7), others 0
    mma.sp(... f=0) -> D -> AS accumulate into C_gate

  UP:
    A from uTopU01.x / uBotU01.x
    e from lanes (t0:e0U_0_3, t1:e0U_4_7)
    mma.sp(... f=0) -> D -> AS accumulate into C_up

[EXEC half1]  f=1
  base1 = 64*g2 + 32
  B1 = ldmatrix.x4.trans(&Xsh_H[base1][0])

  GATE:
    A from uTopG01.y / uBotG01.y
    e from lanes (t2:e1G_0_3, t3:e1G_4_7), others 0
    mma.sp(... f=1) -> D -> AS accumulate into C_gate

  UP:
    A from uTopU01.y / uBotU01.y
    e from lanes (t2:e1U_0_3, t3:e1U_4_7)
    mma.sp(... f=1) -> D -> AS accumulate into C_up

================================================================================
End K2 schematics (. style)
================================================================================



================================================================================
stage_path() — GROUPID/t MAPPING + FLOWCHART (ASCII)
(Your current implementation: row-split load -> group shfl -> S3 decode -> TODO gather/pack)
================================================================================

Warp lane mapping (pinned):
  lane    = tid & 31
  groupID = lane >> 2        // 0..7
  curr_t  = lane & 3         // 0..3

So the warp is 8 independent 4-lane groups:

                (t0 t1 t2 t3)
  group 0: lanes  0  1  2  3    
  group 1: lanes  4  5  6  7
  group 2: lanes  8  9 10 11
  ...
  group 7: lanes 28 29 30 31

Each group owns two output rows inside the warp’s m16 tile:
  row_top = groupID
  row_bot = groupID + 8

Within a warp’s oc_base:
  TOP row index = oc_base + groupID
  BOT row index = oc_base + groupID + 8

--------------------------------------------------------------------------------
Selector context (why src_t exists)
--------------------------------------------------------------------------------
We call stage_path with:
  src_t = 0  when f=0  (metadata lanes = t0,t1)
  src_t = 2  when f=1  (metadata lanes = t2,t3)

So (src_t, src_t+1) is the “loader pair” for the row-split loads:
  if src_t=0: loaders are (t0 loads TOP, t1 loads BOT)
  if src_t=2: loaders are (t2 loads TOP, t3 loads BOT)

This matches . “free pair flips by half”.

================================================================================
stage_path() — HIGH-LEVEL DATAFLOW (one 4-lane group)
================================================================================

Inputs:
  W          : packed weights as ulonglong2 {half0=u64x, half1=u64y}
  g2         : K-block group index
  oc_base    : warp’s output-row base (16 rows)
  groupID    : which pair of rows (top/bot)
  src_t      : loader pair start (0 or 2)

Outputs (what stage_path prepares conceptually):
  qwTop = {top_h0_u64, top_h1_u64}  broadcast to all lanes in the group
  qwBot = {bot_h0_u64, bot_h1_u64}  broadcast to all lanes in the group
  plus per-lane decoded:
    - bf16 values for chunks i_lo=t and i_hi=t+4  (top/bot, half0/half1)
    - idx2 for those chunks (to build phantom A pairs + metadata nibble)
    - scale_bits (top/bot, half0/half1)

--------------------------------------------------------------------------------
[STAGE_PATH] — FLOWCHART (ASCII)
--------------------------------------------------------------------------------

                               (per lane in warp)
        ┌───────────────────────────────────────────────────────────┐
        │ lane -> (groupID, curr_t)                                  │
        │   groupID = lane>>2   curr_t = lane&3                      │
        └───────────────────────────────────────────────────────────┘
                                   │
                                   v
        ┌───────────────────────────────────────────────────────────┐
        │ Define group-local source lanes for broadcast:              │
        │   src_lane_top = (groupID<<2) + src_t                       │
        │   src_lane_bot = (groupID<<2) + (src_t+1)                   │
        └───────────────────────────────────────────────────────────┘
                                   │
                                   v
        ┌───────────────────────────────────────────────────────────┐
        │ S1) Row-split GLOBAL LOAD (only 2 lanes do real loads)     │
        │                                                           │
        │ if curr_t == src_t:                                        │
        │   qwTop = W[ ... (oc_base + groupID) ... ]    // TOP row    │
        │                                                           │
        │ if curr_t == src_t+1:                                      │
        │   qwBot = W[ ... (oc_base + groupID + 8) ... ] // BOT row   │
        └───────────────────────────────────────────────────────────┘
                                   │
                                   v
        ┌───────────────────────────────────────────────────────────┐
        │ S2) GROUP BROADCAST (warp shuffle, per-group src lanes)    │
        │   mask = __activemask()                                    │
        │   qwTop = shfl_u64x2(qwTop, src_lane_top, mask)            │
        │   qwBot = shfl_u64x2(qwBot, src_lane_bot, mask)            │
        │                                                           │
        │ After S2: ALL lanes in the group have qwTop and qwBot.     │
        └───────────────────────────────────────────────────────────┘
                                   │
                                   v
        ┌───────────────────────────────────────────────────────────┐
        │ S3) DISTRIBUTED CHUNK DECODE (per lane t)                  │
        │   i_lo = curr_t       // chunks 0..3                       │
        │   i_hi = curr_t + 4   // chunks 4..7                       │
        │                                                           │
        │ decode(qwTop.x, i_lo) -> top_h0_lo, idx2_top_h0_lo, sc_top_h0│
        │ decode(qwTop.x, i_hi) -> top_h0_hi, idx2_top_h0_hi, sc_top_h0│
        │ decode(qwTop.y, i_lo) -> top_h1_lo, idx2_top_h1_lo, sc_top_h1│
        │ decode(qwTop.y, i_hi) -> top_h1_hi, idx2_top_h1_hi, sc_top_h1│
        │                                                           │
        │ decode(qwBot.x, i_lo) -> bot_h0_lo, idx2_bot_h0_lo, sc_bot_h0│
        │ decode(qwBot.x, i_hi) -> bot_h0_hi, idx2_bot_h0_hi, sc_bot_h0│
        │ decode(qwBot.y, i_lo) -> bot_h1_lo, idx2_bot_h1_lo, sc_bot_h1│
        │ decode(qwBot.y, i_hi) -> bot_h1_hi, idx2_bot_h1_hi, sc_bot_h1│
        └───────────────────────────────────────────────────────────┘
                                   │
                                   v
        ┌───────────────────────────────────────────────────────────┐
        │ S4) TODO (next steps)                                      │
        │   A) Phantom 0 injection (1-of-4 -> 2-of-4 pair):           │
        │      pair = idx2>>1  -> metadata nibble = (pair?0xE:0x4)    │
        │      slot = idx2&1   -> (v0,v1) = (val,0) or (0,val)        │
        │                                                           │
        │   B) Gather/pack metadata e (within 4-lane group):          │
        │      - gather 4 nibbles to form meta0_3 (chunks 0..3)        │
        │      - gather 4 nibbles to form meta4_7 (chunks 4..7)        │
        │      - pack top+bot into u32 e = lo16 | (hi16<<16)           │
        │                                                           │
        │   C) Park e into selector lanes (depends on half / f):       │
        │      - half0 f=0: e lives in (t0,t1), other lanes e=0        │
        │      - half1 f=1: e lives in (t2,t3), other lanes e=0        │
        └───────────────────────────────────────────────────────────┘


================================================================================
GroupID / t visualization for your W loads (per warp)
================================================================================

Within one warp, at a given oc_base:

  groupID 0:
    TOP row index = oc_base + 0
    BOT row index = oc_base + 8
    lanes = 0..3

  groupID 1:
    TOP row index = oc_base + 1
    BOT row index = oc_base + 9
    lanes = 4..7

  ...

  groupID 7:
    TOP row index = oc_base + 7
    BOT row index = oc_base + 15
    lanes = 28..31

Loader pair depends on src_t:
  src_t=0  (f=0 / half0):
    t0 loads TOP, t1 loads BOT, then shfl to group
  src_t=2  (f=1 / half1):
    t2 loads TOP, t3 loads BOT, then shfl to group

================================================================================
stage_path() pseudo (compact)
================================================================================

stage_path(W, curr_t, src_t, g2, oc_base, groupID):
  if curr_t==src_t:     qwTop = W[top_row]
  if curr_t==src_t+1:   qwBot = W[bot_row]

  qwTop = shfl(qwTop, src_lane_top)
  qwBot = shfl(qwBot, src_lane_bot)

  i_lo = curr_t
  i_hi = curr_t+4

  decode(qwTop.x/y, i_lo/i_hi) and decode(qwBot.x/y, i_lo/i_hi)
  // next:
  phantom_pair + nibble, gather/pack e, park e in selector lanes

================================================================================
End stage_path() schematic
================================================================================


================================================================================
v6 “PER-g2” MASTER SCHEMATIC (. + stage/park/exec)
A100 / SM80 — K2+K4 share the same per-g2 cadence; K2 has GATE+UP, K4 is single-path
This is the “don’t lose context” poster: lane roles + data structs + windows.
================================================================================

Pinned facts (recap):
- Warp split into 8 groups (groupID 0..7), each group = 4 lanes (t0..t3)
- Each group owns 2 output rows in the warp’s m16:
    row_top = groupID
    row_bot = groupID + 8
- Selector schedule (Marlin-style):
    half0 -> f=0 -> metadata lanes must be (t0,t1)
    half1 -> f=1 -> metadata lanes must be (t2,t3)
- . P=1:
    While executing CUR(g2), we stage NEXT(g2+1) using the free pair.
- Your stage() produces:
    StageOut { top_h0/bot_h0/top_h1/bot_h1 (int8x4),
              sc_pack (4 bf16 scale bits),
              nib_h0_lo/hi, nib_h1_lo/hi (packed top/bot nibbles for S3) }
- Your park(half) consumes StageOut.nib_* and emits:
    e_0_3_out, e_4_7_out parked into selector lanes (by half)

================================================================================
LANE MAPPING (WARP -> GROUP -> t)
================================================================================

warp lanes: 0..31

groupID = lane >> 2    (0..7)
t      = lane & 3      (0..3)

Group layout (one warp):
  group0: lanes  0  1  2  3    => (t0 t1 t2 t3)
  group1: lanes  4  5  6  7
  ...
  group7: lanes 28 29 30 31

Within a warp tile (oc_base):
  TOP row index = oc_base + groupID
  BOT row index = oc_base + groupID + 8

================================================================================
DATA OBJECTS (per group, per g2)
================================================================================

StageOut (per lane, but conceptually “per group after shfl”):
  vals (int8 phantom pairs packed):
    top_h0 : uint32  // int8x4 = [lo_v0 lo_v1 hi_v0 hi_v1] for TOP row, half0
    bot_h0 : uint32  // same for BOT row, half0
    top_h1 : uint32  // TOP row, half1
    bot_h1 : uint32  // BOT row, half1

  scales:
    sc_pack : uint64
      [15:0]   = bf16 bits scale(top, half0)
      [31:16]  = bf16 bits scale(bot, half0)
      [47:32]  = bf16 bits scale(top, half1)
      [63:48]  = bf16 bits scale(bot, half1)

  nibble tokens for S3 park (packed top/bot nibble):
    nib_h0_lo : uint16   // low4=top nib, high4=bot nib  for chunks 0..3 (i_lo=t)
    nib_h0_hi : uint16   // for chunks 4..7 (i_hi=t+4)
    nib_h1_lo : uint16
    nib_h1_hi : uint16

Meta outputs from park():
  half0 (f=0): e0_0_3 in t0, e0_4_7 in t1
  half1 (f=1): e1_0_3 in t2, e1_4_7 in t3

================================================================================
PER-g2 CADENCE (ONE ITERATION) — WINDOWED POSTER (.)
================================================================================

Notation:
- CUR  = slot_cur (for g2)
- NEXT = slot_next (for g2+1)
- stage(...) produces StageOut (vals + nibs + scales) into NEXT
- park(half) consumes NEXT.nibs and emits NEXT.e for that half
- exec_half(half) consumes CUR.vals + CUR.sc_pack + CUR.e and issues MMA(s)

--------------------------------------------------------------------------------
WINDOW A: EXEC CUR half0  (f=0)   +   STAGE NEXT using free pair (t2/t3)
--------------------------------------------------------------------------------

    CUR half0 uses metadata lanes t0/t1  => t0/t1 are “busy”
    free pair is t2/t3 => perfect time to stage NEXT loads

┌──────────────────────────────────────────────────────────────────────────────┐
│ EXEC CUR half0 (f=0)                                                          │
│   B0 = ldmatrix( Xsh[64*g2 + 0] )                                             │
│   mma.sp reads metadata from lanes (t0,t1)                                    │
│                                                                              │
│   K4: 1 mma per half                                                         │
│   K2: 2 mmas per half  (GATE then UP, same B0)                                │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE NEXT (g2+1) occurs “in the gap” using src_t=2 (t2/t3 loaders)           │
│                                                                              │
│   stage(W, curr_t=t, src_t=2, g2+1, uid, G2, oc_base, R, groupID, NEXT)      │
│                                                                              │
│   Inside stage():                                                             │
│     - only t2 loads TOP row vec2, only t3 loads BOT row vec2                  │
│     - shfl broadcasts qwTop/qwBot to all 4 lanes in the group                 │
│     - decode i_lo=t and i_hi=t+4 into:                                        │
│         * phantom int8 pair packed into int16 (v0,v1)                         │
│         * meta_nibble (0x4/0xE)                                               │
│       then pack into NEXT.top_h{0,1}/bot_h{0,1} and NEXT.nib_h*_{lo,hi}        │
│     - extract NEXT.sc_pack from qwTop/qwBot high bits                          │
└──────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
WINDOW B: PARK NEXT half0 metadata into (t0,t1)  (f=0 lanes)
--------------------------------------------------------------------------------
Now t0/t1 are free (between half0 and half1 setup). Commit NEXT half0 e.

┌──────────────────────────────────────────────────────────────────────────────┐
│ park(NEXT, half=0, groupID, curr_t=t, e0_0_3_out, e0_4_7_out)                 │
│                                                                              │
│ Uses: NEXT.nib_h0_lo and NEXT.nib_h0_hi (each lane has its nibble token)      │
│ Gather across t=0..3 and pack:                                                │
│   meta0_3_top, meta0_3_bot  from nib_h0_lo                                   │
│   meta4_7_top, meta4_7_bot  from nib_h0_hi                                   │
│   e0_0_3 = meta0_3_top | (meta0_3_bot << 16)                                  │
│   e0_4_7 = meta4_7_top | (meta4_7_bot << 16)                                  │
│ Park into lanes:                                                              │
│   t0 gets e0_0_3, t1 gets e0_4_7, other lanes set 0                            │
└──────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
WINDOW C: EXEC CUR half1  (f=1)
--------------------------------------------------------------------------------

┌──────────────────────────────────────────────────────────────────────────────┐
│ EXEC CUR half1 (f=1)                                                          │
│   B1 = ldmatrix( Xsh[64*g2 + 32] )                                            │
│   mma.sp reads metadata from lanes (t2,t3)                                    │
│                                                                              │
│   K4: 1 mma per half                                                         │
│   K2: 2 mmas per half (GATE then UP, same B1)                                 │
└──────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
WINDOW D: PARK NEXT half1 metadata into (t2,t3)  (f=1 lanes)
--------------------------------------------------------------------------------
After half1 MMAs, t2/t3 are free. Commit NEXT half1 e.

┌──────────────────────────────────────────────────────────────────────────────┐
│ park(NEXT, half=1, groupID, curr_t=t, e1_0_3_out, e1_4_7_out)                 │
│                                                                              │
│ Uses: NEXT.nib_h1_lo and NEXT.nib_h1_hi                                       │
│ Packs: e1_0_3 and e1_4_7 (top+bot)                                            │
│ Park into lanes:                                                              │
│   t2 gets e1_0_3, t3 gets e1_4_7, other lanes set 0                            │
└──────────────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
WINDOW E: SWAP
--------------------------------------------------------------------------------

┌──────────────────────────────────────────────────────────────────────────────┐
│ slot_cur <- slot_next                                                         │
│ Now NEXT becomes CUR for g2+1, and we continue.                               │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
ILLUSTRATIVE “IN-ONE-LINE” PER-g2 STRIP (for your README)
================================================================================

g2 loop:
  exec_half0(CUR,f=0)  || stage(NEXT,src_t=2)  -> park(NEXT,half0->t0/t1)
  exec_half1(CUR,f=1)  -> park(NEXT,half1->t2/t3)
  swap

================================================================================
K2 vs K4 differences (so you don’t lose context)
================================================================================

K4:
  - stage once (single path) -> NEXT StageOut
  - exec_half0: B0 + 1 mma
  - exec_half1: B1 + 1 mma

K2:
  - stage twice per g2 (two StageOuts): GATE and UP
    e.g., stage_gate(...) -> NEXT.gate_out
          stage_up(...)   -> NEXT.up_out
  - exec_half0: B0 + mma(GATE) + mma(UP)
  - exec_half1: B1 + mma(GATE) + mma(UP)
  - epilogue per lane: out = silu(C_gate[i]) * C_up[i]

================================================================================
End poster
================================================================================
