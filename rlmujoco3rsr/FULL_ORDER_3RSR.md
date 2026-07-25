# Full-Order Dynamics of the 3RSR Hopper (Floating Base, Closed Chain)

Derivation of the full-order constrained dynamics used for low-level torque
feedforward, sitting **below** the SRB/HLIP planning layer:

```
SRB / HLIP (apex states, desired GRF f*, footstep)     <- template layer
        |  f*, base trajectory
        v
Full-order 3RSR constrained inverse dynamics  ->  tau_ff (hip torques)
        + WBC/feedback                        ->  tau_cmd
```

Verified numerically against MuJoCo (`three_leg_3rsr_closed.xml`) by
`full_order_id.py` — see "Verification" at the bottom.

---

## 1. Topology and coordinates

The MuJoCo model realizes each RSR chain as **R(actuated hip) – U(cross pin
`alpha_i` about Y, lower pin `beta_i` about Z) – S(ball `connect` at the shared
foot point)**. Kinematically this is equivalent to the physical 3RSR for foot
positioning (the distal S is exact; the middle S of the physical robot is split
into U here plus the spin freedom absorbed by the ball).

Bodies and generalized velocity `v ∈ R^24`:

| block | symbol | dim | notes |
|---|---|---|---|
| base linear/angular | `v_b = (ṗ_b, ω_b)` | 6 | free joint |
| chain i = 1..3 | `(θ̇_i, α̇_i, β̇_i)` | 3×3 | θ actuated, α/β passive |
| prop servos | `σ̇ ∈ R^3` | 3 | tilt hinges |
| foot free body | `v_f = (ṗ_f, ω_f)` | 6 | free joint |

Configuration `q` (nq = 26 with quaternions). Actuation selector
`S ∈ R^{3×24}` picks the three hip rows.

Effective mobility: 24 dof − 9 loop constraints = 15 in flight
(= base 6 + hips 3 + servos 3 + foot spin 3); in stance, a point contact
removes 3 more → 12.

## 2. Loop-closure constraints

Let `s_i(q) ∈ R^3` be the world position of the distal site of chain i
(function of base pose and `θ_i, α_i, β_i`), and `p_f` the foot-body origin
(the `connect` anchor). The nine holonomic constraints:

```
Φ_i(q) = s_i(q) − p_f = 0,   i = 1,2,3        Φ(q) ∈ R^9
```

Velocity / acceleration level, with `G(q) = ∂Φ/∂v ∈ R^{9×24}`:

```
G v = 0
G v̇ + Ġ v = 0
```

Rows of `G`: `G_i = J_{s_i} − J_{p_f}` (point Jacobians). Because the anchor
sits at the foot origin, `ω_f` does not enter `G` — the foot spin is exactly
the 3-dim internal freedom left by the three ball joints.

## 3. Open-tree equations of motion (Euler–Lagrange / DAE index 3)

Cutting the three balls yields an open tree (base tree + separate foot).
Standard floating-base Lagrangian dynamics plus constraint forces:

```
M(q) v̇ + C(q,v) v + g(q) = Sᵀ τ + B_p(q) u_p + Gᵀ λ + J_cᵀ f_c + d(v)
Φ(q) = 0
```

- `M ∈ R^{24×24}`: joint-space inertia (includes rotor armature),
- `C v + g`: Coriolis/centrifugal + gravity (`qfrc_bias` in MuJoCo),
- `τ ∈ R^3`: hip torques (the unknown feedforward),
- `B_p u_p`: prop thrust + reaction drag + servo torques (known inputs),
- `λ ∈ R^9`: loop-closure multipliers (internal chain forces),
- `f_c ∈ R^3`: ground reaction at the foot contact point, `J_c` its point
  Jacobian (stance only),
- `d(v)`: joint damping (`qfrc_passive`).

This is an index-3 DAE; all uses below work at the acceleration level with
`Φ, GΦ` used for stabilization/projection.

## 4. Constraint elimination (reduced / independent dynamics)

Choose `T(q) ∈ R^{24×15}` whose columns span `ker G` — the independent
quasi-velocities `ν = (v_b, θ̇_1, θ̇_2, θ̇_3, σ̇, ω_f)`, `v = T ν`.
Since `G T = 0`, multipliers drop out:

```
TᵀM T ν̇ + Tᵀ( M Ṫ ν + C v + g − d ) = Tᵀ Sᵀ τ + Tᵀ B_p u_p + Tᵀ J_cᵀ f_c
```

15 ODEs in flight; in stance append `J_c v̇ + J̇_c v = 0` or restrict `T`
further. The dependent rates recover as
`(α̇_i, β̇_i, ṗ_f) = −G_d^{-1} G_ind ν` with the partition `G = [G_ind G_d]`
(`G_d ∈ R^{9×9}` invertible away from chain singularities
`|β_i| < π/4`, which the joint limits enforce).

**Leg Jacobian used by the template layer.** With the base frozen, solving
`G v = 0` for a unit `θ̇_k` yields the foot linear velocity; stacking gives
`J_leg(q) ∈ R^{3×3}`, `ṗ_f^b = J_leg θ̇`. This is the same object the current
controller uses for its static map `τ_static = −J_legᵀ R_bᵀ f*` — i.e. the
static map is the zero-inertia, zero-velocity special case of the equations
above.

## 5. Inverse dynamics feedforward

### 5.1 Stance (push-off / compression)

Inputs from the SRB layer: desired base acceleration `v̇_b*` (or desired GRF
`f*`), foot anchored. Steps at each control tick:

1. **Consistent accelerations.** Solve the linear system
   `G v̇* = −Ġ v`, `J_c v̇* = −J̇_c v`, rows of `v̇*` at (base, servos)
   pinned to the reference — a 24-var linear solve for the passive-joint and
   foot accelerations.
2. **Torques + internal forces.** With `v̇*` known, the dynamics are linear in
   the unknowns `x = (τ, λ, f_c) ∈ R^{15}`:

```
A x = r,   A = [ Sᵀ  Gᵀ  J_cᵀ ] ∈ R^{24×15}
r = M v̇* + C v + g − d − B_p u_p
```

   24 equations, 15 unknowns: solve least-squares. The residual lives in the
   rows not spanned by `A` (foot spin + base rows unreachable given `f*`);
   its norm is a **dynamic-consistency check** on the reference — large
   residual means the SRB layer asked for something the full model cannot do.

3. `τ_ff = x_{1:3}`; feedback (existing SLX/WBC loop) is added on top.

### 5.2 Flight (swing retraction)

Same system without the `J_cᵀ f_c` column; prescribe hip trajectory
`θ̈*` and thrust `u_p`, solve for `τ` — this replaces the pure PD swing
torque with inertia-aware tracking (matters at fast retraction, where the
0.396 m lower legs generate real Coriolis load).

### 5.3 Degeneration to SRB (paper lemma)

Taking leg + foot inertias → 0 in the reduced dynamics, rows 1–6 collapse to

```
m_b p̈_b = m_b g + f_c + Σ R_thrust,i u_i
I_b ω̇_b + ω_b × I_b ω_b = r_c × f_c + Σ (moments of thrust)
```

and the hip rows collapse to `τ = J_legᵀ R_bᵀ f_c` — exactly the SRB + static
Jacobian map used today. So the template layer is the provable zero-leg-mass
limit of the full-order model, and the feedforward correction
`τ_ff − τ_static` is entirely leg-inertia/Coriolis effects.

## 6. Analytical (closed-form) derivation

`derive_full_order_sympy.py` derives every term of Sec. 3 **symbolically**
(SymPy); this is the version citable in a paper as an analytical model.

**Coordinates.** To keep plain Lagrangian mechanics (no quaternion
quasi-velocity bookkeeping), both free bodies use Euler ZYX angles:

```
q = [ p_b, (φ,θ,ψ)_b, (θ_i,α_i,β_i)_{i=1..3}, σ_{1..3}, p_f, (φ,θ,ψ)_f ] ∈ R^24
```

**Method (per body, closed form).** World pose of each of the 14
mass-carrying bodies is a short chain of fixed transforms and joint
rotations. From `p_com,b(q)` and `R_b(q)`:

```
Jv_b = ∂p_com,b/∂q                       (linear point Jacobian)
ω_b  = unskew(Ṙ_b R_bᵀ) = Jw_b(q) q̇      (angular Jacobian)

M(q)   = Σ_b  m_b Jv_bᵀJv_b + Jw_bᵀ I_b^w Jw_b      + diag(armature)
h(q,q̇) = Σ_b  Jv_bᵀ m_b (a_b^vp − g) + Jw_bᵀ (I_b^w α_b^vp + ω_b×I_b^w ω_b)
```

with `a^vp, α^vp` the velocity-product accelerations (time derivatives of
`v, ω` at `q̈ = 0`). `Φ(q)` and `G = ∂Φ/∂q` come from the symbolic chain-tip
positions. The **thrust input matrix** is likewise closed form: rotor i
produces force `u_i d_i(q)` and reaction drag `±k_m u_i d_i(q)` along the
tilted thrust axis `d_i = R_prop,i(q) e_x`, so

```
B_p(q)[:,i] = Jv_prop,iᵀ d_i + k_m,i Jw_prop,iᵀ d_i        (k_m = ±0.018 m)
```

Numeric parameters (masses, COMs, inertia tensors, fixed transforms, axes,
armature) are read from the compiled MuJoCo model, so structure is symbolic
while parameters cannot drift from the plant.

**Impact map (hybrid model, for the paper).** At touchdown with pre-impact
velocity `v⁻`, the post-impact velocity solves the plastic-impact KKT system
with the stacked constraint `Ĝ = [G; J_c]`:

```
[ M   Ĝᵀ ] [ v⁺ ]   [ M v⁻ ]
[ Ĝ   0  ] [ −Λ ] = [   0  ]
```

**MuJoCo convention bridge.** Verification and any state exchange use the
velocity map `v_mj = E(q) q̇` (identity except the two 3×3 Euler-rate →
angular-velocity blocks). Testing both candidates confirmed MuJoCo free
joints use **linear velocity in world frame, angular velocity in body
frame** (the world-frame assumption fails at 1e-1 level, body frame agrees
to 1e-16).

**Verification against MuJoCo** (random `(q, q̇, q̈)`, base/foot poses and
rates fully excited; max abs error over all entries):

| quantity | max error |
|---|---|
| `M(q)` vs `Eᵀ M_mj E` | 5.3e-16 |
| full inverse dynamics `M q̈ + h` vs mapped MuJoCo ID | 1.4e-14 |
| `G(q)` vs mapped site Jacobians | 5.1e-16 |
| `B_p(q)` vs differenced `qfrc_actuator` | 1.8e-15 |

**Generated code.** The derivation script emits `gen_full_order_dyn.py`
(pure numpy, no sympy at runtime): `mass_matrix`, `bias_force`,
`constraint_jac`, `input_matrix`, `vel_map_body/world`. Generated module
re-verified against MuJoCo to 7e-16. Timings (Python, single call):
`M` 0.40 ms, `h` 0.61 ms, `G` 0.054 ms, `B_p` 0.028 ms — usable as-is at
500 Hz feedforward; for a strict 1 kHz budget evaluate `M`/`h` at 500 Hz or
port the generated file to C (mechanical translation).

**Scope caveat (for paper claims).** The model realizes each physical RSR
chain as R (actuated hip) – U (cross + lower pins) – S (ball at the shared
foot), which is the standard kinematic substitution for foot positioning;
before claiming "full-order dynamics of the physical robot", confirm against
CAD that joint axes/offsets match, and identify inertial parameters on
hardware (the structure stays valid; only parameters change). Actuator rotor
dynamics and friction are not included beyond armature.

## 6b. Numerical realization (MuJoCo-backed)

`full_order_id.py` implements Sec. 5 against the MuJoCo model directly —
the same equations with model terms read numerically:

- `M` ← `mj_fullM`, `C v + g` ← `qfrc_bias`, `d` ← `qfrc_passive`,
- `G` ← site point-Jacobian differences (`mj_jacSite`),
- contact `J_c` ← `mj_jac` at the active foot contact,
- known actuators (servos, thrusts) ← `qfrc_actuator` rows,
- least-squares solve of `A x = r` per tick (~24×15, microseconds).

Use this in sim experiments; use `gen_full_order_dyn.py` (Sec. 6) where an
analytical, MuJoCo-free implementation is wanted (paper, onboard runtime).

## 7. Verification (torque replay)

Protocol in `full_order_id.py`:

1. Run a prescribed squat/hop motion with the model's own position actuators;
   record `(q, v, ctrl)` each step.
2. At each frame recompute `v̇` by `mj_forward`, then solve Sec. 5.2's system
   with `τ` **unknown** and compare against the torque MuJoCo's actuators
   actually applied.
3. Also compute `τ_static = J_legᵀ R_bᵀ f_c` from the measured GRF and compare
   — the gap quantifies what full-order feedforward buys during push-off.

Results (`demos/full_order_id_verify.png`, `demos/full_order_id_verify_3hz.png`):

| motion | true τ range | full-order ID RMS err | static Jac map RMS err |
|---|---|---|---|
| 1.5 Hz squat (always in stance) | ±4.8 Nm | **0.0000 Nm** | 0.52 Nm |
| 3.0 Hz squat→hopping (flight phases) | ±8.8 Nm | **0.0000 Nm** | 2.18 Nm |

- Full-order ID reconstructs the actuator torque to machine precision from
  `(q, v, v̇)` alone, in both stance and flight, including through touchdown
  impacts — the constraint structure (Sec. 3–5) is exact for this plant.
- It simultaneously recovers the GRF as a by-product (`f_c` unknown matches
  the measured contact force), i.e. the same solve doubles as a
  **model-based contact force estimator** — usable to replace/monitor the
  touch sensor on hardware.
- The static Jacobian map (today's controller feedforward) is fine
  quasi-statically but degrades to ~2.2 Nm RMS with 10–15 Nm transient spikes
  at touchdown when the motion is dynamic — that gap is precisely the
  leg-inertia/Coriolis term the full-order feedforward adds (Sec. 5.3).
