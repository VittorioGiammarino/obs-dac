# Nonlinear planar quadrotor: control and disturbance-action learning benchmark

**Version 0.2 adds PID with anti-windup, a nonlinear model-based disturbance observer, planar L1 + PID, and nonlinear MPC + DOB. Start with [BASELINES.md](BASELINES.md) for the expanded configurations, equations, validation, fairness assumptions, and commands. The original five-controller experiment remains available. The original-method description below is retained for reproducibility; deferred baseline items there have now been implemented as described in BASELINES.md.**

A small CPU benchmark for asking **when online DAC corrections help, when filtered bias cancellation suffices, and when inaccurate model gradients hurt**. This is an experimental implementation, not a certified flight controller. No full sweeps have been run or research conclusions inferred.

## Install and run

Python 3.10+; NumPy, SciPy, Matplotlib only. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
# Short single-seed paired execution check (five controller runs)
python -m quad_dac.experiment --config configs/smoke.json --output results/my_smoke
python -m quad_dac.reporting results/my_smoke
# Small paired sweep: three seeds, five controllers, 40 seconds each
python -m quad_dac.experiment --config configs/default.json --output results/core
python -m quad_dac.reporting results/core
# Held-out evaluation: twenty seeds
python -m quad_dac.experiment --config configs/evaluation.json --output results/evaluation
python -m quad_dac.reporting results/evaluation
# Separate tuning seeds; freeze choices before evaluation
python -m quad_dac.experiment --config configs/tuning.json --output results/tuning
# Optional one-factor-at-a-time stress suite and randomized order
python -m quad_dac.experiment --config configs/stress.json --output results/stress
python -m quad_dac.reporting results/stress
python -m quad_dac.experiment --config configs/randomized.json --output results/randomized
```

Output directories must not already exist. Included `results/smoke` is an execution check only. Runs are serial, deterministic except wall-clock timing, and do not model computation-induced delay. Replay is deliberately transparent rather than highly optimized. Full evaluations can be appreciably slower than simulation time.

## Plant, units, and integration

State is `x = [p_x,p_z,v_x,v_z,theta,omega]`, in m, m/s, rad, rad/s. World z points up; positive pitch tilts thrust toward negative x. Input is **applied individual rotor thrusts** `u=[f_left,f_right]` in N. Define total thrust `T=f_left+f_right`, torque `tau=l(f_right-f_left)` in N m. The dynamics are

```
p_dot = v
v_x_dot = -T sin(theta)/m + d_x
v_z_dot =  T cos(theta)/m - g + d_z
theta_dot = omega
omega_dot = tau/J
```

Disturbances `d` are external accelerations in m/s²; equivalent forces are `m*d`. They are piecewise constant over each control interval, including all integration substeps. Default parameters: m=1 kg, J=.02 kg m², l=.2 m, g=9.81 m/s²; each rotor lies in [0,10] N. Default dt=.04 s, two RK4 substeps. Plant and nominal parameters are separate immutable records. Transition is deterministic and expects already feasible rotor inputs.

Hover has arbitrary position, zero velocities/attitude, and each rotor at mg/2. The allocator maps requested `(T,tau)` to `(T-tau/l)/2,(T+tau/l)/2`, clips each rotor, and passes those applied forces to the simulator and estimator. Thus actual input obeys `|tau| <= l*min(T,2*f_max-T)`. Allocation can change both thrust and torque. No fictitious rectangular thrust/torque bounds are used.

## Common nominal controller and reference

The outer loop is `a = a_ref + Kp(p_ref-p) + Kd(v_ref-v) + correction`. Define `q=(a_x,a_z+g)` with vertical component floored at 2 m/s². Desired attitude is `atan2(-q_x,q_z)`, limited to ±.65 rad; thrust is `m*||q||`. Torque is `J*(100*wrapped(theta_des-theta)-20*omega)`. Gains Kp=(2,2), Kd=(2.8,2.8) yield a slow, approximately critically damped translational loop; the faster attitude loop has natural frequency 10 rad/s and damping ratio 1. Desired angular velocity is zero (no differentiated attitude command). This induces lag on moving references and is an intentional simple baseline.

All controller components are **memoryless**. No integral state, command filter, or finite-difference desired-attitude derivative needs resetting in replay. If these are added, their states must also be replayed.

Reference is hover at (0,2), or `p=(A sin(wt),2+.3 A cos(wt))` with exact analytic velocity and acceleration. Defaults A=.5 m, f=.08 Hz. Initial translation/velocity match the reference plus a configured offset, with zero attitude/rate. Initial desired attitude need not be zero. Validation requires nominal no-wind RMSE <.08 m for 12 s of hover and gentle tracking. Gains are analytically selected and validated, not optimized to favor DAC. Tune common gains first using tuning seeds and freeze all hyperparameters before evaluating held-out seeds.

## Correction interface and estimators

Residual methods receive equal elementwise ±2 m/s² outer-loop authority, before attitude generation and physical allocation. This makes a two-dimensional acceleration feature interpretable, while retaining attitude dynamics and actuator limits in replay. It does **not** imply instantaneous realization of commanded acceleration.

Linear model: exact zero-order-hold discretization of the hover linearization. Its continuous Jacobian has position/velocity and pitch/rate integrators, `v_x_dot=-g*theta`, `v_z_dot=(delta f_left+delta f_right)/m`, and `omega_dot=l*(delta f_right-delta f_left)/J`. SciPy matrix exponential computes A,B. Translation invariance permits hover position zero; inputs always use deviation from nominal mg/2. Residual is `w_t=x_(t+1)-A*x_t-B*(u_t-u_hover)`.

Nonlinear model: same RK4 method as the plant, with configurable nominal parameters and zero disturbance. Residual is `w_t=x_(t+1)-F_nom(x_t,u_applied_t)`. Both estimators use applied rotors after allocation. Feature is `z_t = [w_t[2],w_t[3]]/dt` in m/s². This is a velocity projection, not a pseudoinverse or full-state cancellation. Position, pitch and rate residuals cannot directly be canceled by this interface; all six residual coordinates are retained in replay.

The linear residual combines external acceleration and nonlinear mismatch, so it is generally neither physical wind nor exogenous. Nonlinear mismatch can also corrupt the feature. For this particular plant, held additive translational acceleration does not affect attitude or thrust acceleration, so a matched nonlinear model yields exactly `w_velocity=dt*d` and `w_position=.5*dt²*d` up to rounding. This special identity need not hold with drag, state-dependent forces, attitude disturbances, lag, or other coupling; a general discrete residual is an integrated model discrepancy.

The default nonlinear model is **idealized exactly matched**, explicitly configured, not identified online. Controllers receive only the nominal parameter record and observed transitions; the simulator alone receives true parameters/disturbances. The runner logs true disturbances solely for analysis. In mass-mismatch sweeps, nominal mass also changes the feedforward thrust and controller allocation; comparisons within each condition still use exactly the same nominal controller. Thus the mismatch axis tests the full nominal-model pipeline, not a replay-only ablation. True plant, hardware limits, gains, scenario and primary cost remain fixed. Keep nominal gravity, arm and rotor limits equal to known hardware in initial studies.

## Bias cancellation and DAC policy

Filtered disturbance-bias cancellation uses `b <- (1-beta)*b + beta*z_t`, with `beta=1-exp(-dt/.4)`, and correction `clip(-b)`. It is not MRAC, L1, or a general nonlinear adaptive controller.

DAC uses `v_t = clip(sum_i M[i] z_(t-i))` with i=1,...,H. Default H=8 (0.32 s); M has shape (H,2,2), dimensionless entries; history is newest first. Missing history and M initially equal zero. A global Frobenius ball of radius 2 constrains M. There is no imposed decay constraint, and this is not the exact theoretical comparator class from a linear regret theorem.

At time t, action can use features only through t-1. After applying u_t and observing x_(t+1), append w_t,z_t, then update parameters for the next action. Default updates occur every two transitions. Neither the current transition residual nor a future residual enters the current action. The simulator is not queried during optimization.

## Retrospective surrogate and verified sensitivities

After completed transition index t, let `end=t+1`, `s=max(0,end-L)` with L=25 (one second). Anchor replay at the recorded state x_s. Use the same candidate M throughout the replay. For each k=s,...,end-1:

```
z_history = recorded features z_(k-1),...,z_(k-H)
v_k(M) = clipped disturbance-action correction
u_k(M) = common nominal controller(replayed x_k, reference_k, v_k)
replayed x_(k+1) = F_nom(replayed x_k, u_k(M)) + recorded w_k
J(M) = dt * sum_k stage(replayed x_k,u_k,reference_k) + .5*lambda*||M||²
```

Recorded anchor, residuals, feature histories, and references are fixed. Both model variants use identical H,L, costs, constraints and optimizer settings. Replay includes attitude dynamics, the full nonlinear controller, and rotor allocation even for the linear dynamics model. The replay cost uses pre-transition states and all replay stages; the last action is penalized for effort but its next state has no terminal cost. This is a finite retrospective surrogate, not MPC or supervised disturbance prediction. It is not GAPS.

Exact forward sensitivities of the implemented discrete map compute gradients: `S_next=A*S+B*(C*S+D*V)` where C,D are controller Jacobians and V is the correction parameter Jacobian. RK4 stage sensitivities are propagated explicitly. Hard clipping derivatives are zero outside and at bounds, one inside; wrap derivatives are local away from the ±pi branch. Nondifferentiable boundaries do not have unique gradients. Tests compare controller, dynamics and both surrogate gradients to central finite differences away from boundaries, and verify factual replay reconstruction.

Projected gradient descent: raw gradient norm clipped to 10, learning rate .03, L2 regularization .001, then Frobenius projection. Nonfinite gradients skip the update and increment a logged counter. Holding residuals fixed is an approximation when changing the policy changes model errors. It may produce misleading gradients, especially away from hover or with incorrect models. Clipping can yield zero gradients. Defaults are starting settings, not claims of optimal tuning.

