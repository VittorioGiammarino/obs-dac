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

## Experiment stages, disturbances, and pairing

1. Validation tests: equilibrium, no-wind tracking, feasibility, integration refinement, Jacobians, residual timing, replay reconstruction, pairing, failure accounting, recovery censoring.
2. Core: nominal; bias-linear; bias-nonlinear; DAC-linear; DAC-nonlinear. Gentle trajectory, small disturbances. Nominal need only run once because estimator choice does not affect its action.
3. One factor at a time: reference amplitude, nominal mass factor, disturbance amplitude, disturbance structure. Defaults include no simultaneous stress factors. Increase demand cautiously and validate no-wind behavior separately before interpreting learning differences.
4. Deferred: actuator lag, noisy observations, stronger adaptive baseline, nonlinear MPC, TD3, controller-stateful extensions, formal stability/regret analysis.

Disturbance conventions: amplitude is per-axis standard deviation for IID and stationary AR(1) gusts; signed constant for bias; peak for sine/chirp. Defaults [.6,.3] m/s². Gust correlation rho=exp(-dt/tau), tau=.5 s, stationary initialization and innovation std sqrt(1-rho²). Periodic frequency .25 Hz, phase offsets (0,.7) rad. Chirp instantaneous frequency f0+.04*t Hz; local time resets at each regime. IID is sample-held noise, not a discretization-invariant continuous white-noise process. Gaussian disturbances are unbounded. Gust processes reset at boundaries.

Default five regimes last 8 seconds each. `switches` controls segment count minus one; regimes cycle if more segments are requested. `duration` separately controls dwell time, rounded to at least one step. Randomized order permutes the same segment multiset (adjacent equal regimes are possible if repeated). Disturbance and order randomness have separate SeedSequence streams. All five controllers reuse exactly the same exogenous acceleration array, reference and initial state. Scenario hashes verify pairing. No controller randomness modifies the scenario. True parameters are fixed within each paired group.

## Cost, metrics, failures and timing

Primary stage cost is `10*||p-p_ref||² + ||v-v_ref||² + .02*||u-[4.905,4.905]||²`. It is integrated as dt times the sum. Weights carry inverse squared units (m^-2, s²/m², N^-2 respectively) so stage cost is dimensionless and integrated cost has seconds. The fixed effort reference is independent of controller/model; it is deviation from true default hover. Keep it fixed across model mismatch. If changing true mass for a new study, explicitly decide and record the common effort reference. Squared thrust is not physical energy.

Position and velocity RMSE use Euclidean error. Attitude error is relative to each controller's desired attitude, so it is diagnostic only and excluded from the common cost. Applied thrust effort, actuator-saturation frequency, constraint violations, failure count, regime costs, gradient/parameter norms, and transition recovery are logged. RMSE and effort suffix `_observed` deliberately identifies partial-trajectory statistics; do not rank failed controllers using these alone.

Failure is declared after a transition if |pitch|>1.2 rad, z<.1 m, position error>5 m, or state is nonfinite. Simulation then stops. Remaining stage costs receive the predeclared penalty 1000 per second; this convention prevents a simply shortened cost sum, but is not a physical continuation or universally dominant penalty. Failure rates must be reported alongside penalized cost; observed cost is separate. Missing states remain NaN in NPZ rather than fabricating trajectories. Constraint-violation count includes the triggering step and stops at termination.

Recovery uses position-error RMS over a trailing .4 s window below .2 m for .4 s persistence, reset at segment start. With rounded W,P samples, detection time is at least `(W+P-1)*dt` (.76 s by default). This is the detection time, not backdated recovery onset. Nonrecovery is right-censored at segment end; failed tails cannot satisfy recovery. Raw events, censor flags, and counts are saved; no uncensored-only mean recovery statistic is presented. Recovery during persistent oscillatory forcing may not be meaningful and should be interpreted accordingly.

Wall-clock measurements cover action, estimation and update, plus their summed end-to-end controller time (including record bookkeeping between estimation and update). Simulator integration and file IO are excluded. Report mean, p95, p99 and dt deadline-miss frequency. Timings are hardware/load-dependent; no real-time claim is made. Delays do not affect simulated dynamics.

## Statistical summaries and plots

The independent unit is the seed, not a time step. Paired differences use the same seed and scenario hash. Seed bootstrap (2000 draws, fixed analysis seed) gives percentile 95% intervals; one-seed smoke results have **null** intervals. These intervals are descriptive with small n, not evidence of significance or multiple-comparison-adjusted inference. Failures remain included via the declared penalty. Per-regime statistics first average repeated occurrences within a seed; randomized sequences thus retain seed independence. Observed-only metrics can be biased by early failures; failure counts are adjacent in aggregate records.

`reporting` produces JSON/CSV aggregate, per-regime and paired-cost results; trajectory/error/actuator/correction/feature/learning plots; per-regime comparisons; stress-axis performance plots; and computation versus cost. Representative traces always use the smallest configured seed, not a selected winner. In diagnostic time plots solid/dashed indicate x/z or left/right components; gradient and parameter norms share the diagnostic panel. Full time series are saved by default; disable `save_timeseries` for metrics-only sweeps (trace plots then cannot be generated).

Every output records resolved configuration per variant, seeds, parameters, hyperparameters, run metrics, source hashes, dependency versions and git revision when available. No favorable-seed selection. Tuning seeds 0–2 and evaluation seeds 100–119 are disjoint; smoke seed 900 is separate. Tune only with `tuning.json`, then copy frozen settings to evaluation configurations. No automatic tuner is implemented.

## Theory context and limits

Verified against the authors' publication abstracts:

- Agarwal et al., [Online Control with Adversarial Disturbances, ICML 2019](https://proceedings.mlr.press/v97/agarwal19c.html): studies linear dynamics with adversarial disturbances and general convex costs. Its guarantees do not automatically cover this nonlinear plant, cascaded controller, clipping or retrospective update.
- Lin et al., [Online Adaptive Policy Selection in Time-Varying Systems: No-Regret via Contractive Perturbations, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/a7a7180fe7f82ff98eee0827c5e9c141-Abstract-Conference.html): analyzes GAPS under contractive policy classes, with policy-regret results in convex settings and local-regret results in nonconvex settings. Our update is distinct and its hypotheses have not been verified here.
- Boffi, Tu and Slotine, [Regret Bounds for Adaptive Nonlinear Control, L4DC 2021](https://proceedings.mlr.press/v144/boffi21a.html): analyzes known nonlinear dynamics with matched uncertainty in a stochastic setting and an oracle comparator, connecting stability and online optimization. This does not establish guarantees for the present outer-loop DAC scheme.

Stability margins and fading dependence on old states can make finite replay useful; informative model gradients and controlled parameter steps can make local learning practical. These are motivations, not established properties of this implementation. Bounded corrections alone do not ensure bounded state or stability. Hover linearization does not transfer linear-system guarantees to the nonlinear plant. Even the linear-model surrogate can be nonconvex because the policy/controller mapping includes nonlinearities and saturation.

Keep four claims separate: (1) stability/boundedness would need a separate argument; (2) empirical gains require held-out paired experiments; (3) local optimization guarantees require applicable smoothness/step-size conditions and do not follow from passing gradient tests; (4) regret requires a precisely specified comparator and applicable theorem. A possible empirical comparator is the best fixed constrained H-history M under a defined replay or actual-plant objective, but no such oracle is computed here. Controller cost differences are **not** labeled regret. No global optimality or sublinear policy-regret claim is made.

## Source map

`plant.py`: dynamics/RK4/sensitivities; `references.py`: trajectories; `models.py`: nominal models; `estimation.py`: residual/features; `controllers.py`: nominal/bias/DAC/surrogate; `scenarios.py`: exogenous paired scenarios; `experiment.py`: orchestration/output; `metrics.py`: costs/failure/recovery/timing summaries; `reporting.py`: bootstrap and plots; `tests/`: mathematical and execution properties.

## Updated headline reporting

Re-run `python -m quad_dac.reporting results/evaluation` on existing saved results; no simulation rerun is required. For each variant, `*_headline_metrics.png` and `.pdf` show three bar panels: position RMSE, cumulative cost, and control-energy proxy. Matching `.json` and `.csv` files contain all plotted numbers, sample counts, selected seeds and failure counts. Bars are seed means; whiskers are 95% percentile seed-bootstrap intervals, omitted for fewer than two seeds.

Cumulative cost retains all paired seeds and the configured failure-tail penalty. RMSE and effort use the common subset of seeds on which every compared controller completed the horizon. This is a conditional successful-run comparison, not an unconditional performance estimate; failure counts are printed on the chart, and all failed runs remain in the original per-run and aggregate outputs. If no common seed completes, those panels display N/A. The control-energy proxy remains the integral of squared rotor-thrust deviation from hover, in N² s, not joules. The cumulative-cost panel preserves the existing objective and does not change its effort definition.

## Provenance of the implemented corrections

The bias correction is a deliberately simple residual-based estimate/filter/compensate baseline; it is not a reproduction of a specific published MRAC, L1, or nonlinear disturbance observer. Background: Wen-Hua Chen, Jun Yang, Lei Guo and Shihua Li, “Disturbance-Observer-Based Control and Related Methods—An Overview,” IEEE Transactions on Industrial Electronics 63(2):1083–1095, 2016, https://doi.org/10.1109/TIE.2015.2478397. This supports the general disturbance-estimation/compensation architecture, not guarantees for our exact velocity-projection/EWMA implementation.

The DAC parameterization is inspired by Definition 4.1 of Agarwal et al. (2019), cited above. Our nonlinear cascaded baseline, projected residual features, clipping, parameter constraints and retrospective window-sum loss are implementation choices, not an exact reproduction of that paper's algorithm. Lin et al. (2023) supplies related contractive online-policy-selection context, not a theorem for this code. Thus the accurate method label is “retrospective model-based online optimization of a disturbance-action residual policy.”

MRAC/L1 were deferred for initial scope, not because they necessarily need hover linearization. Nonlinear examples include Kotaru, Edmonson and Sreenath, “Geometric L1 Adaptive Attitude Control for a Quadrotor Unmanned Aerial Vehicle,” https://arxiv.org/abs/1910.07730; and Wu et al., “L1Quad: L1 Adaptive Augmentation of Geometric Control for Agile Quadrotors With Performance Guarantees,” IEEE Transactions on Control Systems Technology 33(2):597–612, 2025, https://doi.org/10.1109/TCST.2024.3521182, preprint https://arxiv.org/abs/2302.07208. The latter treats translational and rotational nonlinear uncertainties. Neither is implemented here. A future planar adaptation must explicitly handle the disturbance input channels, actuator saturation, predictor/adaptation states and filter design; published guarantees cannot simply be asserted after modifying those architectures.
