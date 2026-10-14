# Four control baselines and paired stress testing (v0.2)

This extension implements the four requested controller families alongside the existing DAC variants. It is an executable experimental benchmark, not a claim of controller superiority or a transfer of published guarantees. RL remains deferred.

## Run the expanded benchmark

From the extracted `quad_dac` directory, install as described in README, then:

```bash
python -m unittest discover -s tests -v
# Nine controllers, one short constant-bias realization
python -m quad_dac.experiment --config configs/controllers_smoke.json --output results/new_smoke
# Bounded implementation checks: six conditions x nine controllers x one seed
python -m quad_dac.experiment --config configs/controllers_checks.json --output results/new_checks
python -m quad_dac.reporting results/new_checks
# Tune using seeds 0,1,2 only, then freeze settings before held-out tests
python -m quad_dac.experiment --config configs/controllers_tuning.json --output results/tuning_v2
# Twenty paired held-out seeds; this includes expensive nonlinear MPC
python -m quad_dac.experiment --config configs/controllers_evaluation.json --output results/evaluation_v2
python -m quad_dac.reporting results/evaluation_v2
# Randomized regime order; same multiset of regimes, histories retained
python -m quad_dac.experiment --config configs/controllers_randomized.json --output results/randomized_v2
# Select one stress case first, rather than launching the entire matrix
python -m quad_dac.experiment --config configs/controllers_stress.json --variant nominal_mass_0.8 --output results/mass08
# Select controllers or seeds explicitly for debugging
python -m quad_dac.experiment --config configs/controllers_checks.json --variant base --methods pid:nonlinear dob:nonlinear l1:nonlinear nmpc:nonlinear --seeds 902 --output results/debug
```

The full stress matrix is intentionally optional and potentially expensive. The `--variant` names are specified in the JSON's `cases` list; omitting `--variant` runs base plus every listed case. Every CLI override and default is saved in the resolved configuration. Original `default`, `evaluation`, `stress`, and `randomized` configurations still select only the original five controllers. Use the new `controllers_*` configurations to include the four additions. Output directories must be new. Reporting works with old result directories as well as new ones; new transition and observer diagnostics require new runs.

## Controller definitions

### PID with anti-windup (`pid`)

Keep the common nominal PD attitude loop. The outer desired acceleration becomes `a_ref + Kp e_p + Kd e_v + b`, where `b` is an integral acceleration state, initialized to zero. Update after applying the current action: `b_next = clip(b + dt Ki e_p, ±correction_bound)`. Defaults Ki=(.8,.8), with the existing PD gains unchanged. When the requested vertical acceleration, tilt, or rotor command is infeasible, conditional integration permits only increments that reduce the magnitude of the existing integral state. Projection prevents accumulation beyond the correction bounds. The integrator is not reset at disturbance switches, but is reset for every run. No future residual enters its update.

This is a conventional cascaded PID design with conservative conditional anti-windup. It is not claimed to be optimally tuned. Its integral acceleration has the same ±2 m/s² bounds as the bias and DAC corrections.

### Nonlinear model-based disturbance observer (`dob`)

A six-state augmented Kalman disturbance observer estimates `y=(vx,vz,omega)` and `d=(dx,dz,domega)`. At each completed transition, the nominal model produces the known increment `Delta_y_nom = F_nom(x,u)[2,3,5] - y`. The observer propagation is

```
[y_hat_next] = [I dt*I] [y_hat] + [Delta_y_nom]
[d_hat_next]   [0    I] [d_hat]   [0]
```

The measurement is the next observed velocity/rate triple. Disturbance is modeled as a random walk. Continuous disturbance-rate spectral density Qd gives discrete covariance blocks `[dt^3 Qd/3, dt^2 Qd/2; dt^2 Qd/2, dt Qd]`. Measurement covariance R is a tuning regularizer in this full-state, noise-free experiment. Joseph-form covariance update preserves symmetry and positive semidefiniteness numerically. Defaults and clipping bounds are in `dob` configuration.

The translation estimate is negated and bounded at the common outer-loop interface. The angular estimate supplies a matched torque correction `-J_nom*domega_hat`, limited to ±.2 Nm before final rotor allocation. Thus DOB also has an angular compensation channel; this is explicitly broader than current translation-only DAC. The nonlinear known increment is evaluated using measured state; this is a practical augmented-state observer, not an exact nonlinear Bayesian filter or a reproduction of a particular published nonlinear DOB theorem. Initial disturbance mean is zero. The internal velocity estimate starts from measured velocity; every run gets fresh covariance and state.

Reference: Chen et al., “Disturbance-Observer-Based Control and Related Methods—An Overview,” IEEE TIE 63(2), 2016, DOI 10.1109/TIE.2015.2478397. It supports the estimator/compensator architecture; this implementation and covariance choices are our own. An application precedent for Kalman disturbance estimation with quadrotor robust control is “Disturbance Observer-Based Robust Control of a Quadrotor Subject to Parametric Uncertainties and Wind Disturbance,” IEEE Access 10, 2022, DOI 10.1109/ACCESS.2022.3141939. We do not implement that paper's robust LQR.

### Planar L1 adaptive augmentation on PID (`l1`)

The implemented architecture has a velocity/rate predictor, a piecewise-constant uncertainty adaptation law, and a first-order control filter. It estimates both matched and unmatched disturbances, but directly compensates only thrust-axis force and pitch moment. The underlying PID handles lateral position error through tilting and integral action. Reporting labels the method **L1 + PID**, so its nominal-controller difference is visible.

Provenance: planar adaptation of Wu et al., “L1Quad,” arXiv:2302.07208v2 / IEEE TCST 33(2), 2025, Eqs. (19)–(25). Differences: our PID base, 2D channels, sampled predictor quadrature, finite bounds and rotor saturation. The published guarantees are not asserted here.

For our coordinates, define n=(-sin(theta),cos(theta)), its orthogonal tangent q=(cos(theta),sin(theta)), and y=(vx,vz,omega). The invertible channel matrix has columns `(n/m,0)`, `(0,0,1/J)` and `(q/m,0)`. Multiplying its inverse by the estimated velocity-rate discrepancy h yields matched thrust error (N), matched moment error (Nm), and unmatched lateral force (N). Horizontal wind at zero pitch is correctly assigned to the unmatched channel, not treated as instantly cancelable by thrust.

Detailed sampled implementation: choose positive diagonal predictor rates a, E=exp(-a*dt), Phi=(1-E)/a. Initialize predicted y to measured y and h to zero. After receiving y_next, propagate the prediction error by

```
e_next = E*e + Phi*(Delta_y_nom/dt + h - (y_next-y)/dt)
y_hat_next = y_next + e_next
h_next = -E/Phi * e_next
```

This integrates diagonal predictor feedback assuming measured y changes linearly across the observed interval and nominal acceleration equals its interval average. It uses only a completed transition, never simulator substeps or future observations. With constant acceleration discrepancy d and initially zero error, the first predictor error is `-Phi*d` and h_next=E*d; this finite-sampling attenuation is tested and should not be mistaken for an exact disturbance estimate. Faster sampling reduces it. Rates default to 5/s and filter bandwidths to 10 rad/s. At action calculation, project h into the current input channels, filter the two matched estimates with exp(-bandwidth*dt), negate, and add to PID thrust/moment. Input corrections are bounded at ±2 N and ±.2 Nm, then rotors are reallocated to their physical limits. Filter state is bounded to avoid hidden windup. Predictor uses actual allocated commands. All internal states persist across switches and reset between runs.

This is an L1-style sampled planar adaptation with the characteristic predictor/adaptation/filter architecture. It is not MRAC, and it is not an exact L1Quad code reproduction. A dominance claim over the published controller would be unjustified.

### Nonlinear MPC with disturbance estimation (`nmpc`)

Uses the same augmented-state observer as DOB. At each action, hold its current 3D disturbance-rate estimate fixed over a prediction horizon and optimize individual rotor commands directly through nonlinear RK4 dynamics. No true disturbance preview is supplied. Known reference preview is obtained from the same analytic reference generator available to the experiment; other controllers use its instantaneous position/velocity/acceleration.

Default horizon is 20 steps (.8 s), with two-step move blocks (10 two-rotor decisions). Objective is the exact common stage cost times dt plus a terminal quadratic. Terminal P is the discrete Riccati solution at nominal hover with the common tracking/effort weights; terminal state includes reference position, velocity, feedforward attitude and attitude rate. P is a heuristic terminal design for the nonlinear problem, not a verified terminal invariant-set construction. There are hard rotor box bounds only; state failure limits remain shared evaluation limits, not hard predicted-state constraints. There is no claimed recursive feasibility or stability theorem.

The objective gradient uses explicit RK4 Jacobians and a backward discrete adjoint. SciPy L-BFGS-B solves the single-shooting problem for at most 35 iterations. A shifted previous solution and a fresh DOB-compensated nominal rollout initialize the solve; the lower-cost candidate is chosen. Every finite, feasible improvement is retained. If the solver stops before convergence, the best feasible iterate (possibly the initialization) is used and nonconvergence/fallback diagnostics are logged. This preserves actuator feasibility but does not establish optimality. The plan is shifted by one executed step before being reblocked. MPC has no outer-loop correction signal: its `v` array is zero by convention and is excluded from outer-correction diagnostic plots.

This is a conventional nonlinear MPC implementation with disturbance estimation, not a reproduction of a fixed-time observer MPC paper. Related application: Xu et al., “Fixed-time Disturbance Observer-Based MPC Robust Trajectory Tracking Control of Quadrotor,” arXiv:2408.15019 / IEEE/ASME Transactions on Mechatronics, DOI 10.1109/TMECH.2024.3503062. Our observer is Kalman-based, not their fixed-time observer.

## What is shared and what differs

| Method | Base / command interface | Online mechanism | Model access |
|---|---|---|---|
| Nominal | common PD cascade | none | nominal parameters |
| PID | same PD + integral acceleration | tracking-error integration | nominal parameters |
| Bias | common PD + bounded outer correction | EWMA prediction residual | linear or nonlinear nominal model |
| DOB | common PD + outer correction + angular correction | augmented-state observer | nonlinear nominal model |
| L1 + PID | PID + bounded thrust/moment correction | predictor, adaptation, LPF | nonlinear nominal model |
| NMPC + DOB | direct rotor commands | observer + receding-horizon optimization | nonlinear nominal model, reference preview |
| DAC | common PD + bounded outer correction | retrospective policy-parameter descent | linear or nonlinear nominal model |

All receive full state, the same reference, same external disturbance realization, same true plant and physical rotor bounds, and the same evaluated cost. There is no sensor noise or actuator lag. Hardware rotor limits must agree in true/nominal configurations; mass, inertia and drag can differ. No controller reads hidden true mass/inertia/drag or disturbance samples. Online estimator computations for the newly added methods occur after observing the completed transition.

This is a **controller architecture comparison**, not a perfectly isolated correction-learning ablation. DOB/L1 have angular compensation; PID and L1 have integral state; MPC has direct rotor authority and reference preview. Those capabilities are deliberate and should be reported. The old bias-vs-DAC comparison remains the closer common-interface ablation. Comparing methods with different internal states does not invalidate the study, but hides no equality-of-information or equality-of-authority claim.

## Stress assumptions

`controllers_stress.json` varies one factor at a time: nominal mass ±20%, nominal inertia ±30%, unmodeled translational linear drag, unmodeled angular damping, periodic frequency, gust correlation time, disturbance amplitude, dwell time, and number of switches. For nominal-parameter changes the true plant is fixed; for unmodeled drag the nominal drag is fixed at zero and true drag changes. All methods face the same case.

Added physical terms are `-drag*v/m` in translational acceleration and `-angular_drag*omega/J` in angular acceleration. Drag units are kg/s and kg m²/s respectively. This linear damping is an interpretable model-error test, not a calibrated aerodynamic model. The plant transition additionally accepts optional external angular acceleration for observer validation, while default scenarios retain only translational disturbances. No thrust-effectiveness mismatch is silently introduced: commands still mean applied rotor thrusts in N.

These parameter ranges and gains are reasonable starting values, not tuned optima or experimentally calibrated platform bounds. No finite grid establishes universal robustness. Gaussian forcing, abrupt regime switches, sampled implementation and input saturation also lie outside some cited continuous-time theorem assumptions. Failures and solver nonconvergence are retained.

## Timing and reporting

The runner now measures each controller's native observer in its estimation time and native MPC optimization in its action time. It still computes the common prediction residual for all controllers as a logged diagnostic, even if that method does not use it; this small common logging overhead is included in reported controller timing. Solver diagnostics show nonconvergence and fallback counts. All timing includes Python implementation costs and cannot establish an inherent complexity advantage over optimized implementations. Computational delays do not perturb simulated dynamics; check p95/p99 and deadline misses before any real-time interpretation.

Reporting now discovers arbitrary subsets of the supported methods, preserves consistent colors across plots, shows pitch and regime boundaries, separates extra thrust/torque from outer corrections, and adds a diagnostics plot. Nine-controller headline bars include the requested position RMSE, cumulative cost and squared-input effort proxy, with failure counts. Cost retains failed runs via penalties. RMSE/effort bars use the shared completed-seed subset; pairwise differences use the corresponding pair's completed subset. All seed sets are explicit.

`paired.json/csv` now contains all three metrics for each DAC variant against every baseline. Negative differences favor DAC. `phases.json/csv` reports the first configurable `transition_window` seconds (default 2 s) and the remaining portion of each segment, normalized as mean stage cost to account for different window lengths. These phases are descriptive time windows, not certified steady-state periods. For randomized order, repeat occurrences are averaged within seed before aggregate bootstrapping. Per-run phase records also contain position RMSE and effort where complete.

## Tuning and limitations

Use `controllers_tuning.json` seeds 0–2 for tuning all controllers on the same scenario collection. Freeze settings and then run held-out seeds 100–119; the supplied evaluation files are runnable starting configurations, not the output of a completed tuning study. Give every controller an explicit comparable tuning budget. Sweep PID integral gains, observer process/measurement settings, L1 bandwidth/rates, MPC horizon/iterations, and DAC memory/replay/learning rate. This release does not add an automatic hyperparameter tuner or train RL.

A paper-grade comparison still needs matched tuning effort, independent-regime and randomized-order evaluation, realistic disturbance calibration, and review of each baseline design. Bounded checks validate execution and mathematical properties; they do not establish a ranking. Future improvements can retain full experiment metadata and refine this implementation without describing the present provisional baselines as exact reproductions of the cited methods.
