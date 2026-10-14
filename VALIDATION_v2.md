# Validation record — v0.2

These are implementation checks, not held-out evaluation or controller rankings.

- 19 tests passed; exact log: `validation_v2.log`.
- Checks include RK4/nominal/MPC/DAC sensitivities, hover, no-wind tracking,
  rotor feasibility, PID anti-windup, observer convergence/covariance,
  L1 predictor identity and matched/unmatched channels, causality, reset,
  recovery censoring and failure-tail accounting.
- 54 paired short runs: six conditions, nine controller/model combinations,
  one seed (901), three seconds each.
- Nine additional paired switched runs: seed 902, IID → bias → periodic,
  two seconds per regime, six seconds total; controller states carry across switches.
- No predefined failure threshold triggered in these 63 runs.
- Full 20-seed evaluations, tuning sweeps and full stress matrix were NOT run.
- JSON metrics, NPZ traces and PNG/PDF reports are in `results/controllers_checks`
  and `results/controllers_switch_check`. No bootstrap intervals are inferred
  from these single-seed checks.

## MPC computation check

The implementation optimizes in Python/SciPy without a strict real-time deadline.
It uses a finite iteration budget and logs nonconvergence; the best feasible
iterate is applied. These are not certified optimal solutions. Simulated
plant evolution does not include computation delay. Rates below should not be
interpreted as real-time performance of optimized MPC software.

| Condition | Mean controller time (ms) | Nonconverged solves / steps | Deadline-miss fraction |
|---|---:|---:|---:|
| base | 182.3 | 7 / 75 | 1.000 |
| inertia_error | 201.2 | 9 / 75 | 1.000 |
| mass_error | 230.1 | 6 / 75 | 1.000 |
| unmodeled_drag | 145.9 | 4 / 75 | 0.960 |
| periodic | 185.9 | 3 / 75 | 1.000 |
| correlated_gust | 452.1 | 41 / 75 | 1.000 |
| switched | 405.5 | 36 / 150 | 1.000 |

The source hashes, parameters, platform and dependency versions for each execution are recorded in each results directory. The reporting script was then updated to render the expanded controller set.
