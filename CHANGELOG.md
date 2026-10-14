# v0.2

- Added cascaded PID with conditional anti-windup and bounded integral state.
- Added augmented-state Kalman disturbance observation and compensation.
- Added a planar matched-channel L1 augmentation on PID, with explicit limitations.
- Added nonlinear rotor-input MPC, disturbance estimation, gradients and solver diagnostics.
- Added unmodeled drag/damping stress conditions and model-error, frequency,
  correlation, dwell-time and switch-count configurations.
- Added controller/seed/variant CLI selection and backward-compatible defaults.
- Reporting supports all controllers, consistent colors, pitch, extra inputs,
  solver timing, paired differences for three metrics and transition/later costs.
- Included 19 passing validation tests and 63 bounded integration-check runs.
- Kept original five-controller configurations and original smoke data.
- RL, actuator lag and noisy observations remain deferred.
