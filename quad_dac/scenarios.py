import numpy as np
from .references import reference

def generate(cfg, seed):
    c = cfg['scenario']
    dt = cfg['dt']
    nseg = c['switches'] + 1

    # Independent streams: order, stochastic disturbances, phase.
    children = np.random.SeedSequence(seed).spawn(3)
    order_rng = np.random.default_rng(children[0])
    noise_rng = np.random.default_rng(children[1])
    phase_rng = np.random.default_rng(children[2])

    order = (c['regimes']*((nseg + len(c['regimes']) - 1) // len(c['regimes'])))[:nseg]

    if c.get('randomized', False):
        order = list(order_rng.permutation(order))

    steps = max(1, round(c['duration'] / dt))
    N = steps * nseg
    d = np.zeros((N, 2))
    labels = []
    amp = np.asarray(c['amplitude'])
    t = np.arange(steps) * dt

    for j, name in enumerate(order):
        noise = noise_rng.normal(size=(steps, 2))

        # Common random phase preserves the prescribed inter-axis offset.
        phi = (phase_rng.uniform(0., 2 * np.pi) if c.get('random_phase', False) else 0.)
        axis_phase = phi + np.array([0., 0.7])

        if name == 'none':
            y = np.zeros((steps, 2))

        elif name == 'iid':
            y = noise

        elif name == 'bias':
            y = np.ones((steps, 2))

        elif name == 'periodic':
            phase = 2 * np.pi * c['frequency'] * t
            y = np.sin(phase[:, None] + axis_phase)

        elif name == 'chirp':
            phase = 2 * np.pi * (c['frequency'] * t + 0.5 * c['chirp_rate'] * t**2)
            y = np.sin(phase[:, None] + axis_phase)

        elif name == 'gust':
            rho = np.exp(-dt / c['correlation_time'])
            innovation_std = np.sqrt(1 - rho**2)
            y = np.empty_like(noise)
            prev = noise_rng.normal(size=2)

            for k in range(steps):
                prev = rho * prev + innovation_std * noise[k]
                y[k] = prev

        elif name == 'oscillatory_gust':
            rho = np.exp(-dt / c['correlation_time'])
            theta = 2 * np.pi * c['frequency'] * dt

            a1 = 2 * rho * np.cos(theta)
            a2 = -(rho**2)

            innovation_var = ((1 + a2)*((1 - a2)**2 - a1**2)/(1 - a2))
            innovation_std = np.sqrt(max(innovation_var, 0.))

            burn = max(100, int(np.ceil(10 * c['correlation_time'] / dt)),)
            innovations = noise_rng.normal(size=(burn + steps, 2))

            y = np.empty((steps, 2))
            prev = np.zeros(2)
            prev2 = np.zeros(2)

            for k, eps in enumerate(innovations):
                current = (a1 * prev + a2 * prev2 + innovation_std * eps)
                prev2, prev = prev, current

                if k >= burn:
                    y[k - burn] = current

        else:
            raise ValueError(name)

        d[j * steps:(j + 1) * steps] = y * amp
        labels.extend([name] * steps)

    refs = np.array([reference(k * dt, cfg['reference']) for k in range(N + 1)])
    x0 = (np.r_[refs[0, :4], 0., 0.] + np.asarray(cfg['initial_offset']))

    return d, refs, np.array(labels), x0, order
