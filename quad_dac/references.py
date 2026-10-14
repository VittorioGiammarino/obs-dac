import numpy as np

def reference(t, c):
    """Return [px, pz, vx, vz, ax, az]."""
    if c['kind'] == 'hover':
        return np.array([0., 2., 0., 0., 0., 0.])

    a = c['amplitude']
    b = c.get('vertical_ratio', 0.3) * a
    w = 2 * np.pi * c['frequency']

    if c['kind'] == 'figure8':
        return np.array([
            a * np.sin(w*t),
            2. + b * np.sin(2*w*t),
            a*w * np.cos(w*t),
            2*b*w * np.cos(2*w*t),
            -a*w*w * np.sin(w*t),
            -4*b*w*w * np.sin(2*w*t),
        ])

    # Original elliptical reference.
    return np.array([
        a * np.sin(w*t),
        2. + b * np.cos(w*t),
        a*w * np.cos(w*t),
        -b*w * np.sin(w*t),
        -a*w*w * np.sin(w*t),
        -b*w*w * np.cos(w*t),
    ])
