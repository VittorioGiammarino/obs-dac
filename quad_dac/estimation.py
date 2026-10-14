import numpy as np

def estimate(model,x,u,next_x):
    w=next_x-model.step(x,u)
    # Velocity projection only. Other residual coordinates remain in replay.
    z=w[2:4]/model.dt
    return w,z

def history(features,t,H):
    return np.array([features[t-i-1] if t-i-1>=0 else np.zeros(2)
                     for i in range(H)])
