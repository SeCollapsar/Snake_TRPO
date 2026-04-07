import numpy as np


def conjugate_gradient(Avp, b, iters=10):

    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()

    rsold = np.dot(r, r)

    for _ in range(iters):

        Ap = Avp(p)
        alpha = rsold / (np.dot(p, Ap) + 1e-8)

        x += alpha * p
        r -= alpha * Ap

        rsnew = np.dot(r, r)

        if rsnew < 1e-10:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x