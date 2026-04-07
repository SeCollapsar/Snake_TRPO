import numpy as np


def flat_grad(grads):
    return np.concatenate([g.flatten() for g in grads])


def kl_divergence(old_probs, new_probs):
    return np.sum(old_probs * (np.log(old_probs + 1e-8) - np.log(new_probs + 1e-8)))    