"""Discrete corruption and predictor/corrector kernels derived with Bayes' rule."""

from dataclasses import dataclass

import torch

from blockspec.sampling import validate_distribution


@dataclass(frozen=True)
class UniformNoise:
    """Uniform integer proposal noise on [low, high); None resolves to vocabulary size."""

    low: int = 0
    high: int | None = None

    def __post_init__(self):
        if type(self.low) is not int or self.low < 0 or (self.high is not None and (
                type(self.high) is not int or self.high <= self.low)):
            raise ValueError("noise requires integer bounds 0 <= low < high")

    def sample(self, shape, vocab_size, *, device, generator=None):
        high = vocab_size if self.high is None else self.high
        if not self.low < high <= vocab_size:
            raise ValueError("noise bounds must fit the model vocabulary")
        return torch.randint(self.low, high, shape, device=device, generator=generator)


def corrupt(clean, prior, alpha, *, generator=None):
    validate_distribution(prior)
    if prior.ndim != 1 or not 0 <= alpha <= 1:
        raise ValueError("a one-dimensional prior and alpha in [0,1] are required")
    replacement = torch.multinomial(prior, clean.numel(), replacement=True,
                                    generator=generator).reshape_as(clean)
    retained = torch.rand(clean.shape, device=clean.device, generator=generator) < alpha
    return torch.where(retained, clean, replacement)


def posterior(z_t, clean_distribution, prior, alpha_s, alpha_t):
    """Plug-in reverse posterior for a reset-to-prior forward Markov chain.

    For a one-hot clean_distribution this is the exact Bayes posterior. For a
    predicted soft vector it is the paper's plug-in parameterization, not an
    assertion that a nonlinear substitution equals a posterior mixture.
    """
    validate_distribution(clean_distribution)
    validate_distribution(prior)
    if not 0 <= alpha_t < alpha_s <= 1:
        raise ValueError("require 0 <= alpha_t < alpha_s <= 1")
    if prior.ndim != 1 or clean_distribution.shape[-1] != prior.numel():
        raise ValueError("prior vocabulary mismatch")
    if z_t.shape != clean_distribution.shape[:-1]:
        raise ValueError("observed token shape mismatch")
    ratio = alpha_t / alpha_s
    one_hot = torch.nn.functional.one_hot(z_t, prior.numel()).to(clean_distribution.dtype)
    prior_observed = prior[z_t][..., None]
    likelihood = ratio * one_hot + (1 - ratio) * prior_observed
    at_s = alpha_s * clean_distribution + (1 - alpha_s) * prior
    numerator = at_s * likelihood
    denominator = numerator.sum(-1, keepdim=True)
    if (denominator <= 0).any():
        raise ValueError("conditioning on an impossible noisy observation")
    return numerator / denominator


def psi_transition(z_t, clean_distribution, prior, alpha_s, alpha_t, kappa=1.0):
    if not 0 <= kappa <= 1:
        raise ValueError("kappa must be in [0,1]")
    ancestral = posterior(z_t, clean_distribution, prior, alpha_s, alpha_t)
    clean_posterior = posterior(z_t, clean_distribution, prior, 1.0, alpha_t)
    corrected = alpha_s * clean_posterior + (1 - alpha_s) * prior
    return kappa * ancestral + (1 - kappa) * corrected
