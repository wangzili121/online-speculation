import itertools

import pytest
import torch

from blockspec_ablation.diffusion import UniformNoise, corrupt, posterior, psi_transition


def test_default_noise_preserves_original_generator_sequence():
    actual_rng, expected_rng = (torch.Generator().manual_seed(818) for _ in range(2))
    for shape in ((1, 7), (2, 0), (3, 11)):
        actual = UniformNoise().sample(shape, 13, device="cpu", generator=actual_rng)
        expected = torch.randint(13, shape, generator=expected_rng)
        assert torch.equal(actual, expected)
        assert torch.equal(actual_rng.get_state(), expected_rng.get_state())


@pytest.mark.parametrize("noise,high", [(UniformNoise(1), 13), (UniformNoise(2, 7), 7)])
def test_uniform_noise_uses_exact_integer_interval(noise, high):
    actual_rng, expected_rng = (torch.Generator().manual_seed(819) for _ in range(2))
    actual = noise.sample((3, 200), 13, device="cpu", generator=actual_rng)
    expected = torch.randint(noise.low, high, (3, 200), generator=expected_rng)
    assert torch.equal(actual, expected)
    assert set(actual.flatten().tolist()) == set(range(noise.low, high))


@pytest.mark.parametrize("low,high", [(-1, None), (True, None), (1.5, None),
                                      (0, True), (0, 1.5), (0, 0), (3, 2)])
def test_noise_requires_valid_integer_bounds(low, high):
    with pytest.raises(ValueError, match="integer bounds"):
        UniformNoise(low, high)


@pytest.mark.parametrize("noise", [UniformNoise(13), UniformNoise(14), UniformNoise(0, 14)])
def test_noise_bounds_are_checked_against_vocabulary(noise):
    with pytest.raises(ValueError, match="model vocabulary"):
        noise.sample((1, 3), 13, device="cpu")


@pytest.mark.parametrize("alpha_s,alpha_t", [(1., 0.), (.8, .3), (.7, 0.), (1., .7)])
def test_reverse_posterior_against_enumerated_markov_joint(alpha_s, alpha_t):
    prior = torch.tensor([.1, .3, .6], dtype=torch.float64)
    for clean, observed in itertools.product(range(3), repeat=2):
        x = torch.nn.functional.one_hot(torch.tensor(clean), 3).double()
        got = posterior(torch.tensor(observed), x, prior, alpha_s, alpha_t)
        joint = torch.empty(3, dtype=torch.float64)
        for intermediate in range(3):
            first = alpha_s * (intermediate == clean) + (1 - alpha_s) * prior[intermediate]
            ratio = alpha_t / alpha_s
            second = ratio * (intermediate == observed) + (1 - ratio) * prior[observed]
            joint[intermediate] = first * second
        torch.testing.assert_close(got, joint / joint.sum(), atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize("kappa", [0., .3, 1.])
def test_one_step_psi_reduces_to_prediction(kappa):
    prior = torch.tensor([.1, .3, .6], dtype=torch.float64)
    q = torch.tensor([.4, .5, .1], dtype=torch.float64)
    for token in range(3):
        got = psi_transition(torch.tensor(token), q, prior, 1., 0., kappa)
        torch.testing.assert_close(got, q, atol=1e-15, rtol=1e-15)


def test_corruption_endpoints():
    clean = torch.tensor([[0, 1, 2], [2, 1, 0]])
    prior = torch.tensor([0., 1., 0.])
    assert torch.equal(corrupt(clean, prior, 1), clean)
    assert torch.equal(corrupt(clean, prior, 0), torch.ones_like(clean))
