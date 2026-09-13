import torch


def pytest_sessionstart(session):
    # Tiny mathematical tests are slower and less reproducible with 24 threads.
    torch.set_num_threads(1)
