"""pytest configuration: skip markers (CUDA, VeRL, the original experiment code) and a clean QURL_* environment."""
import importlib.util
import os
import sys

import pytest
import torch

sys.dont_write_bytecode = True   # never write bytecode, in particular not next to the original experiment code
torch.set_num_threads(1)         # the models are tiny: more threads only add overhead (and the comparisons are
#                                  between runs in this process, so the thread count cannot affect them)

from qurl_test_support import have_originals  # noqa: E402


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if item.get_closest_marker("verl") and importlib.util.find_spec("verl") is None:
        pytest.skip("needs VeRL")
    if item.get_closest_marker("originals") and not have_originals():
        pytest.skip("the original experiment code is not in this checkout")


@pytest.fixture(autouse=True)
def clean_qurl_env(monkeypatch):
    """No QURL_* variable of the outer environment leaks into a test; no Hugging Face Hub access."""
    for k in list(os.environ):
        if k.startswith("QURL_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)
