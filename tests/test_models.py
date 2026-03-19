import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from digit_recognition.model import LightweightDigitCNN  # noqa: E402


def test_model_forward_shape():
    model = LightweightDigitCNN()
    batch = torch.randn(4, 13, 87)
    output = model(batch)
    assert output.shape == (4, 10)
