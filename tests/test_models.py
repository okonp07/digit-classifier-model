import torch

from digit_recognition.model import LightweightDigitCNN


def test_model_forward_shape():
    model = LightweightDigitCNN()
    batch = torch.randn(4, 13, 87)
    output = model(batch)
    assert output.shape == (4, 10)
