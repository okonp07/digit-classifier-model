from types import SimpleNamespace

from digit_recognition.transcriber import (
    TranscriptionResult,
    TranscriptionSegment,
    _probability_from_logprob,
    _weighted_confidence,
)


def test_probability_from_logprob_is_clamped():
    assert _probability_from_logprob(None) == 0.0
    assert _probability_from_logprob(0.0) == 1.0
    assert 0.0 < _probability_from_logprob(-2.0) < 1.0
    assert _probability_from_logprob(2.0) == 1.0


def test_weighted_confidence_uses_segment_duration():
    short_low = SimpleNamespace(start=0.0, end=0.5, avg_logprob=-2.0, text="short", words=None)
    long_high = SimpleNamespace(start=0.5, end=3.0, avg_logprob=-0.05, text="much longer", words=None)

    confidence = _weighted_confidence([short_low, long_high])

    assert confidence > 0.8


def test_transcription_result_to_dict_includes_segments():
    result = TranscriptionResult(
        text="hello world",
        confidence=0.91,
        language="en",
        language_confidence=0.98,
        duration_seconds=1.4,
        segments=(
            TranscriptionSegment(
                start_seconds=0.0,
                end_seconds=1.4,
                text="hello world",
                confidence=0.91,
            ),
        ),
    )

    payload = result.to_dict()

    assert payload["text"] == "hello world"
    assert payload["language"] == "en"
    assert payload["segments"][0]["text"] == "hello world"
