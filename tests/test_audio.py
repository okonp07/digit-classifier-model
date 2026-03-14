import numpy as np

from digit_recognition.audio import AudioProcessor
from digit_recognition.datasets import AudioRecord, group_split_records


def test_pad_or_trim_enforces_fixed_length():
    processor = AudioProcessor(sample_rate=10, max_duration=1.0)
    short_audio = np.ones(6, dtype=np.float32)
    long_audio = np.ones(20, dtype=np.float32)

    assert len(processor.pad_or_trim(short_audio)) == 10
    assert len(processor.pad_or_trim(long_audio)) == 10


def test_group_split_keeps_all_records():
    records = [
        AudioRecord(path=None, label=idx % 2, source="fsdd", group=f"speaker-{idx // 2}")  # type: ignore[arg-type]
        for idx in range(8)
    ]
    train, val = group_split_records(records, test_size=0.25, random_state=42)
    assert len(train) + len(val) == len(records)
    assert set(train).isdisjoint(set(val))
