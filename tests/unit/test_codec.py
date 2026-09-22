"""The q8rle codec."""

from __future__ import annotations

import numpy as np
import pytest

from spacepresso.core import codec


def test_round_trip_is_exact_to_quantisation():
    rng = np.random.default_rng(0)
    original = rng.random((32, 48)).astype(np.float32)
    decoded = codec.decode_to_float(codec.encode_float(original))
    # 8-bit quantisation: the worst case is half a level, 1/510.
    assert np.abs(decoded - original).max() <= 1.0 / 510 + 1e-6


def test_uint8_round_trip_is_lossless():
    rng = np.random.default_rng(1)
    original = rng.integers(0, 256, size=(17, 23), dtype=np.uint8)
    assert np.array_equal(codec.decode_to_uint8(codec.encode_uint8(original)), original)


def test_shape_is_preserved_when_not_square():
    original = np.zeros((7, 13), dtype=np.uint8)
    assert codec.decode_to_uint8(codec.encode_uint8(original)).shape == (7, 13)
    assert codec.shape_of(codec.encode_uint8(original)) == (7, 13)


def test_constant_image_encodes_to_one_run():
    payload = codec.encode_uint8(np.full((16, 16), 42, dtype=np.uint8))
    assert payload.split() == ["q8rle", "16", "16", "42", "256"]


def test_empty_body_decodes_to_zeros():
    """A payload with a header and no runs is well-formed and means "all zeros".
    That decoder raised on it instead.

    """
    assert np.array_equal(
        codec.decode_to_uint8("q8rle 4 5"), np.zeros((4, 5), np.uint8)
    )
    assert codec.decode_to_float("q8rle 4 5").shape == (4, 5)


def test_column_major_layout():
    """Runs are laid out down columns, not across rows."""
    matrix = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    assert codec.encode_uint8(matrix).split()[3:] == [
        "1",
        "1",
        "3",
        "1",
        "2",
        "1",
        "4",
        "1",
    ]


def test_values_are_clipped_not_wrapped():
    payload = codec.encode_float(np.array([[-1.0, 2.0]], dtype=np.float32))
    assert np.array_equal(
        codec.decode_to_uint8(payload), np.array([[0, 255]], np.uint8)
    )


@pytest.mark.parametrize(
    "payload, reason",
    [
        ("rle 2 2 0 4", "wrong magic"),
        ("q8rle 2", "truncated header"),
        ("q8rle 2 2 5", "odd body"),
        ("q8rle 2 2 5 3", "runs do not sum to H*W"),
        ("q8rle 2 2 300 4", "value out of range"),
    ],
)
def test_malformed_payloads_raise(payload, reason):
    with pytest.raises(codec.Q8RLEError):
        codec.decode_to_uint8(payload)


def test_encode_rejects_non_2d():
    with pytest.raises(codec.Q8RLEError):
        codec.encode_uint8(np.zeros((2, 2, 2), dtype=np.uint8))
