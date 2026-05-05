"""
Re-export misc functions from utils.misc to maintain backward compatibility.
"""
from utils.misc import (
    velocity_prediction,
    next_token_prediction,
    interpolate_pos_encoding,
)

__all__ = [
    'velocity_prediction',
    'next_token_prediction',
    'interpolate_pos_encoding',
]


