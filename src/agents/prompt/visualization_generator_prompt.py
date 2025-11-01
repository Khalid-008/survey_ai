"""
Visualization generator prompts.
This module imports and re-exports visualization prompts for backward compatibility.
"""

from .visualization_generator_from_quantitative_data_prompt import (
    visualization_generator_from_quantitative_data_prompt
)
from .visualization_generator_from_qualitative_data_prompt import (
    visualization_generator_from_qualitative_data_prompt
)

__all__ = [
    'visualization_generator_from_quantitative_data_prompt',
    'visualization_generator_from_qualitative_data_prompt'
]
