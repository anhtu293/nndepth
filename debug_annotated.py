#!/usr/bin/env python3
"""Debug script to test Annotated type detection."""

from typing import Annotated, get_type_hints, get_args
from nndepth.utils import BaseConfiguration


class TestConfig(BaseConfiguration):
    """Test config for debugging."""

    test_field: Annotated[str, "This is a custom help message"] = "default"
    normal_field: str = "normal"


def debug_annotated():
    """Debug the Annotated type detection."""

    # Get raw annotations (preserves Annotated)
    raw_annotations = getattr(TestConfig, '__annotations__', {})
    # Get resolved type hints (strips Annotated)
    type_hints = get_type_hints(TestConfig)

    print("Raw annotations:")
    for name, annotation in raw_annotations.items():
        print(f"  {name}: {annotation}")
        print(f"    Has __metadata__: {hasattr(annotation, '__metadata__')}")
        if hasattr(annotation, '__metadata__'):
            print(f"    Metadata: {getattr(annotation, '__metadata__', [])}")
            args = get_args(annotation)
            print(f"    Args: {args}")
        print()

    print("Type hints (resolved):")
    for name, hint in type_hints.items():
        print(f"  {name}: {hint}")
        print()

    # Test the collection method
    print("Testing _collect_all_args:")
    args_info = TestConfig._collect_all_args()
    for arg_info in args_info:
        print(f"  {arg_info['attr_name']}: custom_help = {arg_info.get('custom_help', 'None')}")


if __name__ == "__main__":
    debug_annotated()
