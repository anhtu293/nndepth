"""
BaseConfiguration class for creating hybrid configuration systems.

This class supports:
- Loading from YAML files
- Command line argument generation and parsing
- Nested configurations
- Type annotations with automatic CLI generation
- Custom help messages using typing.Annotated

Example:
    from typing import Annotated
    from nndepth.utils import BaseConfiguration

    class ModelConfig(BaseConfiguration):
        lr: Annotated[float, "Learning rate for training"] = 0.001
        batch_size: Annotated[int, "Number of samples per batch"] = 32
        debug: bool = False

    # Automatically generates CLI arguments:
    # --lr: Learning rate for training (default: 0.001)
    # --batch_size: Number of samples per batch (default: 32)
    # --debug, --no-debug: Set debug. (default: False)

    # Usage:
    parser = argparse.ArgumentParser()
    ModelConfig.add_args(parser)
    args = parser.parse_args()
    config = ModelConfig.from_args(args)
"""

import argparse
from argparse import BooleanOptionalAction
import yaml
from typing import Dict, Any, Optional, get_type_hints, get_origin, List, get_args

# Import Annotated with fallback for older Python versions
try:
    from typing import Annotated  # type: ignore
except ImportError:
    try:
        from typing_extensions import Annotated  # type: ignore
    except ImportError:
        # Annotated not available - will be handled gracefully in usage
        Annotated = None  # type: ignore

from loguru import logger


class BaseConfiguration:
    """
    Base configuration class that supports loading from YAML files and command line arguments.
    Command line arguments take priority over YAML values.
    Supports nested configurations.

    Usage:
        # Define your config classes with type annotations
        class DatasetConfig(BaseConfiguration):
            path: str = "/data"
            batch_size: int = 32

        class ModelConfig(BaseConfiguration):
            lr: float = 0.001
            dataset: DatasetConfig = DatasetConfig()  # Nested config

        # From CLI args only
        config = ModelConfig.from_args(args)

        # From YAML + CLI args (if config_file is provided in args)
        config = ModelConfig.from_args(args)  # will auto-load YAML if args.config_file exists

        # Add CLI arguments automatically (including --config_file)
        parser = ModelConfig.add_args(parser)
        # This will create: --model_lr, --model_dataset_path, --model_dataset_batch_size, etc.
    """

    def __init__(self, **kwargs):
        """Initialize configuration with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary, handling nested configurations."""
        # Get type hints to identify nested configurations
        type_hints = get_type_hints(cls)

        # Create a new dictionary for processed values
        processed_dict = {}

        for key, value in config_dict.items():
            if key in type_hints:
                expected_type = type_hints[key]

                # Check if this is a nested BaseConfiguration
                is_nested_config = (isinstance(expected_type, type) and
                                    issubclass(expected_type, BaseConfiguration) and
                                    isinstance(value, dict))

                if is_nested_config:
                    # Recursively create nested configuration
                    processed_dict[key] = expected_type._from_dict(value)
                else:
                    # Handle type conversion for special cases
                    if (expected_type == tuple or get_origin(expected_type) == tuple) and isinstance(value, list):
                        # Convert list to tuple for tuple type annotations
                        processed_dict[key] = tuple(value)
                    else:
                        processed_dict[key] = value
            else:
                processed_dict[key] = value

        return cls(**processed_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """
        Load configuration from command line arguments.
        If config_file is provided in args, load from YAML and override with CLI args.

        Args:
            args: Parsed command line arguments
        """
        config_dict = {}

        # Check if config_file is provided
        config_file = getattr(args, "config_file", None)

        if config_file:
            # Load from YAML first
            with open(config_file, "r") as f:
                yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
                config_dict = yaml_dict
            logger.info(f"Loaded configuration from {config_file}")

        # Get the argument information to map CLI args to nested paths
        arg_info = cls._collect_all_args()
        final_args = cls._resolve_arg_conflicts(arg_info)

        # Create mapping from final argument names to their nested paths
        arg_mapping = {}
        for arg_data in final_args:
            final_name = arg_data["final_name"]
            arg_mapping[final_name] = arg_data["full_path"]

        # Override with command line arguments using proper mapping
        args_dict = vars(args)
        nested_args = {}

        for key, value in args_dict.items():
            if value is None:
                continue

            if key != "config_file":
                if key in arg_mapping:
                    nested_path = arg_mapping[key]
                    cls._set_nested_value_by_path(nested_args, nested_path, value)

        # Merge nested args into config_dict
        cls._merge_nested_dicts(config_dict, nested_args)

        # Create configuration with nested support
        return cls._from_dict(config_dict)

    @classmethod
    def _set_nested_value_by_path(cls, nested_dict: Dict[str, Any], path: str, value: Any):
        """
        Set a value in nested dictionary structure based on dot-separated path.
        Example:
            nested_dict = {"a": {"b": {"c": 1}}}
            path = "a.b.c"
            value = 2
            _set_nested_value_by_path(nested_dict, path, value)
            # nested_dict is now {"a": {"b": {"c": 2}}}
            # This is used to set values in nested configurations

        Args:
            nested_dict: Dictionary to set the value in
            path: Path to set the value in
            value: Value to set

        Returns:
            None
        """
        parts = path.split(".")
        current = nested_dict

        # Navigate to the correct nested level
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value

    @classmethod
    def _merge_nested_dicts(cls, base_dict: Dict[str, Any], override_dict: Dict[str, Any]):
        """
        Recursively merge override_dict into base_dict.
        Example:
            base_dict = {"a": {"b": {"c": 1}}}
            override_dict = {"a": {"b": {"c": 2}}}
            _merge_nested_dicts(base_dict, override_dict)
            # base_dict is now {"a": {"b": {"c": 2}}}

        Args:
            base_dict: Dictionary to merge into
            override_dict: Dictionary to merge from

        Returns:
            None
        """
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                cls._merge_nested_dicts(base_dict[key], value)
            else:
                base_dict[key] = value

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Automatically add command line arguments based on type annotations.
        Also adds a --config_file argument for loading YAML configuration.
        Supports nested configurations with intelligent conflict resolution.

        Args:
            parser: ArgumentParser to add arguments to

        Returns:
            Modified ArgumentParser
        """
        # Add config_file argument
        parser.add_argument(
            "--config_file",
            type=str,
            help="Path to YAML configuration file"
        )

        # Collect all argument information first
        arg_info = cls._collect_all_args()

        # Resolve conflicts and determine final argument names
        final_args = cls._resolve_arg_conflicts(arg_info)

        # Actually add the arguments to the parser
        for arg_data in final_args:
            cls._add_single_arg(parser, arg_data)

        return parser

    @classmethod
    def _collect_all_args(cls, path: str = "") -> List[Dict[str, Any]]:
        """
        Collect all argument information from the class and its nested configurations.

        Args:
            path: Current path for nested access (used for conflict resolution)

        Returns:
            List of dictionaries containing argument information
        """
        args_info = []

        # Get raw annotations to preserve Annotated metadata
        raw_annotations = getattr(cls, '__annotations__', {})
        # Get resolved type hints for nested config detection
        type_hints = get_type_hints(cls)

        # Get default values by creating a temporary instance
        temp_instance = cls()

        for attr_name in raw_annotations:
            if attr_name.startswith('_'):  # Skip private attributes
                continue

            current_path = f"{path}.{attr_name}" if path else attr_name

            # Get the raw annotation (preserves Annotated metadata)
            raw_type = raw_annotations[attr_name]
            # Get the resolved type for nested config detection
            resolved_type = type_hints.get(attr_name, raw_type)

            # Check if this is a nested BaseConfiguration using resolved type
            is_nested_config = (isinstance(resolved_type, type) and
                                issubclass(resolved_type, BaseConfiguration))

            if is_nested_config:
                # Recursively collect arguments from nested configuration
                nested_args = resolved_type._collect_all_args(current_path)
                args_info.extend(nested_args)
            else:
                # Add information for this argument
                default_value = getattr(temp_instance, attr_name, None)

                # Extract custom help message from Annotated type
                custom_help = None
                actual_type = resolved_type

                # Check if this is an Annotated type by looking for __metadata__ attribute
                if hasattr(raw_type, '__metadata__'):
                    # This is an Annotated type
                    args = get_args(raw_type)
                    if args:
                        actual_type = args[0]
                        # Look for string in the metadata
                        for metadata in getattr(raw_type, '__metadata__', []):
                            if isinstance(metadata, str):
                                custom_help = metadata
                                break

                args_info.append({
                    'attr_name': attr_name,
                    'attr_type': actual_type,
                    'default_value': default_value,
                    'full_path': current_path,
                    'preferred_name': attr_name,  # Start with shortest name
                    'final_name': None,  # Will be resolved later
                    'custom_help': custom_help
                })

        return args_info

    @classmethod
    def _resolve_arg_conflicts(cls, args_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve argument name conflicts by adding minimal prefixes.

        Args:
            args_info: List of argument information dictionaries

        Returns:
            List of argument information with resolved names
        """
        # Group arguments by their preferred names
        name_groups = {}
        for arg in args_info:
            preferred = arg['preferred_name']
            if preferred not in name_groups:
                name_groups[preferred] = []
            name_groups[preferred].append(arg)

        # Resolve conflicts
        for preferred_name, conflicts in name_groups.items():
            if len(conflicts) == 1:
                # No conflict, use the preferred name
                conflicts[0]['final_name'] = preferred_name
            else:
                # There are conflicts, need to add minimal prefixes
                cls._resolve_conflict_group(conflicts)

        return args_info

    @classmethod
    def _resolve_conflict_group(cls, conflicts: List[Dict[str, Any]]):
        """
        Resolve conflicts within a group of arguments with the same preferred name.

        Args:
            conflicts: List of conflicting argument information
        """
        # For each conflict, try adding minimal prefixes until all are unique
        for i, arg in enumerate(conflicts):
            path_parts = arg["full_path"].split(".")
            attr_name = path_parts[-1]

            # Try adding prefixes from the path until we get a unique name
            for j in range(len(path_parts) - 1, 0, -1):
                # Build prefix from path parts using dots
                prefix_parts = path_parts[j - 1:j]
                if prefix_parts:
                    test_name = f"{'.'.join(prefix_parts)}.{attr_name}"
                else:
                    test_name = attr_name

                # Check if this name conflicts with any other final names
                is_unique = True
                for other_arg in conflicts:
                    if other_arg != arg and other_arg.get("final_name") == test_name:
                        is_unique = False
                        break

                if is_unique:
                    # Also check against all other resolved names
                    for other_i, other_arg in enumerate(conflicts):
                        if other_i != i and other_arg.get("final_name") == test_name:
                            is_unique = False
                            break

                    if is_unique:
                        arg["final_name"] = test_name
                        break

            # If still not resolved, use the full path with dots
            if arg["final_name"] is None:
                path_parts = arg["full_path"].split(".")
                if len(path_parts) > 1:
                    full_prefix = ".".join(path_parts[:-1])
                    arg["final_name"] = f"{full_prefix}.{path_parts[-1]}"
                else:
                    arg["final_name"] = attr_name

    @classmethod
    def _add_single_arg(cls, parser: argparse.ArgumentParser, arg_data: Dict[str, Any]):
        """
        Add a single argument to the parser.

        Args:
            parser: ArgumentParser to add argument to
            arg_data: Dictionary containing argument information
        """
        arg_name = f"--{arg_data['final_name']}"
        attr_type = arg_data["attr_type"]
        default_value = arg_data["default_value"]
        attr_name_for_help = arg_data["attr_name"]
        custom_help = arg_data.get("custom_help")

        # Use custom help message if provided, otherwise generate default
        if custom_help:
            base_help = custom_help
        else:
            base_help = f"Set {attr_name_for_help}"

        # Handle different types
        if attr_type == bool:
            help_text = f"{base_help}. (default: {default_value}). Use --no-{arg_name[2:]} to set to inverse value."
            parser.add_argument(
                arg_name,
                action=BooleanOptionalAction,
                help=help_text
            )
        elif attr_type == list or get_origin(attr_type) == list:
            help_text = f"{base_help} (list, default: {default_value})"
            parser.add_argument(
                arg_name,
                nargs="+",
                help=help_text
            )
        elif attr_type == tuple or get_origin(attr_type) == tuple:
            help_text = f"{base_help} (tuple, default: {default_value})"
            parser.add_argument(
                arg_name,
                nargs="+",
                help=help_text
            )
        else:
            # Handle basic types (int, float, str)
            help_text = f"{base_help} (default: {default_value})"
            if attr_type in (int, float, str):
                parser.add_argument(
                    arg_name,
                    type=attr_type,
                    default=None,
                    help=help_text
                )
            else:
                # For other types, treat as string
                parser.add_argument(
                    arg_name,
                    type=str,
                    default=None,
                    help=help_text
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, handling nested configurations."""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):  # Skip private attributes
                continue

            if isinstance(value, BaseConfiguration):
                # Recursively convert nested configurations
                result[key] = value.to_dict()
            else:
                result[key] = value

        return result

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        """String representation of configuration."""
        items = []
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                items.append(f"{key}=<nested>")
            else:
                items.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(items)})"

    def print_summary(self, title: Optional[str] = None, indent: int = 0):
        """Print a nicely formatted configuration summary."""
        if title is None:
            title = f"{self.__class__.__name__} Configuration"

        indent_str = "  " * indent

        if indent == 0:
            logger.info(f"\n{title}:")
            logger.info("=" * len(title))
        else:
            logger.info(f"{indent_str}{title}:")
            logger.info(f"{indent_str}" + "-" * len(title))

        for key, value in self.__dict__.items():
            if key.startswith("_"):  # Skip private attributes
                continue

            if isinstance(value, BaseConfiguration):
                # Recursively print nested configurations
                value.print_summary(f"{key.title()} Configuration", indent + 1)
            else:
                logger.info(f"{indent_str}  {key}: {value}")

        if indent == 0:
            logger.info("")
