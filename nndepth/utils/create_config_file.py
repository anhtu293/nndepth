from argparse import ArgumentParser
import inspect
import importlib
import yaml
from typing import Type, List, Optional
from io import TextIOWrapper
from loguru import logger


def write_config_file_and_doc(
    file: TextIOWrapper,
    cls_type: Optional[Type] = None,
    module_path: Optional[str] = None,
    cls_name: Optional[str] = None,
    ignored_base_classes: List[str] = ["object", "Module"],
):
    """Create a config file for a class.

    Parameters
        file (TextIOWrapper): File to write the configuration file to.
        cls_type (Type, optional): Class to create the config file for. Defaults to None.
        module_path (str, optional): Path to the module containing the class. Defaults to None.
        cls_name (str, optional): Name of the class. Defaults to None.
        config (dict, optional): Configuration dictionary. Defaults to None.
        docstring (str, optional): Docstring for the class. Defaults to None.
        ignored_base_classes (List[str], optional): List of base classes to ignore. Defaults to ["object", "Module"].

    Returns:
        dict, str: Configuration dictionary and docstring.

    """
    assert cls_type is not None or (
        module_path is not None and cls_name is not None
    ), "Either cls_type or module_path and cls_name must be set."

    # Get cls_type if not provided
    if cls_type is None:
        if "/" in module_path:
            module_path = module_path.rstrip("/")
            module_path = module_path.replace("/", ".")
            if module_path.endswith(".py"):
                module_path = module_path[:-3]

        module = importlib.import_module(module_path)
        cls_type = getattr(module, cls_name, None)
        if cls_type is None:
            raise RuntimeError(f"Class {cls_name} not found in module {module_path}.")

    # Get the class constructor
    sig = inspect.signature(cls_type)
    params = sig.parameters
    config = {}
    for name, param in params.items():
        if name == "kwargs":
            continue
        if param.default == inspect.Parameter.empty:
            config[name] = "assign_value_here"
        else:
            config[name] = param.default

    # Reorder the configuration dictionary: argument without default value first
    ordered_config = {k: v for k, v in config.items() if v == "assign_value_here"}
    ordered_config.update({k: v for k, v in config.items() if v != "assign_value_here"})

    # Get constructor docstring
    docstring = "" if cls_type.__init__.__doc__ is None else cls_type.__init__.__doc__
    docstring = docstring.replace("\n", "\n# ")

    # Write the configuration file
    logger.info(f"Write configuration file for {cls_type.__name__}")
    file.write(f"# {cls_type.__name__} \n")
    if len(docstring) > 0:
        file.write(f"# {docstring} \n")
    if len(ordered_config) > 0:
        file.write(yaml.dump(ordered_config, sort_keys=False))
        file.write("\n")

    # Get parameters of base classes
    base_classes = cls_type.__bases__
    if len(base_classes) > 0:
        for base_cls in base_classes:
            if base_cls.__name__ in ignored_base_classes:
                continue
            write_config_file_and_doc(file=file, cls_type=base_cls, ignored_base_classes=ignored_base_classes)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--module_path", required=True, type=str, help="Path to the module containing the class.")
    parser.add_argument("--cls_name", required=True, type=str, help="Name of the class.")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save the config file.")
    parser.add_argument(
        "--ignore_base_classes",
        type=str,
        default=[],
        nargs="+",
        help="Base classes to ignore. Defaults to ['object', 'Module'].",
    )
    args = parser.parse_args()

    with open(args.save_path, "w") as f:
        f.write(f"# {args.cls_name} configuration file \n\n")
        f.write(f"name: {args.cls_name} \n")
        write_config_file_and_doc(
            file=f,
            module_path=args.module_path,
            cls_name=args.cls_name,
            ignored_base_classes=["object", "Module", *args.ignore_base_classes],
        )

    logger.success(f"Config file saved to {args.save_path}")
