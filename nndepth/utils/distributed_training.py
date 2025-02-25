import os


def is_distributed_training() -> bool:
    return os.environ.get("RANK", -1) != -1


def is_main_process() -> bool:
    return os.environ.get("RANK", -1) == 0
