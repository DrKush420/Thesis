import importlib.util
import sys


class Params:
    """Dummy class for storing (hyper) parameters and objects
    """
    def __init__(self):
        """Set the default values here
        """
        self.num_workers = 8


def get_config_from_path(path):
    """Return python config parameters from file path

    code from: https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location("config", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config"] = mod
    spec.loader.exec_module(mod)

    return mod.params
