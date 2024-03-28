import json
import os
import platform

import importlib_resources
from cmdstanpy import CmdStanModel

from typing import Optional

from ..utils.logger import get_logger

logger = get_logger("orbit")
# Update this to read from orbit/config.json/ORBIT_MODELS for consistency?
MODELS = ["dlt", "ets", "lgt", "ktrlite"]

# Properly set up compiler paths again
orbit_config = importlib_resources.files("orbit") / "config.json"
IS_WINDOWS = platform.platform().startswith("Win")
if IS_WINDOWS:
    with open(orbit_config) as f:
        config = json.load(f)
    CMDSTAN_VERSION = config["CMDSTAN_VERSION"]
    paths = [
        f"~/.cmdstan/cmdstan-{CMDSTAN_VERSION}/stan/lib/stan_math/lib/tbb",
        "~/.cmdstan/RTools40/mingw64/bin",
        "~/.cmdstan/RTools40/usr/bin",
    ]
    for path_string in paths:
        # or use sys.path.append, potentially works across platforms
        os.environ["Path"] += ";" + os.path.normpath(os.path.expanduser(path_string))


def get_compiled_stan_model(
    stan_model_name: str = "",
    stan_file_path: Optional[str] = None,
    exe_file_path: Optional[str] = None,
    force_compile: bool = False,
) -> CmdStanModel:
    """Return a compiled Stan model using CmdStan.
    This includes both prepackaged models as well as user provided models through stan_file_path.

    Parameters
    ----------
    stan_model_name :
        The name of the Stan model to use. Use this for the built in models (dlt, ets, ktrlite, lgt)
    stan_file_path :
        The path to the Stan file to use. If not provided, the default is to search for the file in the 'orbit' package.
        If provided, function will ignore the stan_model_name parameter, and will compile the provide stan_file_path
        into executable in place (same folder as stan_file_path)
    exe_file_path :
        The path to the Stan-exe file to use. If not provided, the default is to search for the file
        in the 'orbit' package. If provided, function will ignore the stan_model_name parameter,
        and will compile the provide stan_file_path into executable in place (same folder as stan_file_path)
    Returns
    -------
    sm : CmdStanModel
        A compiled Stan model.
    """
    if (stan_file_path is not None) or (exe_file_path is not None):
        sm = CmdStanModel(
            stan_file=stan_file, exe_file=exe_file_path, force_compile=force_compile
        )
    else:
        # Load orbit included cmdstan models
        # Some oddities here. if not providing exe_file, CmdStanModel would delete the actual executable file.
        # This is a stop gap fix until actual cause is identified.
        stan_file = importlib_resources.files("orbit") / f"stan/{stan_model_name}.stan"
        EXTENSION = ".exe" if platform.system() == "Windows" else ""
        exe_file = (
            importlib_resources.files("orbit") / f"stan/{stan_model_name}{EXTENSION}"
        )
        # Check if exe is older than .stan file.
        # This behavior is default on CmdStanModel if we don't have to specify the exe_file.
        # if not os.path.isfile(exe_file) or (
        #     os.path.getmtime(exe_file) <= os.path.getmtime(stan_file)
        # ):

        if not os.path.isfile(exe_file) or force_compile:
            logger.info(f"Compiling stan model:{stan_file}. ETA 3 - 5 mins.")
            sm = CmdStanModel(stan_file=stan_file)
        else:
            sm = CmdStanModel(stan_file=stan_file, exe_file=exe_file)

    return sm


# TODO: Is this needed?
class suppress_stdout_stderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
