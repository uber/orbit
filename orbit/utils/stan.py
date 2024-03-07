import json
import os
import platform
from multiprocessing import cpu_count

import cmdstanpy
import importlib_resources
from cmdstanpy import CmdStanModel

from ..utils.logger import get_logger

logger = get_logger("orbit")
# Update this to read from orbit/config.json/ORBIT_MODELS for consistency?
MODELS = ["dlt", "ets", "lgt", "ktrlite"]


def get_compiled_stan_model(
    stan_model_name: str = "", stan_file_path: str = ""
) -> CmdStanModel:
    """Return a compiled Stan model using CmdStan.
    This includes both prepackaged models as well as user provided models through stan_file_path.

    Parameters
    ----------
    stan_model_name : str
        The name of the Stan model to use. Use this for the built in models (dlt, ets, ktrlite, lgt)
    stan_file_path : str, optional
        The path to the Stan file to use. If not provided, the default is to search for the file in the 'orbit' package.
        If provided, function will ignore the stan_model_name parameter, and will compile the provide stan_file_path
        into executable in place (same folder as stan_file_path)
    Returns
    -------
    sm : CmdStanModel
        A compiled Stan model.
    """
    if stan_model_name not in MODELS:
        raise ValueError("stan_model_name must be one of dlt, ets, ktrlite, lgt")
    if stan_file_path != "":
        # On windows, compiler path is often not set properly after cmdstanpy_install
        # For safety, consider running it again. Below is the function from setup.py
        """
        IS_WINDOWS = platform.platform().startswith("Win")
        with open("orbit/config.json") as f: # Need to fix path to read from proper package.
            config = json.load(f)
        CMDSTAN_VERSION = config["CMDSTAN_VERSION"]

        if not cmdstanpy.install_cmdstan(
            version=CMDSTAN_VERSION,
            overwrite=True,
            verbose=True,
            cores=cpu_count(),
            progress=True,
            compiler=IS_WINDOWS,
        ):
            raise RuntimeError("CmdStan failed to install.")
        print(f"Installed cmdstanpy (cmdstan v.{CMDSTAN_VERSION}) and compiler.")
        """

        stan_file = stan_file_path
        sm = CmdStanModel(stan_file=stan_file)
    else:
        # Load orbit included cmdstan models
        # Some oddities here. if not providing exe_file, CmdStanModel would delete the actual executable file.
        # This is a stop gap fix until actual cause is identified.
        stan_file = importlib_resources.files("orbit") / f"stan/{stan_model_name}.stan"
        EXTENSION = ".exe" if platform.system() == "Windows" else ""
        exe_file = (
            importlib_resources.files("orbit") / f"stan/{stan_model_name}{EXTENSION}"
        )
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
