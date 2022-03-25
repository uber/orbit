from __future__ import absolute_import, division, print_function
from pystan import StanModel
import pickle
import pkg_resources
import os

from orbit.constants.constants import CompiledStanModelPath


def set_compiled_stan_path(parent, child="stan_compiled"):
    """
    Set the path for compiled stan models.

    parent: the primary directory level
    child: the secondary directory level
    """
    CompiledStanModelPath.PARENT = parent
    CompiledStanModelPath.CHILD = child


def compile_stan_model(stan_model_name):
    """
    Compile stan model and save as pkl
    """
    source_model = pkg_resources.resource_filename(
        "orbit", "stan/{}.stan".format(stan_model_name)
    )
    if CompiledStanModelPath.PARENT == "orbit":
        compiled_model = pkg_resources.resource_filename(
            "orbit", "{}/{}.pkl".format(CompiledStanModelPath.CHILD, stan_model_name)
        )
    else:
        compiled_model = os.path.join(
            CompiledStanModelPath.PARENT,
            "{}/{}.pkl".format(CompiledStanModelPath.CHILD, stan_model_name),
        )

    # updated for py3
    os.makedirs(os.path.dirname(compiled_model), exist_ok=True)

    # compile if stan source has changed
    if not os.path.isfile(compiled_model) or os.path.getmtime(
        compiled_model
    ) < os.path.getmtime(source_model):

        with open(source_model, encoding="utf-8") as f:
            model_code = f.read()

        sm = StanModel(model_code=model_code)

        with open(compiled_model, "wb") as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)

    return compiled_model


def get_compiled_stan_model(stan_model_name):
    """
    Load compiled Stan model
    """

    compiled_model = compile_stan_model(stan_model_name)

    with open(compiled_model, "rb") as f:
        return pickle.load(f)


class suppress_stdout_stderr(object):
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
