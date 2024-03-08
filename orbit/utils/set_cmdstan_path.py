import json

import cmdstanpy
import importlib_resources
from ..utils.logger import get_logger

logger = get_logger("orbit")


def set_cmdstan_path():
    with open(importlib_resources.files("orbit") / "config.json") as f:
        config = json.load(f)
    CMDSTAN_VERSION = config["CMDSTAN_VERSION"]

    local_cmdstan = (
        importlib_resources.files("orbit")
        / "stan_compiled"
        / f"cmdstan-{CMDSTAN_VERSION}"
    )
    if local_cmdstan.exists():
        cmdstanpy.set_cmdstan_path(str(local_cmdstan))
        logger.debug(
            f"Local/repackaged cmdstan exists, setting path to {str(local_cmdstan)}"
        )
        return 1
    logger.debug(
        f"Cannot find local cmdstan in {str(local_cmdstan)}, using default path at ~/.cmdstan."
    )
    return 1
