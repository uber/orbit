"""
Compatibility utilities for cmdstanpy integration.

This module contains patches and workarounds for cmdstanpy compatibility issues.
"""

import os
from typing import Dict, List, Optional, Callable

from .logger import get_logger

logger = get_logger("orbit")


def patch_tqdm_progress_hook():
    """
    Patch cmdstanpy progress hook to handle TQDM_DISABLE safely.

    When TQDM_DISABLE=1 is set, tqdm creates disabled progress bar objects
    that don't have the 'postfix' attribute. cmdstanpy assumes this attribute
    exists and tries to access it, causing AttributeError.

    This patch adds safe access checks to prevent the crash.

    See: https://github.com/uber/orbit/issues/887
    """
    # Only patch if TQDM_DISABLE is set
    if os.environ.get("TQDM_DISABLE") != "1":
        return

    try:
        import cmdstanpy.model
        import re

        # Store reference to original method to avoid patching multiple times
        if hasattr(cmdstanpy.model.CmdStanModel, "_orbit_tqdm_patched"):
            return

        original_hook = getattr(
            cmdstanpy.model.CmdStanModel, "_wrap_sampler_progress_hook", None
        )
        if original_hook is None:
            return

        @staticmethod
        def safe_wrap_sampler_progress_hook(
            chain_ids: List[int],
            total: int,
        ) -> Optional[Callable[[str, int], None]]:
            """Safe version that handles disabled tqdm progress bars."""
            try:
                from tqdm import tqdm

                pat = re.compile(r"Chain \[(\d*)\] (Iteration.*)")
                pbars: Dict[int, tqdm] = {
                    chain_id: tqdm(
                        total=total,
                        bar_format="{desc} |{bar}| {elapsed} {postfix[0][value]}",
                        postfix=[{"value": "Status"}],
                        desc=f"chain {chain_id}",
                        colour="yellow",
                    )
                    for chain_id in chain_ids
                }

                def progress_hook(line: str, idx: int) -> None:
                    if line == "Done":
                        for pbar in pbars.values():
                            # safe postfix access
                            if hasattr(pbar, "postfix") and pbar.postfix:
                                try:
                                    pbar.postfix[0]["value"] = "Sampling completed"
                                except (AttributeError, KeyError, IndexError):
                                    pass
                            pbar.update(total - pbar.n)
                            pbar.close()
                    else:
                        match = pat.match(line)
                        if match:
                            idx = int(match.group(1))
                            mline = match.group(2).strip()
                        elif line.startswith("Iteration"):
                            mline = line
                            idx = chain_ids[idx]
                        else:
                            return

                        if idx in pbars:
                            if "Sampling" in mline and hasattr(pbars[idx], "colour"):
                                pbars[idx].colour = "blue"
                            pbars[idx].update(1)

                            # safe postfix access
                            if hasattr(pbars[idx], "postfix") and pbars[idx].postfix:
                                try:
                                    pbars[idx].postfix[0]["value"] = mline
                                except (AttributeError, KeyError, IndexError):
                                    pass

                return progress_hook

            except Exception as e:
                logger.warning(
                    f"Progress bar setup failed: {e}. Disabling progress bars."
                )
                return None

        # apply the patch
        cmdstanpy.model.CmdStanModel._wrap_sampler_progress_hook = (
            safe_wrap_sampler_progress_hook
        )
        cmdstanpy.model.CmdStanModel._orbit_tqdm_patched = True
        logger.debug("cmdstanpy progress hook patched for TQDM_DISABLE compatibility")

    except Exception as e:
        logger.warning(f"Failed to patch cmdstanpy progress hook: {e}")
