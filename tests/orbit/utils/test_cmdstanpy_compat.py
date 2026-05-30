import os
import pytest
from unittest.mock import patch, MagicMock

from orbit.utils.cmdstanpy_compat import patch_tqdm_progress_hook


@pytest.mark.parametrize(
    "env_value",
    [None, "0", "false", "true", ""],
)
def test_patch_tqdm_progress_hook_no_patch_scenarios(env_value):
    """Test that patch is not applied when TQDM_DISABLE is not '1'."""
    env_dict = {"TQDM_DISABLE": env_value} if env_value is not None else {}

    with patch.dict(os.environ, env_dict, clear=False):
        if env_value is None and "TQDM_DISABLE" in os.environ:
            del os.environ["TQDM_DISABLE"]

        # Should return early without doing anything
        patch_tqdm_progress_hook()
        # Test passes if no exception raised


def test_patch_tqdm_progress_hook_applies_patch():
    """Test that patch is applied when TQDM_DISABLE=1."""
    with patch.dict(os.environ, {"TQDM_DISABLE": "1"}):
        with patch("cmdstanpy.model") as mock_cmdstanpy_model:
            mock_model = MagicMock()
            mock_model._wrap_sampler_progress_hook = MagicMock()
            # Ensure not already patched
            del mock_model._orbit_tqdm_patched
            mock_cmdstanpy_model.CmdStanModel = mock_model

            patch_tqdm_progress_hook()

            assert mock_model._orbit_tqdm_patched is True


def test_patch_tqdm_progress_hook_no_double_patch():
    """Test that patch is not applied multiple times."""
    with patch.dict(os.environ, {"TQDM_DISABLE": "1"}):
        with patch("cmdstanpy.model") as mock_cmdstanpy_model:
            mock_model = MagicMock()
            mock_model._orbit_tqdm_patched = True  # Already patched
            original_hook = MagicMock()
            mock_model._wrap_sampler_progress_hook = original_hook
            mock_cmdstanpy_model.CmdStanModel = mock_model

            patch_tqdm_progress_hook()

            # Original hook should remain unchanged
            assert mock_model._wrap_sampler_progress_hook is original_hook


def test_patch_tqdm_progress_hook_handles_missing_method():
    """Test graceful handling when original method doesn't exist."""
    with patch.dict(os.environ, {"TQDM_DISABLE": "1"}):
        with patch("cmdstanpy.model") as mock_cmdstanpy_model:
            # Simple object without the method
            mock_model = type("MockModel", (), {})()
            mock_cmdstanpy_model.CmdStanModel = mock_model

            # Should not raise exception
            patch_tqdm_progress_hook()

            # Should not set patched flag
            assert not hasattr(mock_model, "_orbit_tqdm_patched")


def test_patch_tqdm_progress_hook_handles_import_error():
    """Test graceful handling of import errors."""
    with patch.dict(os.environ, {"TQDM_DISABLE": "1"}):
        with patch("cmdstanpy.model", side_effect=ImportError("No module")):
            # Should not raise exception
            patch_tqdm_progress_hook()


def test_integration_with_actual_cmdstanpy():
    """Integration test with actual cmdstanpy if available."""
    cmdstanpy = pytest.importorskip("cmdstanpy")

    with patch.dict(os.environ, {"TQDM_DISABLE": "1"}):
        # Store original state
        original_method = getattr(
            cmdstanpy.model.CmdStanModel, "_wrap_sampler_progress_hook", None
        )

        try:
            patch_tqdm_progress_hook()

            # Verify patch was applied
            assert hasattr(cmdstanpy.model.CmdStanModel, "_orbit_tqdm_patched")
            assert cmdstanpy.model.CmdStanModel._orbit_tqdm_patched is True

        finally:
            # Clean up
            if hasattr(cmdstanpy.model.CmdStanModel, "_orbit_tqdm_patched"):
                delattr(cmdstanpy.model.CmdStanModel, "_orbit_tqdm_patched")
            if original_method is not None:
                cmdstanpy.model.CmdStanModel._wrap_sampler_progress_hook = (
                    original_method
                )
