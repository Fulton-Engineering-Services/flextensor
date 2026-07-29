# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadManager and TensorManager save_profile/load_profile functionality.

This test suite validates profile saving and loading operations:

1. save_profile: Saves tensor manager state to a directory as profile.json
2. load_profile: Loads tensor manager state from a directory and restores it to a model

Tests use pytest's tmp_path fixture to create temporary directories that are
automatically cleaned up after each test.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.offload_manager import OffloadConfig, OffloadManager


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


class MockTensorManagerState:
    """Mock TensorManagerState for testing."""

    def __init__(self):
        self.load_strategy = {}
        self.stats = []
        self.release_strategy = {}

    def to_dict(self):
        return {
            "loader_type": "strategy",
            "tensor_id_to_name_map": {},
            "allocation_ordered": {},
            "label_to_size_map": {},
            "block_sizes": {},
            "load_strategy": {},
            "release_strategy": {},
            "label_to_block_id": {},
            "stats": [],
            "transfer_to_compute_map": {},
            "view_tensors_ids": [],
            "view_tensors_names": [],
            "gpu_tensors_names": [],
            "shm_block_name_map": None,
        }


class TestOffloadManagerSaveProfile:
    """Test cases for OffloadManager.save_profile()."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_calls_tensor_manager_with_directory(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that save_profile passes the directory path to tensor_manager.save_profile()."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["linear1"], skip_discovery=False)
        om.offload(self.model, config=config)

        profile_dir = tmp_path / "new_profile_dir"

        om.save_profile(str(profile_dir))

        # Verify tensor_manager.save_profile was called with the directory path
        mock_tensor_manager.save_profile.assert_called_once_with(str(profile_dir))

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_delegates_to_tensor_manager(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that save_profile delegates to tensor_manager.save_profile()."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["linear1"], skip_discovery=False)
        om.offload(self.model, config=config)

        profile_dir = str(tmp_path / "profile")
        om.save_profile(profile_dir)

        mock_tensor_manager.save_profile.assert_called_once_with(profile_dir)

    def test_save_profile_raises_if_not_initialized(self, tmp_path):
        """Test that save_profile raises RuntimeError if tensor_manager is not initialized."""
        om = OffloadManager("test")

        with pytest.raises(RuntimeError, match="Tensor manager not initialized"):
            om.save_profile(str(tmp_path / "profile"))


class TestOffloadManagerLoadProfile:
    """Test cases for OffloadManager.load_profile()."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_profile_delegates_to_tensor_manager(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that load_profile delegates to tensor_manager.load_profile()."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["linear1"], skip_discovery=False)
        om.offload(self.model, config=config)

        profile_dir = str(tmp_path / "profile")
        om.load_profile(profile_dir, self.model)

        mock_tensor_manager.load_profile.assert_called_once_with(profile_dir, self.model)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_profile_uses_current_model_if_not_provided(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that load_profile uses the current model if model is not provided."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["linear1"], skip_discovery=False)
        om.offload(self.model, config=config)

        profile_dir = str(tmp_path / "profile")
        om.load_profile(profile_dir)

        # Should use the internal model
        mock_tensor_manager.load_profile.assert_called_once()
        call_args = mock_tensor_manager.load_profile.call_args
        assert call_args[0][0] == profile_dir
        # The model passed should be the internal model
        assert call_args[0][1] is not None

    def test_load_profile_raises_if_no_model(self, tmp_path):
        """Test that load_profile raises RuntimeError if no model is available."""
        om = OffloadManager("test")
        # Initialize tensor manager without a model
        om._initialize_tensor_manager = MagicMock()

        with pytest.raises(RuntimeError, match="No model provided"):
            om.load_profile(str(tmp_path / "profile"))


class TestTensorManagerSaveProfile:
    """Test cases for TensorManager.save_profile()."""

    def test_save_profile_creates_directory_and_file(self, tmp_path):
        """Test that save_profile creates the directory and profile.json file."""
        from flextensor.state_handler import TensorManagerState
        from flextensor.tensor_manager import TensorManager

        # Create a minimal TensorManager with mocked dependencies
        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)
            tm.tensor_manager_state = MagicMock(spec=TensorManagerState)
            tm.tensor_manager_state.to_dict.return_value = {
                "loader_type": "strategy",
                "tensor_id_to_name_map": {},
                "allocation_ordered": {},
                "label_to_size_map": {},
                "block_sizes": {},
                "load_strategy": {},
                "release_strategy": {},
                "label_to_block_id": {},
                "stats": [],
                "transfer_to_compute_map": {},
                "view_tensors_ids": [],
                "view_tensors_names": [],
                "gpu_tensors_names": [],
                "shm_block_name_map": None,
            }

            profile_dir = tmp_path / "profile"
            assert not profile_dir.exists()

            tm.save_profile(str(profile_dir))

            assert profile_dir.exists()
            profile_file = profile_dir / "profile.json"
            assert profile_file.exists()

            # Verify JSON content is valid
            with profile_file.open() as f:
                data = json.load(f)
            assert "loader_type" in data

    def test_save_profile_overwrites_existing(self, tmp_path):
        """Test that save_profile overwrites existing profile.json."""
        from flextensor.state_handler import TensorManagerState
        from flextensor.tensor_manager import TensorManager

        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        profile_file = profile_dir / "profile.json"

        # Write initial content
        profile_file.write_text('{"old": "data"}')

        # Create TensorManager and save new profile
        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)
            tm.tensor_manager_state = MagicMock(spec=TensorManagerState)
            tm.tensor_manager_state.to_dict.return_value = {
                "loader_type": "new_strategy",
                "tensor_id_to_name_map": {},
                "allocation_ordered": {},
                "label_to_size_map": {},
                "block_sizes": {},
                "load_strategy": {},
                "release_strategy": {},
                "label_to_block_id": {},
                "stats": [],
                "transfer_to_compute_map": {},
                "view_tensors_ids": [],
                "view_tensors_names": [],
                "gpu_tensors_names": [],
                "shm_block_name_map": None,
            }

            tm.save_profile(str(profile_dir))

            # Verify new content
            with profile_file.open() as f:
                data = json.load(f)
            assert data["loader_type"] == "new_strategy"
            assert "old" not in data


class TestTensorManagerLoadProfile:
    """Test cases for TensorManager.load_profile()."""

    def test_load_profile_reads_from_directory(self, tmp_path):
        """Test that load_profile reads profile.json from the specified directory."""
        from flextensor.tensor_manager import TensorManager

        # Create a valid profile
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        profile_file = profile_dir / "profile.json"

        from flextensor.state_handler import TensorManagerState

        profile_data = {
            "version": TensorManagerState.SCHEMA_VERSION,
            "loader_type": "strategy",
            "tensor_id_to_name_map": {"123": "layer.weight"},
            "allocation_ordered": {},
            "label_to_size_map": {},
            "block_sizes": {},
            "load_strategy": {},
            "release_strategy": {},
            "label_to_block_id": {},
            "stats": [],
            "transfer_to_compute_map": {},
            "view_tensors_ids": [],
            "view_tensors_names": ["layer.weight"],
            "gpu_tensors_names": [],
            "shm_block_name_map": None,
        }
        profile_file.write_text(json.dumps(profile_data))

        # Create TensorManager with mocked restore_state
        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)
            tm.restore_state = MagicMock()

            model = SimpleModel()
            tm.load_profile(str(profile_dir), model)

            # Verify restore_state was called with the loaded state
            tm.restore_state.assert_called_once()
            call_args = tm.restore_state.call_args
            assert call_args[0][0] is model
            loaded_state = call_args[0][1]
            assert loaded_state.loader_type == "strategy"

    def test_load_profile_raises_if_file_not_found(self, tmp_path):
        """Test that load_profile raises error if profile.json doesn't exist."""
        from flextensor.tensor_manager import TensorManager

        profile_dir = tmp_path / "nonexistent"

        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)

            model = SimpleModel()
            with pytest.raises(FileNotFoundError):
                tm.load_profile(str(profile_dir), model)


class TestProfileRoundTrip:
    """Test cases for save/load profile round-trip functionality."""

    def test_profile_json_structure(self, tmp_path):
        """Test that saved profile has correct JSON structure."""
        from flextensor.state_handler import TensorManagerState
        from flextensor.tensor_manager import TensorManager

        profile_dir = tmp_path / "profile"

        with patch("flextensor.tensor_manager.TensorManager.__init__", lambda self: None):
            tm = TensorManager.__new__(TensorManager)
            tm.tensor_manager_state = MagicMock(spec=TensorManagerState)
            tm.tensor_manager_state.to_dict.return_value = {
                "loader_type": "allocation_block_transfer",
                "tensor_id_to_name_map": {"1": "linear1.weight", "2": "linear1.bias"},
                "allocation_ordered": {"0": ["layer1", "layer2"]},
                "label_to_size_map": {"layer1": 1024, "layer2": 512},
                "block_sizes": {"0": 2048},
                "load_strategy": {"layer1": []},
                "release_strategy": {"layer1": []},
                "label_to_block_id": {"layer1": 0},
                "stats": [{"label": "layer1", "tensors": [], "duration": 1.5}],
                "transfer_to_compute_map": {"layer1": "layer1"},
                "view_tensors_ids": [1, 2],
                "view_tensors_names": ["linear1.weight", "linear1.bias"],
                "gpu_tensors_names": ["linear2.weight"],
                "shm_block_name_map": {"block_0": "shm_block_0"},
            }

            tm.save_profile(str(profile_dir))

            profile_file = profile_dir / "profile.json"
            with profile_file.open() as f:
                data = json.load(f)

            # Verify all required fields are present
            assert data["loader_type"] == "allocation_block_transfer"
            assert "1" in data["tensor_id_to_name_map"]
            assert data["allocation_ordered"]["0"] == ["layer1", "layer2"]
            assert data["view_tensors_names"] == ["linear1.weight", "linear1.bias"]
            assert data["shm_block_name_map"]["block_0"] == "shm_block_0"


class TestProfileDirectoryFromConfig:
    """Test cases for profile_directory falling back to config.profile_storage_dir."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

    def test_resolve_profile_directory_uses_argument(self):
        """Test that explicit argument takes precedence over config."""
        om = OffloadManager("test")
        om.set_config(OffloadConfig(profile_storage_dir="/config/path"))

        resolved = om._resolve_profile_directory("/explicit/path")
        assert resolved == "/explicit/path"

    def test_resolve_profile_directory_falls_back_to_config(self):
        """Test that config.profile_storage_dir is used when argument is None."""
        om = OffloadManager("test")
        om.set_config(OffloadConfig(profile_storage_dir="/config/path"))

        resolved = om._resolve_profile_directory(None)
        assert resolved == "/config/path"

    def test_resolve_profile_directory_raises_when_neither_set(self):
        """Test that ValueError is raised when no directory is available."""
        om = OffloadManager("test")
        om.set_config(OffloadConfig(profile_storage_dir=None))

        with pytest.raises(ValueError, match="No profile directory specified"):
            om._resolve_profile_directory(None)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_uses_config_directory(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that save_profile() without arguments uses config.profile_storage_dir."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        profile_dir = str(tmp_path / "config_profile")
        config = OffloadConfig(
            enabled=True, include_patterns=["linear1"], profile_storage_dir=profile_dir, skip_discovery=False
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        om.save_profile()

        mock_tensor_manager.save_profile.assert_called_once_with(profile_dir)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_argument_overrides_config(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that explicit argument overrides config.profile_storage_dir."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        config = OffloadConfig(
            enabled=True, include_patterns=["linear1"], profile_storage_dir="/config/path", skip_discovery=False
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        explicit_dir = str(tmp_path / "explicit_profile")
        om.save_profile(explicit_dir)

        mock_tensor_manager.save_profile.assert_called_once_with(explicit_dir)

    def test_save_profile_raises_when_no_directory(self):
        """Test that save_profile raises ValueError when no directory is available."""
        om = OffloadManager("test")
        om._tensor_manager = MagicMock()

        with pytest.raises(ValueError, match="No profile directory specified"):
            om.save_profile()

    def test_save_profile_raises_when_read_only(self):
        """Test that save_profile raises ValueError when profile_read_only is True."""
        om = OffloadManager("test")
        om.set_config(OffloadConfig(profile_read_only=True, profile_storage_dir="/some/path"))
        om._tensor_manager = MagicMock()

        with pytest.raises(ValueError, match="profile_read_only is True"):
            om.save_profile()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_profile_succeeds_when_not_read_only(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test that save_profile works when profile_read_only is False."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        config = OffloadConfig(
            enabled=True,
            include_patterns=["linear1"],
            profile_storage_dir="/some/path",
            skip_discovery=False,
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        om.save_profile()

        mock_tensor_manager.save_profile.assert_called_once_with("/some/path")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_profile_uses_config_directory(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that load_profile() without directory uses config.profile_storage_dir."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        profile_dir = str(tmp_path / "config_profile")
        config = OffloadConfig(
            enabled=True, include_patterns=["linear1"], profile_storage_dir=profile_dir, skip_discovery=False
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        om.load_profile(model=self.model)

        mock_tensor_manager.load_profile.assert_called_once_with(profile_dir, self.model)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_profile_argument_overrides_config(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        tmp_path,
    ):
        """Test that explicit argument overrides config.profile_storage_dir for load."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.trap = MagicMock()

        config = OffloadConfig(
            enabled=True, include_patterns=["linear1"], profile_storage_dir="/config/path", skip_discovery=False
        )
        om = OffloadManager("test")
        om.offload(self.model, config=config)

        explicit_dir = str(tmp_path / "explicit_profile")
        om.load_profile(explicit_dir, self.model)

        mock_tensor_manager.load_profile.assert_called_once_with(explicit_dir, self.model)

    def test_load_profile_raises_when_no_directory(self):
        """Test that load_profile raises ValueError when no directory is available."""
        om = OffloadManager("test")
        om._tensor_manager = MagicMock()
        om._model = self.model

        with pytest.raises(ValueError, match="No profile directory specified"):
            om.load_profile()


class TestModuleLevelProfileFunctions:
    """Test cases for module-level save_profile and load_profile functions."""

    @patch("flextensor.offload_manager.get_offload_manager")
    def test_module_save_profile(self, mock_get_om, tmp_path):
        """Test that module-level save_profile delegates to OffloadManager."""
        from flextensor.offload_manager import save_profile

        mock_om = MagicMock()
        mock_get_om.return_value = mock_om

        profile_dir = str(tmp_path / "profile")
        save_profile(profile_dir)

        mock_om.save_profile.assert_called_once_with(profile_dir)

    @patch("flextensor.offload_manager.get_offload_manager")
    def test_module_load_profile(self, mock_get_om, tmp_path):
        """Test that module-level load_profile delegates to OffloadManager."""
        from flextensor.offload_manager import load_profile

        mock_om = MagicMock()
        mock_get_om.return_value = mock_om

        model = SimpleModel()
        profile_dir = str(tmp_path / "profile")
        load_profile(profile_dir, model)

        mock_om.load_profile.assert_called_once_with(profile_dir, model)

    @patch("flextensor.offload_manager.get_offload_manager")
    def test_module_load_profile_without_model(self, mock_get_om, tmp_path):
        """Test that module-level load_profile works without model argument."""
        from flextensor.offload_manager import load_profile

        mock_om = MagicMock()
        mock_get_om.return_value = mock_om

        profile_dir = str(tmp_path / "profile")
        load_profile(profile_dir)

        mock_om.load_profile.assert_called_once_with(profile_dir, None)


class TestOffloadFromProfile:
    """Test cases for the offload_from_profile convenience function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_calls_init_load_profile_offload_in_order(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that offload_from_profile calls init, load_profile, and offload in correct order."""
        from flextensor.offload_manager import offload_from_profile

        mock_offload.return_value = self.model
        config = OffloadConfig(include_patterns=["linear1"])
        profile_dir = str(tmp_path / "profile")

        call_order = []
        mock_init.side_effect = lambda **kw: call_order.append("init")
        mock_load.side_effect = lambda *a, **kw: call_order.append("load_profile")
        mock_offload.side_effect = lambda *a, **kw: (call_order.append("offload"), self.model)[1]

        offload_from_profile(self.model, profile_dir, config=config, name="test")

        assert call_order == ["init", "load_profile", "offload"]

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_passes_correct_arguments(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that offload_from_profile passes the correct arguments to each function."""
        from flextensor.offload_manager import offload_from_profile

        mock_offload.return_value = self.model
        config = OffloadConfig(include_patterns=["linear1"])
        profile_dir = str(tmp_path / "profile")

        offload_from_profile(self.model, profile_dir, config=config, name="mymodel")

        mock_init.assert_called_once_with(config=config, name="mymodel")
        mock_load.assert_called_once_with(profile_dir, model=self.model, name="mymodel")
        mock_offload.assert_called_once_with(self.model, config=config, name="mymodel", compile_fn=None)

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_returns_offload_result(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that offload_from_profile returns the result from offload()."""
        from flextensor.offload_manager import offload_from_profile

        proxy_model = SimpleModel()
        mock_offload.return_value = proxy_model
        config = OffloadConfig(include_patterns=["linear1"])
        profile_dir = str(tmp_path / "profile")

        result = offload_from_profile(self.model, profile_dir, config=config, name="test")

        assert result is proxy_model

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_uses_default_name(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that offload_from_profile uses DEFAULT_MANAGER_NAME when name is not provided."""
        from flextensor.offload_manager import DEFAULT_MANAGER_NAME, offload_from_profile

        mock_offload.return_value = self.model
        profile_dir = str(tmp_path / "profile")

        offload_from_profile(self.model, profile_dir)

        mock_init.assert_called_once_with(config=None, name=DEFAULT_MANAGER_NAME)
        mock_load.assert_called_once_with(profile_dir, model=self.model, name=DEFAULT_MANAGER_NAME)
        mock_offload.assert_called_once_with(self.model, config=None, name=DEFAULT_MANAGER_NAME, compile_fn=None)

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_forwards_compile_fn(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that compile_fn is forwarded to offload()."""
        from flextensor.offload_manager import offload_from_profile

        mock_offload.return_value = self.model
        config = OffloadConfig(include_patterns=["linear1"])
        profile_dir = str(tmp_path / "profile")

        def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
            return module

        offload_from_profile(self.model, profile_dir, config=config, name="test", compile_fn=compile_fn)

        mock_offload.assert_called_once_with(self.model, config=config, name="test", compile_fn=compile_fn)

    @patch("flextensor.offload_manager.offload")
    @patch("flextensor.offload_manager.load_profile")
    @patch("flextensor.offload_manager.init")
    def test_propagates_load_profile_error(self, mock_init, mock_load, mock_offload, tmp_path):
        """Test that errors from load_profile propagate without calling offload."""
        from flextensor.offload_manager import offload_from_profile

        mock_load.side_effect = FileNotFoundError("profile.json not found")
        profile_dir = str(tmp_path / "nonexistent")

        with pytest.raises(FileNotFoundError, match=r"profile\.json not found"):
            offload_from_profile(self.model, profile_dir)

        mock_init.assert_called_once()
        mock_offload.assert_not_called()
