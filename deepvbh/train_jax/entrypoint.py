"""Runtime-safe training entrypoint for DeepVBH."""

from __future__ import annotations

from .config import PROJECT_ROOT, TrainingConfigLoader
from .types import RuntimeConfig


def load_config():
  """Loads the config and applies environment settings before JAX import."""

  config_loader = TrainingConfigLoader(PROJECT_ROOT)
  config = config_loader.load(config_loader.config_path())
  config.runtime.apply_environment()
  RuntimeConfig.sanitize_sys_path(PROJECT_ROOT)
  return config


def main() -> None:
  """Runs the DeepVBH training application."""

  config = load_config()
  from .app import TrainingApplication

  TrainingApplication(config).run()
