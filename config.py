"""
Central configuration for AI Product Lab.

All settings are loaded from environment variables (via .env).
Nothing is hardcoded here — change behavior by editing .env.
"""

from pathlib import Path

from dotenv import load_dotenv

from src.utils import get_env, get_env_float, get_env_int

# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env from the project root so the app works from any cwd
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Runtime configuration sourced entirely from the environment."""

    ollama_base_url: str = get_env("OLLAMA_BASE_URL")
    model_name: str = get_env("MODEL_NAME")
    temperature: float = get_env_float("TEMPERATURE")
    top_p: float = get_env_float("TOP_P")
    max_tokens: int = get_env_int("MAX_TOKENS")

    prompt_path: Path = PROJECT_ROOT / get_env("PROMPT_PATH")
    dataset_path: Path = PROJECT_ROOT / get_env("DATASET_PATH")
    taxonomy_path: Path = PROJECT_ROOT / get_env("TAXONOMY_PATH")
    output_dir: Path = PROJECT_ROOT / get_env("OUTPUT_DIR")

    @property
    def predictions_dir(self) -> Path:
        """Directory where per-run prediction CSVs are written."""
        return self.output_dir / "predictions"


config = Config()
