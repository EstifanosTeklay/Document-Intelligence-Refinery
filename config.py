import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Project root is 2 levels up from this file (src/utils/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RULES_PATH = PROJECT_ROOT / "rubric" / "extraction_rules.yaml"
REFINERY_DIR = PROJECT_ROOT / ".refinery"


@lru_cache(maxsize=1)
def load_rules() -> dict:
    """
    Load extraction_rules.yaml once and cache it.
    All agents and strategies read thresholds from here — never hardcode.
    """
    if not RULES_PATH.exists():
        raise FileNotFoundError(f"extraction_rules.yaml not found at {RULES_PATH}")
    with open(RULES_PATH, "r") as f:
        return yaml.safe_load(f)


class Config:
    """
    Single access point for all configuration.
    Reads from extraction_rules.yaml and environment variables.
    """

    def __init__(self):
        self._rules = load_rules()

    # ----------------------------------------------------------
    # Origin detection thresholds
    # ----------------------------------------------------------
    @property
    def min_chars_per_page(self) -> int:
        return int(os.getenv(
            "MIN_CHARS_PER_PAGE",
            self._rules["origin_detection"]["min_chars_per_page"]
        ))

    @property
    def max_image_area_ratio(self) -> float:
        return float(os.getenv(
            "MAX_IMAGE_AREA_RATIO",
            self._rules["origin_detection"]["max_image_area_ratio"]
        ))

    @property
    def scanned_page_threshold(self) -> float:
        return self._rules["origin_detection"]["scanned_page_threshold"]

    # ----------------------------------------------------------
    # Layout detection thresholds
    # ----------------------------------------------------------
    @property
    def multi_column_threshold(self) -> float:
        return self._rules["layout_detection"]["multi_column_threshold"]

    @property
    def table_heavy_page_ratio(self) -> float:
        return self._rules["layout_detection"]["table_heavy_page_ratio"]

    @property
    def figure_heavy_page_ratio(self) -> float:
        return self._rules["layout_detection"]["figure_heavy_page_ratio"]

    # ----------------------------------------------------------
    # Domain hint keywords
    # ----------------------------------------------------------
    @property
    def domain_keywords(self) -> dict[str, list[str]]:
        return self._rules["domain_hints"]

    # ----------------------------------------------------------
    # Strategy routing
    # ----------------------------------------------------------
    @property
    def strategy_a_min_confidence(self) -> float:
        return self._rules["strategy_routing"]["strategy_a"]["min_confidence_to_pass"]

    @property
    def strategy_b_min_confidence(self) -> float:
        return self._rules["strategy_routing"]["strategy_b"]["min_confidence_to_pass"]

    @property
    def strategy_c_max_cost_usd(self) -> float:
        return float(os.getenv(
            "MAX_COST_PER_DOCUMENT_USD",
            self._rules["strategy_routing"]["strategy_c"]["max_cost_usd"]
        ))

    @property
    def strategy_c_max_pages(self) -> int:
        return int(os.getenv(
            "MAX_VISION_PAGES_PER_DOC",
            self._rules["strategy_routing"]["strategy_c"]["max_pages"]
        ))

    # ----------------------------------------------------------
    # API keys
    # ----------------------------------------------------------
    @property
    def openrouter_api_key(self) -> str:
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        return key

    @property
    def vision_model(self) -> str:
        return os.getenv("VISION_MODEL", "google/gemini-flash-1.5")

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------
    @property
    def refinery_dir(self) -> Path:
        path = Path(os.getenv("REFINERY_DIR", str(REFINERY_DIR)))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def profiles_dir(self) -> Path:
        path = self.refinery_dir / "profiles"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def ledger_path(self) -> Path:
        return self.refinery_dir / "extraction_ledger.jsonl"

    # ----------------------------------------------------------
    # Raw rules access (for anything not explicitly exposed above)
    # ----------------------------------------------------------
    def get_raw(self, *keys: str):
        """Traverse nested keys in rules. E.g. get_raw('chunking', 'max_tokens_per_chunk')"""
        node = self._rules
        for key in keys:
            node = node[key]
        return node


# Single shared instance — import this everywhere
config = Config()
