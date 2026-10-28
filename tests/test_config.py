"""Your first test: project_config.yml loads with the course's ProjectConfig for dev.

`ci.yml` runs every test under tests/ on each pull request and each push to main; add yours here.
"""

from pathlib import Path

from arxiv_curator.config import ProjectConfig

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def test_project_config_loads_for_dev() -> None:
    """Load the provisioned settings without requiring optional lecture resources."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")

    assert cfg.full_volume_path == "llmops_dev.pablo.lecture_files"
    assert cfg.warehouse_id == "8d8d38f86ea6cc4a"


def test_project_config_carries_lakebase_and_usage_policy() -> None:
    """Each environment names its Lakebase project and shares the top-level usage policy."""
    for env in ["dev", "acc", "prd"]:
        cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), env)
        assert cfg.lakebase_project_id == "lb-llmops-pablo"
        assert cfg.usage_policy_id == "f91fbdfb-b1c9-39a9-9d93-d132a1e63daa"
