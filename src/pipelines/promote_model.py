"""
Promote a model version from staging (challenger) to production (champion).

Usage:
    python -m src.pipelines.promote_model              # promote latest staging
    python -m src.pipelines.promote_model --version 3  # promote specific version

This demotes the current champion (if any) back to "archived" and sets the
specified version as the new production model.

Stage is encoded in the model description field as a prefix:
  [staging], [production], or [archived]
"""

import sys
import argparse
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
Path("/tmp").mkdir(exist_ok=True)

import hopsworks
from src.config import MODEL_NAME, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

STAGE_PREFIX = {"staging": "[staging]", "production": "[production]",
                "archived": "[archived]"}


def _get_stage(model) -> str:
    desc = model.description or ""
    for stage, prefix in STAGE_PREFIX.items():
        if desc.startswith(prefix):
            return stage
    return "untagged"


def _set_stage(model, stage: str) -> None:
    """Update model description to reflect new stage."""
    desc = model.description or ""
    # Strip existing prefix
    for prefix in STAGE_PREFIX.values():
        if desc.startswith(prefix):
            desc = desc[len(prefix):].lstrip()
            break
    new_prefix = STAGE_PREFIX.get(stage, "")
    model.description = f"{new_prefix} {desc}".strip()
    model.save(str(Path(tempfile.gettempdir()) / "model_promote"))


def promote(version: int | None = None) -> None:
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=tempfile.gettempdir(),
    )
    mr = project.get_model_registry()
    models = mr.get_models(MODEL_NAME)

    if not models:
        print("No models found in the registry.")
        return

    # Find the version to promote
    if version is None:
        staging = [m for m in models if _get_stage(m) == "staging"]
        if not staging:
            print("No model with [staging] prefix found. Nothing to promote.")
            return
        candidate = max(staging, key=lambda m: m.version)
    else:
        candidate = mr.get_model(MODEL_NAME, version=version)

    metrics = candidate.training_metrics or {}
    print(f"Promoting: {candidate.name} v{candidate.version}")
    print(f"  Metrics: accuracy={metrics.get('accuracy', '?')}, "
          f"f1={metrics.get('f1_score', '?')}")

    # Demote current champion (if any)
    for m in models:
        if _get_stage(m) == "production" and m.version != candidate.version:
            _set_stage(m, "archived")
            print(f"  Demoted v{m.version} → archived")

    # Promote candidate
    _set_stage(candidate, "production")
    print(f"  ✓ v{candidate.version} is now the production champion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Promote a model to production"
    )
    parser.add_argument(
        "--version", type=int, default=None,
        help="Specific model version to promote (default: latest staging)",
    )
    args = parser.parse_args()
    promote(args.version)
