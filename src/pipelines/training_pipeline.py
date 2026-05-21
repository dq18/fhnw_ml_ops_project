"""
Training Pipeline — containerized version.

Steps:
  A. Connect to Hopsworks and get Feature Groups
  B. Create a Feature View joining weather_daily + crag_static + climb_logs
  C. Create Hopsworks-managed train/test split
  D. Build model pipeline from model_config.json
  E. Train & evaluate
  F. Save model locally + register in Hopsworks Model Registry

Run:
    python -m src.pipelines.training_pipeline
"""

import json
import sys
import tempfile
from pathlib import Path

import hopsworks
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# hsfs Kafka engine uses a hardcoded /tmp path for SSL certs.
Path("/tmp").mkdir(exist_ok=True)

from src.config import (
    CRAG_FG_NAME, CRAG_FG_VERSION,
    WEATHER_FG_NAME, WEATHER_FG_VERSION,
    CLIMB_LOGS_FG_NAME, CLIMB_LOGS_FG_VERSION,
    FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION,
    MODEL_NAME, MODEL_LOCAL_PATH, MODELS_DIR,
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
)

# ── Load model configuration ─────────────────────────────────────────────────
MODEL_CONFIG_PATH = PROJECT_ROOT / "model_config.json"


def _load_model_config() -> dict:
    """Load model_config.json; return defaults if missing."""
    if MODEL_CONFIG_PATH.exists():
        with open(MODEL_CONFIG_PATH) as f:
            return json.load(f)
    return {
        "classifier": {
            "type": "RandomForestClassifier",
            "params": {"n_estimators": 100, "class_weight": "balanced",
                       "random_state": 42},
        },
        "preprocessing": {
            "cat_features": ["rocks", "rain_exposure", "sun_exposure"],
            "drop_cols": ["date", "crag_id"],
        },
    }


def _build_classifier(config: dict):
    """Instantiate a sklearn classifier from config."""
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    classifiers = {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "LogisticRegression": LogisticRegression,
    }
    cls_name = config["classifier"]["type"]
    cls_params = config["classifier"].get("params", {})

    if cls_name not in classifiers:
        raise ValueError(
            f"Unknown classifier '{cls_name}'. "
            f"Supported: {list(classifiers.keys())}"
        )
    # Drop any params the chosen classifier doesn't accept (e.g. class_weight
    # is valid for RandomForest/ExtraTrees but not GradientBoosting).
    import inspect
    valid_params = inspect.signature(classifiers[cls_name].__init__).parameters
    filtered = {k: v for k, v in cls_params.items() if k in valid_params}
    dropped = set(cls_params) - set(filtered)
    if dropped:
        print(f"  Warning: ignoring unsupported params for {cls_name}: {dropped}")
    return classifiers[cls_name](**filtered)


# ── Stage helpers (encoded in model description) ─────────────────────────────
STAGE_PREFIX = {"staging": "[staging]", "production": "[production]",
                "archived": "[archived]"}


def _get_stage(model) -> str:
    """Extract stage from model description prefix."""
    desc = model.description or ""
    for stage, prefix in STAGE_PREFIX.items():
        if desc.startswith(prefix):
            return stage
    return "untagged"


def _make_description(stage: str, config: dict) -> str:
    """Build description with stage prefix + classifier info."""
    prefix = STAGE_PREFIX.get(stage, "")
    cls_name = config["classifier"]["type"]
    return f"{prefix} {cls_name} classifier for crag climbability prediction"


def run_training_pipeline() -> None:
    """Execute the full training pipeline."""
    MODELS_DIR.mkdir(exist_ok=True)
    config = _load_model_config()
    print(f"Model config: {config['classifier']['type']} "
          f"with {config['classifier'].get('params', {})}")

    # ── A: Connect to Hopsworks ──────────────────────────────────────────
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=tempfile.gettempdir(),
    )
    fs = project.get_feature_store()

    crag_fg = fs.get_feature_group(CRAG_FG_NAME, CRAG_FG_VERSION)
    weather_fg = fs.get_feature_group(WEATHER_FG_NAME, WEATHER_FG_VERSION)
    logs_fg = fs.get_feature_group(CLIMB_LOGS_FG_NAME, CLIMB_LOGS_FG_VERSION)

    print(f"Feature Groups loaded: {crag_fg.name}, "
          f"{weather_fg.name}, {logs_fg.name}")

    # ── B: Create Feature View (3-way join) ──────────────────────────────
    query = (
        weather_fg.select_all()
        .join(crag_fg.select_except(["name", "crag_id"]), on="crag_id")
        .join(logs_fg.select(["climbable"]), on=["crag_id", "date"])
    )

    fv = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        query=query,
        labels=["climbable"],
    )
    print(f"Feature View: {fv.name} v{fv.version}")

    # ── C: Train/test split (Hopsworks-managed) ──────────────────────────
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)

    # Drop rows where the label is NaN (left-join artefact)
    train_mask = y_train["climbable"].notna()
    X_train = X_train[train_mask].reset_index(drop=True)
    y_train = y_train[train_mask].reset_index(drop=True)

    test_mask = y_test["climbable"].notna()
    X_test = X_test[test_mask].reset_index(drop=True)
    y_test = y_test[test_mask].reset_index(drop=True)

    y_train["climbable"] = y_train["climbable"].astype(int)
    y_test["climbable"] = y_test["climbable"].astype(int)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Train label distribution:\n"
          f"{y_train['climbable'].value_counts()}")

    # ── D: Build model pipeline from config ──────────────────────────────
    prep_config = config.get("preprocessing", {})
    cat_features = prep_config.get(
        "cat_features", ["rocks", "rain_exposure", "sun_exposure"]
    )
    drop_cols = [
        c for c in prep_config.get("drop_cols", ["date", "crag_id"])
        if c in X_train.columns
    ]

    X_train_clean = X_train.drop(columns=drop_cols, errors="ignore")
    X_test_clean = X_test.drop(columns=drop_cols, errors="ignore")
    num_features = [
        c for c in X_train_clean.columns if c not in cat_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )

    classifier = _build_classifier(config)
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    print(f"Pipeline: {type(classifier).__name__}")

    # ── E: Train & evaluate ──────────────────────────────────────────────
    model_pipeline.fit(X_train_clean, y_train["climbable"])
    y_pred = model_pipeline.predict(X_test_clean)

    acc = accuracy_score(y_test["climbable"], y_pred)
    f1 = f1_score(y_test["climbable"], y_pred, average="macro")

    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(classification_report(
        y_test["climbable"], y_pred, zero_division=0
    ))

    # ── F: Save model locally ────────────────────────────────────────────
    joblib.dump(model_pipeline, MODEL_LOCAL_PATH)
    print(f"Model saved to {MODEL_LOCAL_PATH}")

    # ── G: Register in Hopsworks Model Registry ──────────────────────────
    mr = project.get_model_registry()

    # Auto-increment: find next available version
    try:
        existing = mr.get_models(MODEL_NAME)
        next_version = max(m.version for m in existing) + 1
    except Exception:
        next_version = 1

    model = mr.sklearn.create_model(
        name=MODEL_NAME,
        version=next_version,
        metrics={"accuracy": acc, "f1_score": f1},
        description=_make_description("staging", config),
    )
    model.save(str(MODELS_DIR))
    print(f"Model registered: {model.name} v{model.version} (staging)")

    print("\n✓ Training pipeline complete.")


if __name__ == "__main__":
    run_training_pipeline()
