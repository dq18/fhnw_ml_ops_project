"""
Streamlit app for interactive crag climbability prediction.

Two tabs:
  1. Single Crag — prediction via the online feature store (get_feature_vector,
     SQL client using sqlalchemy + aiomysql), falling back to batch data.
  2. All Crags Map — batch prediction for all crags with a colour-coded map.

Climbability boolean is always derived from probability ≥ 50% (using
predict_proba only, no separate predict() call).

Run with:
    streamlit run src/app/streamlit_app.py
"""

import sys
import os
import pathlib
import tempfile
from datetime import date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# hsfs Kafka engine uses a hardcoded /tmp path for SSL certs.
pathlib.Path("/tmp").mkdir(exist_ok=True)

import json
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import hopsworks
import joblib

from src.config import (
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MODEL_NAME,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT,
)
from src.weather_client import fetch_forecast_current, fetch_forecast_3day
from src.features.crag_features import prepare_crag_df
from src.features.weather_features import add_rolling_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crag Climbability",
    page_icon=None,
    layout="wide",
)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Hopsworks…")
def get_hopsworks_resources():
    """Connect to Hopsworks once per session and cache FV + model."""
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=tempfile.gettempdir(),
    )
    fs = project.get_feature_store()
    fv = fs.get_feature_view(FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION)
    try:
        fv.init_serving()
    except Exception:
        pass

    # Load the production (champion) model; check overrides file then description prefix.
    mr = project.get_model_registry()
    models = mr.get_models(MODEL_NAME)
    dataset_api = project.get_dataset_api()
    try:
        resp = dataset_api.read_content("/Resources/model_stages.json")
        stage_ov = json.loads(resp.content).get(MODEL_NAME, {}) if resp else {}
    except Exception:
        stage_ov = {}

    def _prod_stage(m):
        return stage_ov.get(str(m.version)) or (
            "production" if (m.description or "").startswith("[production]") else None
        )

    production = [m for m in models if _prod_stage(m) == "production"]
    if production:
        model_hw = max(production, key=lambda m: m.version)
    else:
        model_hw = max(models, key=lambda m: m.version)
    model_dir = model_hw.download()
    model_pipeline = joblib.load(
        os.path.join(model_dir, "crag_classifier.joblib")
    )
    return fv, model_pipeline


@st.cache_data(ttl=3600, show_spinner="Loading features from Hopsworks…")
def get_batch_features() -> pd.DataFrame:
    """Fetch the latest offline batch features (cached 1 h)."""
    fv, _ = get_hopsworks_resources()
    df = fv.get_batch_data()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def get_crag_data() -> pd.DataFrame:
    return prepare_crag_df()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _predict_row(X: pd.DataFrame, model_pipeline) -> tuple[int, float]:
    """Return (0/1 climbable, probability). Boolean derived from prob ≥ 0.5."""
    if hasattr(model_pipeline, "feature_names_in_"):
        X = X.reindex(columns=model_pipeline.feature_names_in_, fill_value=0)
    prob = float(model_pipeline.predict_proba(X)[0][1])
    return int(prob >= 0.5), prob


def _build_map(
    crag_df: pd.DataFrame,
    results: pd.DataFrame | None = None,
) -> pdk.Deck:
    """
    Build a PyDeck scatter map.

    Before prediction (results=None): all pins are blue.
    After prediction: green = climbable, red = not climbable.
    """
    plot_df = crag_df[["crag_id", "name", "latitude", "longitude"]].copy()

    if results is not None and not results.empty:
        plot_df = plot_df.merge(
            results[["crag_id", "prediction", "probability"]],
            on="crag_id",
            how="left",
        )
        plot_df["color"] = plot_df["prediction"].apply(
            lambda p: [50, 200, 80, 220] if p == 1 else [220, 50, 50, 220]
        )
        plot_df["label"] = plot_df.apply(
            lambda r: (
                f"{r['name']}: {'CLIMBABLE' if r['prediction'] == 1 else 'NOT climbable'}"
                f" ({r['probability']:.0%})"
            ),
            axis=1,
        )
        plot_df["radius"] = 2500
    else:
        plot_df["color"] = [[30, 100, 200, 180]] * len(plot_df)
        plot_df["label"] = plot_df["name"]
        plot_df["radius"] = 2000

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(latitude=47.65, longitude=7.50, zoom=8, pitch=0)

    return pdk.Deck(
        layers=[scatter],
        initial_view_state=view_state,
        tooltip={"text": "{label}"},
    )


# ── App header ────────────────────────────────────────────────────────────────
st.title("Crag Climbability Predictor")
st.caption(f"Basel area · {date.today().isoformat()}")

crag_df = get_crag_data()
tab1, tab2, tab3 = st.tabs(["Single Crag", "All Crags Map", "Model Performance"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single crag prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_sel, col_res = st.columns([1, 2])

    with col_sel:
        st.subheader("Select Crag")
        crag_options = {row["name"]: row["crag_id"] for _, row in crag_df.iterrows()}
        selected_name = st.selectbox("Crag", options=list(crag_options.keys()), index=0)
        selected_id = int(crag_options[selected_name])
        selected_crag = crag_df[crag_df["crag_id"] == selected_id].iloc[0]

        st.markdown("---")
        st.markdown(f"**Elevation:** {selected_crag['elevation_m']} m")
        st.markdown(f"**Rock type:** {selected_crag['rocks']}")
        st.markdown(f"**Rain exposure:** {selected_crag['rain_exposure']}")
        st.markdown(f"**Sun exposure:** {selected_crag['sun_exposure']}")
        st.markdown(
            f"**Coordinates:** {selected_crag['latitude']:.4f}°N,"
            f" {selected_crag['longitude']:.4f}°E"
        )
        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    with col_res:
        if predict_btn:
            with st.spinner("Fetching weather & running prediction…"):
                rt = fetch_forecast_current(
                    float(selected_crag["latitude"]),
                    float(selected_crag["longitude"]),
                )
                fv, model_pipeline = get_hopsworks_resources()

                # ── Feature retrieval: online store (SQL) → batch fallback ──
                feat_row_df = None
                try:
                    fv_vec = fv.get_feature_vector(
                        {"crag_id": selected_id},
                        return_type="pandas",
                    )
                    feat_row_df = fv_vec.drop(
                        columns=["date", "crag_id"], errors="ignore"
                    )
                except Exception:
                    pass  # fall through to batch

                if feat_row_df is None or feat_row_df.empty:
                    batch_df = get_batch_features()
                    crag_rows = (
                        batch_df[batch_df["crag_id"] == selected_id]
                        .sort_values("date")
                        .tail(1)
                    )
                    if crag_rows.empty:
                        st.error(
                            "No features found for this crag. "
                            "Run the feature pipeline first."
                        )
                        st.stop()
                    feat_row_df = crag_rows.drop(
                        columns=["date", "crag_id"], errors="ignore"
                    )

                X = feat_row_df.copy()
                X["rt_temperature"] = rt.get("temperature_2m", 0)
                X["rt_wind_speed"] = rt.get("wind_speed_10m", 0)
                X["rt_cloud_cover"] = rt.get("cloud_cover", 0)
                X["rt_precipitation"] = rt.get("precipitation", 0)

                pred, prob = _predict_row(X, model_pipeline)

            # ── Result ─────────────────────────────────────────────────
            if pred == 1:
                st.success(f"## ✓ {selected_name} is CLIMBABLE today!")
            else:
                st.error(f"## ✗ {selected_name} is NOT climbable today")
            _rule = (
                f"{prob:.1%} ≥ 50% → climbable" if pred == 1
                else f"{prob:.1%} < 50% → not climbable"
            )
            col_prob, col_rule = st.columns(2)
            col_prob.metric("Climbability probability", f"{prob:.1%}")
            col_rule.metric("Decision rule", _rule)

            # ── 3-day forecast ──────────────────────────────────────────
            st.markdown("### 3-Day Forecast")
            try:
                from datetime import date as _date
                fc_raw = fetch_forecast_3day(
                    float(selected_crag["latitude"]),
                    float(selected_crag["longitude"]),
                )
                fc_featured = add_rolling_features(
                    fc_raw.sort_values("date").reset_index(drop=True)
                )
                today_d = _date.today()
                future_rows = fc_featured[
                    fc_featured["date"].apply(
                        lambda d: (d if isinstance(d, _date) else d.date()) > today_d
                    )
                ].head(3).reset_index(drop=True)

                if future_rows.empty:
                    st.caption("No forecast data available.")
                else:
                    fc_cols = st.columns(len(future_rows))
                    _RAW_WEATHER = [
                        "precipitation_sum", "wind_speed_10m_max",
                        "sunshine_duration", "temperature_2m_max",
                        "temperature_2m_min", "shortwave_radiation_sum",
                    ]
                    _ROLLING = [
                        "rain_3d_sum", "rain_7d_sum", "wind_3d_avg",
                        "sun_3d_hours", "days_since_rain",
                    ]
                    _DAY_LABELS = ["Tomorrow", "+2 days", "+3 days"]
                    for i, fc_row in future_rows.iterrows():
                        base = feat_row_df.iloc[0].to_dict()
                        for col in _RAW_WEATHER + _ROLLING:
                            if col in fc_row.index:
                                base[col] = fc_row[col]
                        X_fc = pd.DataFrame([base])
                        fc_pred, fc_prob = _predict_row(X_fc, model_pipeline)
                        fc_date = fc_row["date"]
                        with fc_cols[i]:
                            st.markdown(f"**{_DAY_LABELS[i]}**  \n{fc_date}")
                            if fc_pred == 1:
                                st.success(f"✓ {fc_prob:.0%}")
                            else:
                                st.error(f"✗ {fc_prob:.0%}")
                            rain = fc_row.get("precipitation_sum", 0) or 0
                            wind = fc_row.get("wind_speed_10m_max", 0) or 0
                            st.caption(f"Rain {rain:.1f} mm · Wind {wind:.1f} km/h")
            except Exception as _e:
                st.caption(f"Forecast unavailable: {_e}")

            # ── Current weather ─────────────────────────────────────────
            st.markdown("### Current Weather")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temperature", f"{rt.get('temperature_2m', 0):.1f} °C")
            c2.metric("Wind Speed", f"{rt.get('wind_speed_10m', 0):.1f} km/h")
            c3.metric("Cloud Cover", f"{rt.get('cloud_cover', 0):.0f} %")
            c4.metric("Precipitation", f"{rt.get('precipitation', 0):.1f} mm")

            # ── Rolling feature summary ─────────────────────────────────
            st.markdown("### Rolling Features (last window)")
            feat_row = feat_row_df.iloc[0]

            def _fmt(col: str, unit: str = "") -> str:
                v = feat_row.get(col)
                if v is None:
                    return "N/A"
                if isinstance(v, float) and np.isnan(v):
                    return "N/A"
                if isinstance(v, (int, float, np.integer, np.floating)):
                    return f"{v:.1f}{unit}"
                return "N/A"

            feat_df = pd.DataFrame(
                {
                    "Feature": [
                        "Rain 3d sum",
                        "Rain 7d sum",
                        "Wind 3d avg",
                        "Sun 3d hours",
                        "Days since rain",
                    ],
                    "Value": [
                        _fmt("rain_3d_sum", " mm"),
                        _fmt("rain_7d_sum", " mm"),
                        _fmt("wind_3d_avg", " km/h"),
                        _fmt("sun_3d_hours", " h"),
                        _fmt("days_since_rain"),
                    ],
                }
            )
            st.table(feat_df)

            # ── Mini map ────────────────────────────────────────────────
            st.markdown("### Location")
            single_result = pd.DataFrame(
                [{"crag_id": selected_id, "prediction": pred, "probability": prob}]
            )
            st.pydeck_chart(
                _build_map(
                    crag_df[crag_df["crag_id"] == selected_id].reset_index(drop=True),
                    single_result,
                )
            )

        else:
            st.info("Select a crag and click **Predict** to run the climbability model.")
            st.pydeck_chart(
                _build_map(
                    crag_df[crag_df["crag_id"] == selected_id].reset_index(drop=True)
                )
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — All crags map
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("All Crags — Batch Prediction")
    st.markdown(
        "Click **Predict All** to run the model for every crag and update the map."
    )

    if st.button("Predict All Crags", type="primary"):
        with st.spinner("Fetching weather for all crags…"):
            batch_df = get_batch_features()
            latest_df = (
                batch_df.sort_values("date")
                .groupby("crag_id")
                .tail(1)
                .reset_index(drop=True)
            )

            rt_rows: list[dict] = []
            progress = st.progress(0.0, text="Fetching real-time weather…")
            n = len(crag_df)
            for i, (_, crag) in enumerate(crag_df.iterrows()):
                rt = fetch_forecast_current(
                    float(crag["latitude"]), float(crag["longitude"])
                )
                rt["crag_id"] = int(crag["crag_id"])
                rt_rows.append(rt)
                progress.progress((i + 1) / n, text=f"Weather: {crag['name']}")
            progress.empty()

        with st.spinner("Running predictions…"):
            rt_df = pd.DataFrame(rt_rows).rename(
                columns={
                    "temperature_2m": "rt_temperature",
                    "wind_speed_10m": "rt_wind_speed",
                    "cloud_cover": "rt_cloud_cover",
                    "precipitation": "rt_precipitation",
                }
            )
            inference_df = pd.merge(latest_df, rt_df, on="crag_id", how="left")
            X_all = inference_df.drop(
                columns=["date", "crag_id"], errors="ignore"
            ).copy()

            _, model_pipeline = get_hopsworks_resources()
            if hasattr(model_pipeline, "feature_names_in_"):
                X_all = X_all.reindex(
                    columns=model_pipeline.feature_names_in_, fill_value=0
                )
            probs = model_pipeline.predict_proba(X_all)[:, 1]
            preds = (probs >= 0.5).astype(int)  # consistent with _predict_row

        def _col(col_name: str):
            return (
                inference_df[col_name].values
                if col_name in inference_df.columns
                else [None] * len(inference_df)
            )

        crag_names = crag_df.set_index("crag_id")["name"]
        results_df = pd.DataFrame(
            {
                "crag_id": inference_df["crag_id"].values,
                "crag_name": inference_df["crag_id"].map(crag_names).values,
                "prediction": preds,
                "probability": probs,
                "rt_temperature": _col("rt_temperature"),
                "rt_wind_speed": _col("rt_wind_speed"),
                "rt_cloud_cover": _col("rt_cloud_cover"),
                "rt_precipitation": _col("rt_precipitation"),
                "rain_3d_mm": _col("rain_3d_sum"),
            }
        )

        # ── Summary metrics ─────────────────────────────────────────────
        n_climb = int(preds.sum())
        mc1, mc2 = st.columns(2)
        mc1.metric("Climbable crags", f"{n_climb} / {len(preds)}")
        mc2.metric("Date", date.today().isoformat())

        # ── Map ─────────────────────────────────────────────────────────
        st.markdown("### Map")
        st.caption("Green = climbable  ·  Red = not climbable")
        st.pydeck_chart(_build_map(crag_df, results_df))

        # ── Results table ────────────────────────────────────────────────
        st.markdown("### Results")
        st.caption("Climbable = probability ≥ 50%")
        disp = results_df[
            ["crag_name", "prediction", "probability",
             "rt_temperature", "rt_wind_speed", "rt_cloud_cover", "rt_precipitation",
             "rain_3d_mm"]
        ].copy()
        disp["Climbable?"] = disp["prediction"].map({1: "✓ YES", 0: "✗ NO"})
        disp["Probability"] = disp["probability"].apply(lambda x: f"{x:.1%}")
        disp = (
            disp[["crag_name", "Climbable?", "Probability",
                  "rt_temperature", "rt_wind_speed", "rt_cloud_cover", "rt_precipitation",
                  "rain_3d_mm"]]
            .rename(
                columns={
                    "crag_name": "Crag",
                    "rt_temperature": "Temp (°C)",
                    "rt_wind_speed": "Wind (km/h)",
                    "rt_cloud_cover": "Cloud (%)",
                    "rt_precipitation": "Precip (mm)",
                    "rain_3d_mm": "Rain 3d (mm)",
                }
            )
            .sort_values("Climbable?", ascending=False)
            .reset_index(drop=True)
        )
        for col in ["Temp (°C)", "Wind (km/h)", "Cloud (%)", "Precip (mm)", "Rain 3d (mm)"]:
            if col in disp.columns:
                disp[col] = pd.to_numeric(disp[col], errors="coerce").round(1)
        st.dataframe(disp, use_container_width=True)

    else:
        st.info("Click **Predict All Crags** to run batch predictions and update the map.")
        st.pydeck_chart(_build_map(crag_df))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance (Champion vs. Challenger)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    hdr_col, btn_col = st.columns([0.85, 0.15])
    with hdr_col:
        st.subheader("Model Registry — Champion vs. Challenger")
    with btn_col:
        st.write("")  # vertical spacer to align button
        if st.button("🔄 Refresh", help="Re-fetch model versions from Hopsworks"):
            st.rerun()

    st.markdown(
        "Compare registered model versions. The **production** model (champion) "
        "is used by the inference pipeline and this app."
    )

    try:
        fv_unused, _ = get_hopsworks_resources()
        # Re-login not needed — use the cached project from get_hopsworks_resources
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT,
            cert_folder=tempfile.gettempdir(),
        )
        mr = project.get_model_registry()
        models = mr.get_models(MODEL_NAME)

        # ── Stage override helpers ────────────────────────────────────────
        _STAGES_PATH = "/Resources/model_stages.json"
        _STAGES_DIR  = "Resources"
        _STAGES_FILE = "model_stages.json"

        def _load_stage_overrides():
            """Read stage overrides JSON from Hopsworks Resources dataset."""
            dataset_api = project.get_dataset_api()
            try:
                resp = dataset_api.read_content(_STAGES_PATH)
                if resp is None:
                    return {}
                return json.loads(resp.content)
            except Exception:
                return {}

        def _save_stage_overrides(overrides: dict):
            """Persist stage overrides JSON to Hopsworks Resources dataset."""
            dataset_api = project.get_dataset_api()
            tmp_file = os.path.join(tempfile.mkdtemp(), _STAGES_FILE)
            with open(tmp_file, "w") as f:
                json.dump(overrides, f)
            dataset_api.upload(tmp_file, _STAGES_DIR, overwrite=True)

        def _get_stage(m, overrides=None):
            """Return stage string, checking overrides first then description prefix."""
            if overrides:
                stage = overrides.get(MODEL_NAME, {}).get(str(m.version))
                if stage:
                    return stage
            desc = m.description or ""
            if desc.startswith("[production]"):
                return "production"
            elif desc.startswith("[staging]"):
                return "staging"
            elif desc.startswith("[archived]"):
                return "archived"
            return "untagged"

        if not models:
            st.warning("No models registered yet. Run the training pipeline first.")
        else:
            stage_overrides = _load_stage_overrides()

            # Build comparison table
            rows = []
            for m in sorted(models, key=lambda x: x.version, reverse=True):
                metrics = m.training_metrics or {}
                stage = _get_stage(m, stage_overrides)
                rows.append({
                    "Version": m.version,
                    "Stage": stage.upper(),
                    "Accuracy": metrics.get("accuracy"),
                    "F1 (macro)": metrics.get("f1_score"),
                    "Description": (m.description or "")[:60],
                })

            model_df = pd.DataFrame(rows)

            # Highlight champion
            champion = model_df[model_df["Stage"] == "PRODUCTION"]
            challenger = model_df[model_df["Stage"] == "STAGING"]

            # Summary metrics
            col_champ, col_chall = st.columns(2)
            with col_champ:
                st.markdown("#### 🏆 Champion (Production)")
                if not champion.empty:
                    c = champion.iloc[0]
                    st.metric("Version", f"v{c['Version']}")
                    st.metric("Accuracy", f"{c['Accuracy']:.4f}" if c['Accuracy'] else "N/A")
                    st.metric("F1 Score", f"{c['F1 (macro)']:.4f}" if c['F1 (macro)'] else "N/A")
                else:
                    st.info("No champion yet. Use the **Promote to Champion** button on the right.")

            def _do_promote(challenger_version: int, all_models: list):
                """Store stage override in Hopsworks Dataset (avoids PUT/duplicate-key error)."""
                overrides = _load_stage_overrides()
                model_ov = overrides.setdefault(MODEL_NAME, {})
                for m in all_models:
                    current = _get_stage(m, overrides)
                    if m.version == challenger_version:
                        model_ov[str(m.version)] = "production"
                    elif current == "production":
                        model_ov[str(m.version)] = "archived"
                _save_stage_overrides(overrides)

            with col_chall:
                st.markdown("#### 🥊 Latest Challenger (Staging)")
                if not challenger.empty:
                    ch = challenger.iloc[0]
                    st.metric("Version", f"v{ch['Version']}")
                    st.metric("Accuracy", f"{ch['Accuracy']:.4f}" if ch['Accuracy'] else "N/A")
                    st.metric("F1 Score", f"{ch['F1 (macro)']:.4f}" if ch['F1 (macro)'] else "N/A")

                    # Delta vs champion
                    if not champion.empty and ch['Accuracy'] and champion.iloc[0]['Accuracy']:
                        delta_acc = ch['Accuracy'] - champion.iloc[0]['Accuracy']
                        delta_f1 = (ch['F1 (macro)'] or 0) - (champion.iloc[0]['F1 (macro)'] or 0)
                        st.metric("Δ Accuracy vs Champion", f"{delta_acc:+.4f}")
                        st.metric("Δ F1 vs Champion", f"{delta_f1:+.4f}")

                    st.markdown("")
                    if st.button(
                        f"🚀 Promote v{ch['Version']} to Champion",
                        type="primary",
                        help="Sets this version to production and archives the current champion.",
                    ):
                        with st.spinner("Promoting…"):
                            _do_promote(int(ch["Version"]), models)
                        st.success(f"v{ch['Version']} is now the champion!")
                        get_hopsworks_resources.clear()
                        st.rerun()
                else:
                    st.info("No challenger. Run the training pipeline to create one.")

            # Full version history
            st.markdown("---")
            st.markdown("#### All Registered Versions")
            _BADGE = {
                "PRODUCTION": "🏆 Production",
                "STAGING":    "🥊 Staging",
                "ARCHIVED":   "📦 Archived",
                "UNTAGGED":   "—  Untagged",
            }
            display_df = model_df.copy()
            display_df["Status"] = display_df["Stage"].map(_BADGE).fillna(display_df["Stage"])
            display_df["Accuracy"] = display_df["Accuracy"].apply(
                lambda v: f"{v:.4f}" if v is not None else "—"
            )
            display_df["F1 (macro)"] = display_df["F1 (macro)"].apply(
                lambda v: f"{v:.4f}" if v is not None else "—"
            )
            st.dataframe(
                display_df[["Status", "Version", "Accuracy", "F1 (macro)", "Description"]],
                use_container_width=True,
                hide_index=True,
            )

            # Bar chart comparison
            chart_df = model_df[model_df["Accuracy"].notna()].copy()
            if not chart_df.empty:
                st.markdown("#### Metrics by Version")
                import altair as alt
                chart_df["Version"] = chart_df["Version"].apply(lambda v: f"v{v}")
                melted = chart_df.melt(
                    id_vars=["Version", "Stage"],
                    value_vars=["Accuracy", "F1 (macro)"],
                    var_name="Metric",
                    value_name="Score",
                )
                chart = (
                    alt.Chart(melted)
                    .mark_bar()
                    .encode(
                        x=alt.X("Version:N", sort=None),
                        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.5, 1.0])),
                        color="Metric:N",
                        column="Metric:N",
                        tooltip=["Version", "Metric", "Score", "Stage"],
                    )
                    .properties(width=200, height=300)
                )
                st.altair_chart(chart)

    except Exception as e:
        st.error(f"Could not load model registry data: {e}")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: [oblyk.org](https://oblyk.org) · "
    "[thecrag.com](https://www.thecrag.com) · "
    "[ukclimbing.com](https://www.ukclimbing.com) (crags & ascents) · "
    "[Open-Meteo](https://open-meteo.com) (weather) · "
    "MLOps project — CAS AI Operations"
)
