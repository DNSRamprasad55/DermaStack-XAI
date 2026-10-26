"""
DermaStack-XAI — Streamlit Research Interface (Stage 7)
========================================================
An Explainable Multi-Model Deep Learning Framework for Skin Lesion Classification.

Launch command:
    py -3.12 -m streamlit run app/app.py

Research prototype only. NOT a medical device. See disclaimer below.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image

# ── Project root must be on sys.path when launched via `streamlit run app/app.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES_MAPPING,
    HAM10000_CLASSES,
    MEDICAL_DISCLAIMER,
    DEFAULT_CONFIG,
)
from src.inference.pipeline import (
    BASE_MODELS,
    ENSEMBLE_NOTICE,
    MODEL_DISPLAY_NAMES,
    TARGET_LAYERS,
    format_probability_table,
    load_all_fold_models,
    load_stacking_meta_model,
    run_full_inference,
    run_gradcam,
    top_prediction,
    validate_and_load_image,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG  (must be FIRST Streamlit call)
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DermaStack-XAI | Skin Lesion Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# 2.  CUSTOM CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Typography ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* ── Header banner ───────────────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #0f3460;
}
.main-header h1 { color: #e94560; font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p  { color: #a8b2d8; font-size: 1.0rem; margin: 0.4rem 0 0; }

/* ── Metric cards ────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
    border: 1px solid #2a4a6b;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.metric-card .label  { color: #6b84a3; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value  { color: #e2e8f0; font-size: 1.5rem; font-weight: 600; margin-top: 0.2rem; }
.metric-card .sublabel { color: #94a3b8; font-size: 0.85rem; margin-top: 0.1rem; }

/* ── Prediction highlight ────────────────────────────────────────────────── */
.prediction-box {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a5c 100%);
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin: 1rem 0;
}
.prediction-box .pred-label  { color: #60a5fa; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; }
.prediction-box .pred-class  { color: #f8fafc; font-size: 1.7rem; font-weight: 700; margin: 0.3rem 0; }
.prediction-box .pred-prob   { color: #34d399; font-size: 1.1rem; font-weight: 500; }
.prediction-box .pred-code   { color: #94a3b8; font-size: 0.85rem; font-family: monospace; }

/* ── Section dividers ────────────────────────────────────────────────────── */
.section-header {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.4rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #2a4a6b;
}

/* ── Disclaimer box ─────────────────────────────────────────────────────── */
.disclaimer-box {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-size: 0.83rem;
    color: #fca5a5;
    margin-bottom: 1rem;
}

/* ── Tag / badge ─────────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.2em 0.75em;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 0.4rem;
}
.badge-blue   { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.badge-green  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-yellow { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* ── Heatmap caption ─────────────────────────────────────────────────────── */
.heatmap-caption {
    font-size: 0.78rem;
    color: #6b84a3;
    text-align: center;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  CACHED MODEL LOADERS  (loaded once per Streamlit session)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading Custom CNN checkpoints…")
def _cached_cnn():
    return load_all_fold_models("custom_cnn")

@st.cache_resource(show_spinner="Loading EfficientNetB0 checkpoints…")
def _cached_eff():
    return load_all_fold_models("efficientnet_b0")

@st.cache_resource(show_spinner="Loading ResNet50 checkpoints…")
def _cached_res():
    return load_all_fold_models("resnet50")

@st.cache_resource(show_spinner="Loading Standard Stacking meta-classifier…")
def _cached_meta_std():
    return load_stacking_meta_model(balanced=False)

@st.cache_resource(show_spinner="Loading Balanced Stacking meta-classifier…")
def _cached_meta_bal():
    return load_stacking_meta_model(balanced=True)


def _check_artifacts_present() -> List[str]:
    """Returns a list of missing critical artifact paths (empty = all present)."""
    required = (
        [DEFAULT_CONFIG.paths.checkpoints_dir / f"{m}_fold_{f}_best.keras"
         for m in BASE_MODELS for f in range(5)]
        + [DEFAULT_CONFIG.paths.meta_models_dir / "stacking_meta_model.joblib",
           DEFAULT_CONFIG.paths.meta_models_dir / "stacking_meta_model_balanced.joblib"]
    )
    return [str(p) for p in required if not p.exists()]


# ═════════════════════════════════════════════════════════════════════════════
# 4.  HEADER & DISCLAIMER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🔬 DermaStack-XAI</h1>
  <p>Explainable Multi-Model Deep Learning Framework for Dermoscopic Skin Lesion Classification</p>
  <p style="margin-top:0.5rem; font-size:0.8rem; color:#6b84a3;">
    HAM10000 · Custom CNN · EfficientNetB0 · ResNet50 · Logistic Regression Stacking · Grad-CAM XAI
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="disclaimer-box">
⚠️ <strong>RESEARCH PROTOTYPE DISCLAIMER</strong> — {MEDICAL_DISCLAIMER}
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  SIDEBAR — SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Inference Settings")

    # ── Model selection ──────────────────────────────────────────────────────
    st.markdown("### 🧠 Model / Ensemble Configuration")
    MODE_LABELS = {
        "standard_stacking":
            "🏆 Standard Stacking  (best accuracy)",
        "balanced_stacking":
            "🎯 Balanced Stacking  (best sensitivity)",
        "custom_cnn":
            "🔷 Custom Lightweight CNN",
        "efficientnet_b0":
            "🔶 EfficientNetB0",
        "resnet50":
            "🔴 ResNet50",
    }
    selected_mode = st.selectbox(
        "Select inference engine",
        list(MODE_LABELS.keys()),
        format_func=lambda k: MODE_LABELS[k],
        index=0,
    )

    is_stacking = selected_mode in ("standard_stacking", "balanced_stacking")

    if is_stacking:
        st.info(
            "The **Standard** configuration maximises overall accuracy (73.1% test).  \n"
            "The **Balanced** configuration maximises minority-class sensitivity "
            "(Melanoma recall 54.1% vs 14.5%).  \n\n"
            "Neither configuration is a clinically validated diagnostic mode."
        )

    st.markdown("---")

    # ── Grad-CAM settings ────────────────────────────────────────────────────
    st.markdown("### 🔍 Grad-CAM Explainability")
    enable_gradcam = st.checkbox("Generate Grad-CAM heatmaps", value=True)

    if enable_gradcam:
        xai_class_mode = st.radio(
            "Explain class",
            ["Predicted class (argmax)", "Choose class manually"],
            index=0,
        )
        xai_class_idx: Optional[int] = None
        if xai_class_mode == "Choose class manually":
            chosen_cls = st.selectbox(
                "Target diagnostic class",
                HAM10000_CLASSES,
                format_func=lambda c: f"{c} — {CLASS_NAMES_MAPPING[c]}",
            )
            xai_class_idx = HAM10000_CLASSES.index(chosen_cls)

        colormap_choice = st.selectbox(
            "Heatmap colormap",
            ["jet", "viridis", "inferno", "plasma", "hot"],
            index=0,
        )
        heatmap_alpha = st.slider(
            "Heatmap overlay opacity",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
        )
        gradcam_fold = st.selectbox(
            "Checkpoint fold for Grad-CAM",
            list(range(5)),
            index=0,
            help="Grad-CAM is computed against a single fold checkpoint. Fold 0 is the default.",
        )

    st.markdown("---")

    # ── Artifact status ──────────────────────────────────────────────────────
    st.markdown("### 📦 Artifact Status")
    missing = _check_artifacts_present()
    if missing:
        st.error(f"⛔ {len(missing)} artifact(s) missing. Run Stages 3–4 first.")
        with st.expander("Missing files"):
            for m in missing:
                st.code(m, language="text")
    else:
        st.success("✅ All 17 model artifacts present (15 checkpoints + 2 meta-models).")

    st.markdown("---")
    st.caption(
        "DermaStack-XAI · TensorFlow 2.21 · Keras 3.15 · scikit-learn 1.x · "
        "Streamlit 1.64 · Python 3.12"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN CONTENT — IMAGE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
col_upload, col_results = st.columns([1, 1.3], gap="large")

with col_upload:
    st.markdown('<p class="section-header">📤 1 · Upload Dermoscopic Image</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    rgb_array: Optional[np.ndarray] = None
    upload_error: Optional[str] = None

    if uploaded_file is not None:
        try:
            rgb_array, orig_size = validate_and_load_image(uploaded_file)
            pil_display = Image.fromarray(rgb_array)
            st.image(pil_display, caption="Uploaded dermoscopic image", use_container_width=True)
            st.caption(
                f"Format: {uploaded_file.type}  ·  "
                f"Original size: {orig_size[0]}×{orig_size[1]} px  ·  "
                f"Converted to: RGB 3-channel"
            )
        except ValueError as exc:
            upload_error = str(exc)
            st.error(f"❌ Invalid image: {upload_error}")
    else:
        st.info(
            "Upload a JPEG or PNG dermoscopic image to begin analysis.\n\n"
            "Images are processed locally. No data is transmitted externally."
        )

    # Class reference table
    with st.expander("📋 HAM10000 Diagnostic Classes Reference", expanded=False):
        for code in HAM10000_CLASSES:
            st.markdown(
                f"**`{code}`** — {CLASS_NAMES_MAPPING[code]}  \n"
                f"<span style='color:#6b84a3;font-size:0.83rem;'>{CLASS_DESCRIPTIONS[code]}</span>",
                unsafe_allow_html=True
            )
            st.markdown("")


# ═════════════════════════════════════════════════════════════════════════════
# 7.  MAIN CONTENT — PREDICTION RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with col_results:
    st.markdown('<p class="section-header">📊 2 · Prediction & Analysis</p>', unsafe_allow_html=True)

    if upload_error:
        st.warning("Fix the image upload error in the left panel before running inference.")
    elif rgb_array is None:
        st.info("Awaiting image upload…")
    elif missing:
        st.error("⛔ Model artifacts are missing. See the sidebar for details.")
    else:
        run_btn = st.button(
            "🚀 Run Inference",
            type="primary",
            use_container_width=True,
        )

        if run_btn or st.session_state.get("last_inference") is not None:
            # ── Load models (cached) ───────────────────────────────────────
            load_error = None
            try:
                fold_cnn = _cached_cnn()
                fold_eff = _cached_eff()
                fold_res = _cached_res()
                meta_std = _cached_meta_std()
                meta_bal = _cached_meta_bal()
            except Exception as exc:
                load_error = exc

            if load_error:
                st.error(f"❌ Failed to load model artifacts: {load_error}")
                st.code(traceback.format_exc(), language="text")
            else:
                # ── Run inference ──────────────────────────────────────────
                if run_btn:
                    with st.spinner("Running inference across all models…"):
                        try:
                            infer_result = run_full_inference(
                                rgb_array=rgb_array,
                                fold_models_cnn=fold_cnn,
                                fold_models_eff=fold_eff,
                                fold_models_res=fold_res,
                                meta_std=meta_std,
                                meta_bal=meta_bal,
                            )
                            st.session_state["last_inference"] = infer_result
                        except Exception as exc:
                            st.error(f"❌ Inference failed: {exc}")
                            st.code(traceback.format_exc(), language="text")
                            st.stop()

                infer_result = st.session_state.get("last_inference")
                if infer_result is None:
                    st.stop()

                # ── Display prediction for selected mode ───────────────────
                sel_probs    = infer_result[selected_mode]["probs"]
                sel_prob_dict = infer_result[selected_mode]["prob_dict"]
                pred_code, pred_name, pred_prob = top_prediction(sel_probs)

                # Research-only class label. Do not convert a model class into a
                # benign/malignant diagnosis or reassurance.
                badge_cls = "badge-blue"
                badge_text = "Experimental class prediction — not a diagnosis"

                st.markdown(f"""
<div class="prediction-box">
  <div class="pred-label">Top Predicted Class · {MODEL_DISPLAY_NAMES.get(selected_mode, selected_mode)}</div>
  <div class="pred-class">{pred_name}</div>
  <div class="pred-code">Class code: <code>{pred_code}</code></div>
  <div class="pred-prob">Model score: {pred_prob*100:.1f}%</div>
  <div style="margin-top:0.6rem;">
    <span class="badge {badge_cls}">{badge_text}</span>
  </div>
</div>
""", unsafe_allow_html=True)

                st.warning(
                    "This model was trained on dermoscopic images (HAM10000). "
                    "Ordinary phone photos, screenshots, or images with surrounding skin may be "
                    "outside its training domain; this prediction may be unreliable. "
                    "The displayed score is not the probability that a lesion is harmless."
                )

                # Research notice for stacking modes
                if is_stacking:
                    st.caption(
                        "ℹ️ Stacking meta-classifier output is derived from the 5-fold ensemble "
                        "probability distributions of all three base models. It is NOT refitted on "
                        "this image."
                    )

                # Class description
                st.caption(f"📖 {CLASS_DESCRIPTIONS.get(pred_code, '')}")

                # ── Probability bar chart ──────────────────────────────────
                st.markdown('<p class="section-header">📈 Model Score Distribution</p>',
                            unsafe_allow_html=True)

                import pandas as pd
                prob_rows = format_probability_table(sel_prob_dict)
                df_probs = pd.DataFrame(prob_rows).set_index("Class")[["Probability"]]

                st.bar_chart(df_probs, use_container_width=True, height=220)

                with st.expander("View full probability table"):
                    df_display = pd.DataFrame(prob_rows)[["Class", "Diagnosis", "Probability (%)"]]
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

                # ── All-model comparison ───────────────────────────────────
                with st.expander("🔎 Compare all 5 model configurations", expanded=False):
                    comp_rows = []
                    for key in ["custom_cnn", "efficientnet_b0", "resnet50",
                                "standard_stacking", "balanced_stacking"]:
                        p = infer_result[key]["probs"]
                        c, n, prob = top_prediction(p)
                        comp_rows.append({
                            "Model": MODEL_DISPLAY_NAMES[key],
                            "Predicted Class": f"{c} ({n})",
                            "Top-class score": f"{prob*100:.1f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(comp_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                # ── Disclaimer reminder ────────────────────────────────────
                st.markdown("""
<div class="disclaimer-box" style="font-size:0.78rem;">
🚫 These scores are <strong>experimental model outputs</strong>, not calibrated clinical risks or diagnoses.
Do not make health decisions based on these results. Consult a qualified dermatologist about any concerning lesion.
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 8.  GRAD-CAM SECTION
# ═════════════════════════════════════════════════════════════════════════════
if rgb_array is not None and not missing:
    infer_result = st.session_state.get("last_inference")
    if infer_result is not None and enable_gradcam:
        st.markdown("---")
        st.markdown('<p class="section-header">🌡️ 3 · Grad-CAM Visual Explanations</p>',
                    unsafe_allow_html=True)

        # Ensemble attribution notice
        if is_stacking:
            st.info(ENSEMBLE_NOTICE)

        st.caption(
            "Grad-CAM heatmaps highlight image regions that drove each model's prediction "
            "for the selected class. Warm colours (red/yellow) indicate high feature importance. "
            "Heatmaps do NOT establish biological causation or clinical validity."
        )

        run_xai = st.button(
            "🔥 Generate Grad-CAM Heatmaps",
            type="secondary",
            use_container_width=False,
        )

        if run_xai or st.session_state.get("gradcam_results") is not None:
            if run_xai:
                gradcam_results: Dict[str, Any] = {}
                gradcam_errors:  Dict[str, str] = {}

                for model_name in BASE_MODELS:
                    with st.spinner(f"Computing Grad-CAM for {MODEL_DISPLAY_NAMES[model_name]}…"):
                        try:
                            res = run_gradcam(
                                rgb_array=rgb_array,
                                model_name=model_name,
                                fold_idx=gradcam_fold,
                                class_idx=xai_class_idx,
                                alpha=heatmap_alpha,
                                colormap=colormap_choice,
                            )
                            gradcam_results[model_name] = res
                        except Exception as exc:
                            gradcam_errors[model_name] = str(exc)
                            st.warning(
                                f"⚠️ Grad-CAM failed for {MODEL_DISPLAY_NAMES[model_name]}: {exc}"
                            )

                st.session_state["gradcam_results"] = gradcam_results
                st.session_state["gradcam_errors"]  = gradcam_errors

            gradcam_results = st.session_state.get("gradcam_results", {})
            gradcam_errors  = st.session_state.get("gradcam_errors",  {})

            if gradcam_results:
                n_models = len(gradcam_results)
                cols = st.columns(n_models, gap="medium")

                for i, (model_name, res) in enumerate(gradcam_results.items()):
                    with cols[i]:
                        exp_code  = res["explained_class_name"]
                        exp_name  = CLASS_NAMES_MAPPING.get(exp_code, exp_code)
                        exp_prob  = res["class_probability"]
                        tgt_layer = res["target_layer"]

                        st.markdown(
                            f"**{MODEL_DISPLAY_NAMES[model_name]}**  \n"
                            f"<span class='badge badge-blue'>{tgt_layer}</span>",
                            unsafe_allow_html=True,
                        )

                        # Side-by-side original vs overlay
                        sub_c1, sub_c2 = st.columns(2)
                        with sub_c1:
                            st.image(
                                res["original_image"],
                                caption="Original",
                                use_container_width=True,
                            )
                        with sub_c2:
                            st.image(
                                res["overlay_image"],
                                caption="Grad-CAM overlay",
                                use_container_width=True,
                            )

                        st.markdown(
                            f"<div class='heatmap-caption'>"
                            f"Explained: <b>{exp_code}</b> — {exp_name}<br/>"
                            f"Model probability: <b>{exp_prob*100:.1f}%</b><br/>"
                            f"Fold {gradcam_fold} checkpoint · {colormap_choice} colormap"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # Per-model probability table inside expander
                        with st.expander(f"All class probs ({model_name})"):
                            import pandas as pd
                            pdict = res["all_probabilities"]
                            rows = [{"Class": c, "Prob": f"{v*100:.1f}%"}
                                    for c, v in pdict.items()]
                            df = pd.DataFrame(rows).sort_values("Prob", ascending=False)
                            st.dataframe(df, use_container_width=True, hide_index=True)

            if gradcam_errors:
                for model_name, err_msg in gradcam_errors.items():
                    st.error(
                        f"❌ Grad-CAM failed for **{MODEL_DISPLAY_NAMES[model_name]}**: {err_msg}"
                    )


# ═════════════════════════════════════════════════════════════════════════════
# 9.  BENCHMARK CONTEXT PANEL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("📉 Stage 6 Holdout Test Benchmark Results (N = 1,431)", expanded=False):
    import json
    metrics_path = DEFAULT_CONFIG.paths.metrics_dir / "evaluation_summary.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            eval_summary = json.load(f)

        bench = eval_summary.get("holdout_test_benchmark", {})
        if bench:
            import pandas as pd
            rows = []
            for model_key in ["Custom CNN", "EfficientNetB0", "ResNet50",
                              "Standard Stacking", "Balanced Stacking"]:
                m = bench.get(model_key, {})
                if m:
                    rows.append({
                        "Model": model_key,
                        "Accuracy": f"{m.get('accuracy', 0)*100:.1f}%",
                        "Macro F1": f"{m.get('macro_f1', 0):.3f}",
                        "Macro Recall": f"{m.get('macro_recall', 0):.3f}",
                        "Macro ROC-AUC": str(m.get("roc_auc_ovr_macro", "—")),
                        "Brier Score": str(m.get("multiclass_brier_score", "—")),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "Stage 6 evaluation summary not found. Run "
            "`py -3.12 -m src.evaluation.evaluate_holdout` to generate it."
        )

    st.caption(
        "Benchmark results are from the frozen holdout test partition (N = 1,431). "
        "These results were computed ONCE after all training and hyperparameter decisions "
        "were finalised. They are not updated based on any uploaded image."
    )


# ═════════════════════════════════════════════════════════════════════════════
# 10.  FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; color:#4a6080; font-size:0.78rem; padding: 2rem 0 0.5rem;">
  DermaStack-XAI · Research Prototype · HAM10000 Benchmark · <br/>
  TensorFlow 2.21 · Keras 3.15 · scikit-learn · Streamlit 1.64 · Python 3.12<br/>
  <span style="color:#e94560;">NOT A MEDICAL DEVICE</span> — Do not use for clinical decision making.
</div>
""", unsafe_allow_html=True)
