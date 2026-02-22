"""
Age Prediction App  ·  EfficientNetB0 backbone
Outputs: exact age (regression) + age group (5-class classification)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st
from tensorflow import keras

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Age Predictor",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Constants (must match training) ──────────────────────────────────────────
IMAGE_SIZE = (224, 224)
AGE_GROUP_NAMES   = ["Child (0–12)", "Youth (13–25)", "Adult (26–42)",
                     "Middle Age (43–60)", "Senior (60+)"]
AGE_GROUP_COLOURS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"]

MODEL_PATH    = "age_prediction_best_finetune.keras"  # primary
FALLBACK_PATH = "age_prediction_model_final.keras"    # secondary

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-card {
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border: 1px solid #444;
    border-radius: 16px;
    padding: 24px 32px;
    margin-top: 16px;
    color: #f0f0f0;
}
.age-display {
    font-size: 3.5rem;
    font-weight: 800;
    color: #7ecfff;
    line-height: 1.0;
}
.group-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 6px;
    color: #fff;
}
.section-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #aaa;
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    for path in [MODEL_PATH, FALLBACK_PATH]:
        if Path(path).exists():
            return keras.models.load_model(path), path
    raise FileNotFoundError(
        f"No model found. Expected '{MODEL_PATH}' or '{FALLBACK_PATH}'."
    )


# ── YOLO face detector (optional — gracefully skipped if unavailable) ─────────
@st.cache_resource(show_spinner="Loading face detector…")
def load_yolo():
    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="face_yolov8n.pt",
            token=os.getenv("HF_TOKEN"),
        )
        return YOLO(weights)
    except Exception:
        return None


def crop_face_yolo(yolo_model, image_bgr: np.ndarray, conf_thres=0.3, margin=0.15):
    """Detect & crop the largest face. Returns uint8 RGB crop, or None."""
    result = yolo_model.predict(source=image_bgr, conf=conf_thres, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[np.argmax(areas)].astype(int)

    h, w = image_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)

    face = image_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Resize to 224×224 and normalise to [0, 1] — matches training pipeline."""
    resized = cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0


def age_to_group(age: float) -> int:
    if age <= 12: return 0
    if age <= 25: return 1
    if age <= 42: return 2
    if age <= 60: return 3
    return 4


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ About")
    st.markdown(
        """
        **Model:** EfficientNetB0 with YOLOv8 face cropping\n
        **Training:** Two-phase (warm-up + fine-tune)\n  
        **Outputs:**
        - 📊 Age group (5 classes)
        - 🎂 Exact age (regression)

        **Age groups:**
        """
    )
    for name, colour in zip(AGE_GROUP_NAMES, AGE_GROUP_COLOURS):
        st.markdown(
            f'<span style="background:{colour};border-radius:8px;padding:2px 10px;'
            f'color:#fff;font-size:0.85rem;">{name}</span>',
            unsafe_allow_html=True,
        )
        st.write("")

    st.markdown("---")
    use_yolo = st.checkbox("Auto-detect & crop face (YOLO)", value=True)
    st.caption("Unless... the image is already a tight face crop.")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🧬 Age Predictor")
st.markdown(
    "Upload a clear photo and the model will estimate the person's **exact age** and **age group**"
)
st.markdown("---")

# Load model
try:
    model, loaded_path = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Load YOLO if requested
yolo_model = load_yolo() if use_yolo else None
if use_yolo and yolo_model is None:
    st.info("ℹ️ YOLO face detector could not be loaded — using full image instead.")

# File uploader
uploaded = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "webp"],
    help="JPEG / PNG / WebP, any resolution. YOLO will crop the face automatically.",
)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Face detection & crop
    face_rgb = None
    if use_yolo and yolo_model is not None:
        face_rgb = crop_face_yolo(yolo_model, img_bgr)
        if face_rgb is None:
            st.warning(
                "⚠️ No face detected — predicting on the full image. "
                "Results may be less accurate."
            )

    working_img = face_rgb if face_rgb is not None else img_rgb

    # Show original + cropped side by side
    col_orig, col_crop = st.columns(2, gap="medium")
    with col_orig:
        st.markdown("**Uploaded image**")
        st.image(pil_img, use_container_width=True)
    with col_crop:
        st.markdown("**Face region used for prediction**")
        st.image(face_rgb if face_rgb is not None else pil_img, use_container_width=True)

    # ── Inference ────────────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        batch = np.expand_dims(preprocess(working_img), axis=0)   # (1, 224, 224, 3)
        preds = model.predict(batch, verbose=0)

    # Dual-output model: [age_group_probs (5,), age_value (1,)]
    if isinstance(preds, (list, tuple)) and len(preds) == 2:
        group_probs   = preds[0][0]            # shape (5,)
        predicted_age = float(preds[1][0][0])  # scalar
    else:
        predicted_age = float(preds[0][0])
        group_probs = None

    predicted_age = max(0.0, min(predicted_age, 100.0))

    if group_probs is not None:
        top_idx    = int(np.argmax(group_probs))
    else:
        top_idx    = age_to_group(predicted_age)
        group_probs = np.zeros(5)
        group_probs[top_idx] = 1.0

    top_name   = AGE_GROUP_NAMES[top_idx]
    top_colour = AGE_GROUP_COLOURS[top_idx]

    # ── Result card ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="result-card">
            <p class="section-label">Predicted Age</p>
            <div class="age-display">{predicted_age:.1f}
                <span style="font-size:1.4rem;color:#aaa;">yrs</span>
            </div>
            <br>
            <p class="section-label">Age Group</p>
            <span class="group-badge" style="background:{top_colour};">{top_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Confidence bars ───────────────────────────────────────────────────────
    st.markdown("#### Age Group Confidence")
    for idx, (name, prob, colour) in enumerate(
        zip(AGE_GROUP_NAMES, group_probs, AGE_GROUP_COLOURS)
    ):
        is_top = idx == top_idx
        label  = f"{'★ ' if is_top else ''}{name}"
        pct    = prob * 100
        st.markdown(
            f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.85rem;
                            color:{'#fff' if is_top else '#bbb'};
                            font-weight:{'700' if is_top else '400'};">
                    <span>{label}</span><span>{pct:.1f}%</span>
                </div>
                <div style="background:#2a2a3e;border-radius:6px;height:10px;overflow:hidden;">
                    <div style="width:{pct}%;background:{colour};height:100%;
                                border-radius:6px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption(f"Model: `{loaded_path}` · Input: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} px normalised [0, 1]")