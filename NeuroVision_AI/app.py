import os
import sys
import json
import time
import numpy as np
import cv2
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.model import build_model, compile_model
from src.gradcam import get_gradcam, overlay_gradcam
from src.llm_advisor import get_treatment_advice

IMG_SIZE = 224
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")
CPU_METRICS_PATH = os.path.join(MODEL_DIR, "cpu_training_metrics.json")
GPU_METRICS_PATH = os.path.join(MODEL_DIR, "gpu_training_metrics.json")

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1a1a2e; text-align: center; }
    .result-card { padding: 1.5rem; border-radius: 1rem; margin: 1rem 0; }
    .tumor-positive { background: linear-gradient(135deg, #ff6b6b33, #ff8e8e33); border: 2px solid #ff6b6b; }
    .tumor-negative { background: linear-gradient(135deg, #51cf6633, #69db7c33); border: 2px solid #51cf66; }
    .disclaimer { background: #fff3cd; border: 1px solid #ffc107; border-radius: 0.5rem; padding: 1rem; margin-top: 1rem; }
    .compare-card { padding: 1rem; border-radius: 0.75rem; border: 1px solid #dee2e6; background: #f8f9fa; }
    .cpu-card { border-left: 4px solid #4dabf7; }
    .gpu-card { border-left: 4px solid #ff6b6b; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model, _ = build_model(freeze_base=False)
    model = compile_model(model)
    model.load_weights(MODEL_PATH)
    return model


def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32)
    return img_rgb, img_normalized


def load_metrics(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def render_comparison_tab():
    st.markdown("## ⚡ CPU vs GPU Eğitim Karşılaştırması")
    st.markdown("AMD RX 7800 XT (DirectML) ile CPU eğitim sonuçlarını karşılaştırın.")

    cpu = load_metrics(CPU_METRICS_PATH)
    gpu = load_metrics(GPU_METRICS_PATH)

    if cpu is None and gpu is None:
        st.warning("Henüz eğitim metrikleri bulunamadı. Lütfen önce modeli eğitin:")
        st.code(
            "python src/train.py        # CPU ile eğit\npython src/train_gpu.py   # GPU ile eğit",
            language="bash",
        )
        return

    col_cpu, col_gpu = st.columns(2)

    with col_cpu:
        st.markdown("### 💻 CPU Eğitim")
        if cpu:
            st.markdown(f"**Cihaz:** {cpu.get('device', 'CPU')}")
            st.markdown(f"**Faz 1 Süresi:** {cpu.get('phase1_time', 0):.1f}s")
            st.markdown(f"**Faz 2 Süresi:** {cpu.get('phase2_time', 0):.1f}s")
            st.markdown(f"**Toplam Süre:** {cpu.get('total_time', 0):.1f}s")
            test = cpu.get("test", {})
            if test:
                st.markdown("---")
                st.markdown("**Test Metrikleri:**")
                st.markdown(f"- Accuracy: **%{test.get('accuracy', 0) * 100:.1f}**")
                st.markdown(
                    f"- Precision (Sağlıklı): **%{test.get('precision_saglikli', 0) * 100:.1f}**"
                )
                st.markdown(
                    f"- Recall (Sağlıklı): **%{test.get('recall_saglikli', 0) * 100:.1f}**"
                )
                st.markdown(
                    f"- Precision (Tümör): **%{test.get('precision_tumor', 0) * 100:.1f}**"
                )
                st.markdown(
                    f"- Recall (Tümör): **%{test.get('recall_tumor', 0) * 100:.1f}**"
                )
                st.markdown(f"- F1 (Tümör): **%{test.get('f1_tumor', 0) * 100:.1f}**")
                st.markdown(f"- AUC-ROC: **{test.get('auc_roc', 0):.3f}**")
        else:
            st.info(
                "CPU eğitim metrikleri bulunamadı. `python src/train.py` çalıştırın."
            )

    with col_gpu:
        st.markdown("### 🔴 GPU Eğitim (AMD RX 7800 XT)")
        if gpu:
            st.markdown(f"**Cihaz:** {gpu.get('device', 'GPU')}")
            st.markdown(f"**Faz 1 Süresi:** {gpu.get('phase1_time', 0):.1f}s")
            st.markdown(f"**Faz 2 Süresi:** {gpu.get('phase2_time', 0):.1f}s")
            st.markdown(f"**Toplam Süre:** {gpu.get('total_time', 0):.1f}s")
            test = gpu.get("test", {})
            if test:
                st.markdown("---")
                st.markdown("**Test Metrikleri:**")
                st.markdown(f"- Accuracy: **%{test.get('accuracy', 0) * 100:.1f}**")
                st.markdown(
                    f"- Precision (Sağlıklı): **%{test.get('precision_saglikli', 0) * 100:.1f}**"
                )
                st.markdown(
                    f"- Recall (Sağlıklı): **%{test.get('recall_saglikli', 0) * 100:.1f}**"
                )
                st.markdown(
                    f"- Precision (Tümör): **%{test.get('precision_tumor', 0) * 100:.1f}**"
                )
                st.markdown(
                    f"- Recall (Tümör): **%{test.get('recall_tumor', 0) * 100:.1f}**"
                )
                st.markdown(f"- F1 (Tümör): **%{test.get('f1_tumor', 0) * 100:.1f}**")
                st.markdown(f"- AUC-ROC: **{test.get('auc_roc', 0):.3f}**")
        else:
            st.info(
                "GPU eğitim metrikleri bulunamadı. `python src/train_gpu.py` çalıştırın."
            )

    if cpu and gpu:
        st.markdown("---")
        st.markdown("### 📊 Hız Karşılaştırması")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Faz 1",
                f"{cpu['phase1_time']:.1f}s",
                f"{cpu['phase1_time'] - gpu['phase1_time']:.1f}s",
                delta_color="inverse",
            )
        with col_b:
            st.metric(
                "Faz 2",
                f"{cpu['phase2_time']:.1f}s",
                f"{cpu['phase2_time'] - gpu['phase2_time']:.1f}s",
                delta_color="inverse",
            )
        with col_c:
            st.metric(
                "Toplam",
                f"{cpu['total_time']:.1f}s",
                f"{cpu['total_time'] - gpu['total_time']:.1f}s",
                delta_color="inverse",
            )

        speedup = cpu["total_time"] / max(gpu["total_time"], 0.01)
        st.info(f"🚀 GPU hız artışı: **{speedup:.2f}x**")

        st.markdown("---")
        st.markdown("### 📈 Doğruluk Karşılaştırması")
        cpu_test = cpu.get("test", {})
        gpu_test = gpu.get("test", {})
        metrics_names = [
            ("Accuracy", "accuracy"),
            ("Recall (Tümör)", "recall_tumor"),
            ("Precision (Tümör)", "precision_tumor"),
            ("F1 (Tümör)", "f1_tumor"),
            ("AUC-ROC", "auc_roc"),
        ]
        for name, key in metrics_names:
            c_val = cpu_test.get(key, 0)
            g_val = gpu_test.get(key, 0)
            diff = g_val - c_val
            delta_str = f"{diff:+.4f}" if key != "auc_roc" else f"{diff:+.3f}"
            fmt = ".4f" if key != "auc_roc" else ".3f"
            st.markdown(
                f"**{name}:** CPU {c_val:{fmt}} | GPU {g_val:{fmt}} | Fark {delta_str}"
            )

    st.markdown("---")
    st.markdown("### 🖼️ Eğitim Grafikleri")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        cpu_plot = os.path.join(MODEL_DIR, "training_history.png")
        if os.path.exists(cpu_plot):
            st.image(cpu_plot, caption="CPU Eğitim Geçmişi", use_container_width=True)
    with col_img2:
        gpu_plot = os.path.join(MODEL_DIR, "gpu_training_history.png")
        if os.path.exists(gpu_plot):
            st.image(gpu_plot, caption="GPU Eğitim Geçmişi", use_container_width=True)

    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        cpu_eval = os.path.join(MODEL_DIR, "evaluation.png")
        if os.path.exists(cpu_eval):
            st.image(cpu_eval, caption="CPU Değerlendirme", use_container_width=True)
    with col_eval2:
        gpu_eval = os.path.join(MODEL_DIR, "gpu_evaluation.png")
        if os.path.exists(gpu_eval):
            st.image(gpu_eval, caption="GPU Değerlendirme", use_container_width=True)


def main():
    with st.sidebar:
        st.markdown("## 🧠 NeuroVision AI")
        st.markdown("**Beyin Tümörü Tespit Sistemi**")
        st.markdown("---")
        st.markdown("### ⚙️ Ayarlar")
        llm_provider = st.selectbox(
            "LLM Sağlayıcı", ["Groq (Hızlı)", "Google Gemini"], index=0
        )
        st.markdown("---")
        st.markdown("### 📊 Model Bilgisi")
        st.markdown("- **Mimari**: EfficientNetB0")
        st.markdown("- **Eğitim**: Transfer Learning")
        st.markdown("- **Giriş**: 224×224 MRI")
        st.markdown("---")
        st.markdown("### ⚠️ Uyarı")
        st.warning(
            "Bu sistem bir karar destek aracıdır, tıbbi tanı koyma aracı değildir. Tüm sonuçlar bir uzman doktor tarafından doğrulanmalıdır."
        )

    tab1, tab2 = st.tabs(["🔍 Tümör Tespiti", "⚡ CPU vs GPU Karşılaştırma"])

    with tab1:
        st.markdown(
            '<p class="main-header">🧠 NeuroVision AI</p>', unsafe_allow_html=True
        )
        st.markdown(
            '<p style="text-align:center; color:#666;">Beyin MRI Görüntülerinden Tümör Tespiti ve Tedavi Önerisi</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "📁 MRI Görüntüsü Yükle",
            type=["jpg", "jpeg", "png"],
            help="Beyin MRI görüntüsünü yükleyin (.jpg, .jpeg, .png)",
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 📤 Yüklenen Görüntü")
                img_rgb, img_normalized = preprocess_image(uploaded_file.read())
                if img_rgb is None:
                    st.error("Görüntü okunamadı. Lütfen geçerli bir dosya yükleyin.")
                    return
                st.image(img_rgb, use_container_width=True)

            if not os.path.exists(MODEL_PATH):
                st.error(
                    f"Model dosyası bulunamadı: {MODEL_PATH}. Lütfen önce modeli eğitin: `python src/train.py`"
                )
                return

            model = load_model()

            with st.spinner("🔍 Analiz ediliyor..."):
                prediction_prob = model.predict(np.expand_dims(img_normalized, axis=0))[
                    0
                ][0]
                has_tumor = prediction_prob >= 0.5
                confidence = float(
                    prediction_prob if has_tumor else 1.0 - prediction_prob
                )
                label = "Tümör VAR" if has_tumor else "Sağlıklı"

            with col2:
                st.markdown("### 📊 Sonuç")
                if has_tumor:
                    st.markdown(
                        f'<div class="result-card tumor-positive"><h2>⚠️ {label}</h2><p>Tümör tespit edildi</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result-card tumor-negative"><h2>✅ {label}</h2><p>Tümör tespit edilmedi</p></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**Güven Oranı:** %{confidence * 100:.1f}")
                st.progress(confidence)

            st.markdown("---")

            col3, col4 = st.columns([1, 1])
            with col3:
                st.markdown("### 🔥 Grad-CAM Isı Haritası")
                try:
                    heatmap = get_gradcam(img_normalized, model)
                    overlay = overlay_gradcam(
                        cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)), heatmap
                    )
                    st.image(
                        overlay,
                        use_container_width=True,
                        caption="Tümör Bölgesi Isı Haritası",
                    )
                except Exception as e:
                    st.warning(f"Grad-CAM oluşturulamadı: {e}")

            with col4:
                st.markdown("### 🤖 Tedavi Önerisi (LLM)")
                provider_key = "groq" if "Groq" in llm_provider else "gemini"
                if st.button("💡 Tedavi Önerisi Al", key="llm_btn"):
                    with st.spinner("LLM yanıtı oluşturuluyor..."):
                        advice = get_treatment_advice(
                            label, confidence, provider=provider_key
                        )
                    st.markdown(advice)

            st.markdown("---")
            st.markdown(
                """
            <div class="disclaimer">
                <strong>⚠️ Tıbbi Sorumluluk Reddi:</strong> Bu uygulama bir karar destek aracıdır ve tıbbi tanı koyma amacıyla kullanılamaz. 
                Tüm sonuçlar bir nörolog veya onkolog tarafından doğrulanmalıdır. Sağlık sorunlarınız için mutlaka bir uzmana başvurun.
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            ### 📋 Nasıl Kullanılır?
            
            1. **MRI görüntüsü yükleyin** — Soldaki alana bir beyin MRI dosyası sürükleyin
            2. **Otomatik analiz** — Model görüntüyü analiz ederek tümör var/yok tahmini yapar
            3. **Grad-CAM ısı haritası** — Modelin hangi bölgeye odaklandığını görselleştirir
            4. **Tedavi önerisi** — LLM tabanlı tedavi önerisi alabilirsiniz
            
            ---
            
            *Desteklenen formatlar: .jpg, .jpeg, .png*
            """)

    with tab2:
        render_comparison_tab()


if __name__ == "__main__":
    main()
