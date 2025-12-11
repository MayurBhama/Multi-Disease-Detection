# web/app.py
"""
Multi-Disease Detection - Streamlit Frontend
=============================================
Professional medical image analysis dashboard.

Run with: streamlit run web/app.py
"""

import streamlit as st
import pandas as pd
import os

from api_client import (
    APIClient, 
    format_confidence, 
    validate_image, 
    get_disease_info,
    get_gradcam_interpretation
)
from styles import (
    get_custom_css, 
    render_header, 
    render_footer
)


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Multi-Disease Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Custom CSS for larger fonts
st.markdown("""
<style>
    .big-text {
        font-size: 1.2rem !important;
        line-height: 1.8 !important;
    }
    .section-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        color: #0891b2 !important;
    }
    .interpretation-text {
        font-size: 1.15rem !important;
        line-height: 1.9 !important;
        padding: 1rem !important;
        background: rgba(8, 145, 178, 0.05) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# INITIALIZE STATE
# =====================================================
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient(base_url="http://127.0.0.1:8001")

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### Settings")
    
    # API Status
    is_healthy, health_data = st.session_state.api_client.health_check()
    if is_healthy:
        st.success("API Online")
    else:
        st.error("API Offline")
        st.caption(health_data.get("error", "Cannot connect"))
    
    st.divider()
    
    # Disease Type Selector
    st.markdown("### Analysis Type")
    disease_type = st.radio(
        "Select analysis type:",
        options=["brain_mri", "pneumonia", "retina"],
        format_func=lambda x: {
            "brain_mri": "Brain MRI",
            "pneumonia": "Chest X-Ray",
            "retina": "Retinal Scan"
        }[x],
        label_visibility="collapsed"
    )
    
    # Show disease info
    info = get_disease_info(disease_type)
    if info:
        st.caption(info["description"])
    
    st.divider()
    
    # Options
    st.markdown("### Options")
    generate_gradcam = st.toggle("Generate Grad-CAM", value=True)
    batch_mode = st.toggle("Batch Mode", value=False)
    
    st.divider()
    
    with st.expander("Advanced"):
        api_url = st.text_input("API URL", value="http://127.0.0.1:8001")
        if st.button("Update"):
            st.session_state.api_client = APIClient(base_url=api_url)
            st.rerun()


# =====================================================
# MAIN CONTENT
# =====================================================
st.markdown(render_header(), unsafe_allow_html=True)

if not is_healthy:
    st.warning("Cannot connect to FastAPI backend. Ensure server is running.")
    st.code("uvicorn src.api.main:app --port 8001", language="bash")
    st.stop()


# =====================================================
# SINGLE IMAGE MODE - STACKED LAYOUT
# =====================================================
if not batch_mode:
    # Upload Section
    st.markdown("### Upload Image")
    
    col_upload, col_preview = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["png", "jpg", "jpeg", "bmp"]
        )
    with col_preview:
        if uploaded_file:
            st.image(uploaded_file, width=120)
    
    if uploaded_file:
        is_valid, error_msg = validate_image(uploaded_file)
        if not is_valid:
            st.error(error_msg)
        else:
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    success, result = st.session_state.api_client.predict(
                        image_bytes=image_bytes,
                        filename=uploaded_file.name,
                        disease_type=disease_type,
                        generate_gradcam=generate_gradcam
                    )
                    
                    if success:
                        st.session_state.prediction_result = result
                    else:
                        st.error(result.get('error', 'Prediction failed'))
    
    # =====================================================
    # RESULTS - STACKED SECTIONS
    # =====================================================
    result = st.session_state.prediction_result
    
    if result:
        confidence = result.get("confidence", 0)
        predicted_class = result.get("predicted_class", "N/A")
        disease_names = {"brain_mri": "Brain MRI", "pneumonia": "Chest X-Ray", "retina": "Retinal"}
        
        # -------------------------------------------------
        # SECTION 1: ANALYSIS RESULTS
        # -------------------------------------------------
        st.divider()
        st.markdown('<p class="section-header">1. Analysis Results</p>', unsafe_allow_html=True)
        
        if confidence < 0.5:
            st.warning("Low Confidence - Manual review recommended")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Condition", predicted_class)
        col2.metric("Confidence Score", format_confidence(confidence))
        col3.metric("Analysis Type", disease_names.get(disease_type, disease_type))
        
        # Probability table
        probs = result.get("probabilities", {})
        if probs:
            st.markdown("**Probability Distribution:**")
            prob_data = [{"Class": k, "Probability": f"{v*100:.1f}%"} for k, v in sorted(probs.items(), key=lambda x: -x[1])]
            st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)
        
        # -------------------------------------------------
        # SECTION 2: AI INTERPRETATION
        # -------------------------------------------------
        st.divider()
        st.markdown('<p class="section-header">2. AI Interpretation</p>', unsafe_allow_html=True)
        
        disease_info = get_disease_info(disease_type)
        pred_key = predicted_class.lower().replace(" ", "_")
        medical_details = disease_info.get("medical_details", {})
        
        class_info = None
        for key, value in medical_details.items():
            if key.lower().replace("_", "") == pred_key.replace("_", ""):
                class_info = value
                break
        
        if class_info:
            interpretation = f"""
<div class="interpretation-text">
<p><strong>Detection Summary:</strong><br>
The AI model analyzed this {disease_names.get(disease_type, 'medical')} image and identified patterns consistent with <strong>{predicted_class}</strong> with <strong>{format_confidence(confidence)}</strong> confidence.</p>

<p><strong>What is {predicted_class}?</strong><br>
{class_info.get('description', 'Information not available.')}</p>

<p><strong>Severity Assessment:</strong> <span style="color: {'#ef4444' if 'High' in class_info.get('severity', '') or 'Critical' in class_info.get('severity', '') else '#f59e0b' if 'Moderate' in class_info.get('severity', '') else '#10b981'}; font-weight: bold;">{class_info.get('severity', 'Unknown')}</span><br>
{class_info.get('prevalence', '')}</p>

<p><strong>Recommended Next Steps:</strong><br>
{class_info.get('recommendation', 'Please consult a qualified healthcare professional for proper evaluation and diagnosis.')}</p>
</div>
"""
            st.markdown(interpretation, unsafe_allow_html=True)
        else:
            st.info(f"Detailed interpretation for '{predicted_class}' is not available.")
        
        # -------------------------------------------------
        # SECTION 3: GRAD-CAM VISUALIZATION
        # -------------------------------------------------
        if generate_gradcam and result.get("gradcam_url"):
            st.divider()
            st.markdown('<p class="section-header">3. Grad-CAM Visualization</p>', unsafe_allow_html=True)
            
            gradcam_url = result["gradcam_url"]
            
            if gradcam_url.startswith("/static/"):
                local_path = gradcam_url.replace("/static/", "outputs/")
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                full_path = os.path.join(project_root, local_path.replace("/", os.sep))
                
                col_img, col_space = st.columns([2, 1])
                with col_img:
                    if os.path.exists(full_path):
                        st.image(full_path, caption="Model Attention Heatmap", width=500)
                    else:
                        gradcam_bytes = st.session_state.api_client.get_gradcam_image(gradcam_url)
                        if gradcam_bytes:
                            st.image(gradcam_bytes, caption="Model Attention Heatmap", width=500)
                        else:
                            st.info("Grad-CAM image not available")
            
            # -------------------------------------------------
            # SECTION 4: GRAD-CAM INTERPRETATION
            # -------------------------------------------------
            st.divider()
            st.markdown('<p class="section-header">4. How to Read Grad-CAM</p>', unsafe_allow_html=True)
            
            gradcam_guide = get_gradcam_interpretation()
            
            guide_html = f"""
<div class="interpretation-text">
<p><strong>What is Grad-CAM?</strong><br>
{gradcam_guide['description']}</p>

<p><strong>Color Interpretation:</strong></p>
<ul style="font-size: 1.1rem; line-height: 1.8;">
"""
            for color, meaning in gradcam_guide['colors'].items():
                guide_html += f"<li><strong>{color}:</strong> {meaning}</li>"
            
            guide_html += f"""
</ul>

<p><strong>Clinical Significance:</strong><br>
{gradcam_guide['clinical_note']}</p>

<p style="color: #6b7280; font-style: italic;">{gradcam_guide['disclaimer']}</p>
</div>
"""
            st.markdown(guide_html, unsafe_allow_html=True)
        
        elif generate_gradcam:
            st.divider()
            st.markdown('<p class="section-header">3. Grad-CAM Visualization</p>', unsafe_allow_html=True)
            st.info("Grad-CAM was not generated. Try re-analyzing the image.")
    
    else:
        st.info("Upload an image and click 'Analyze' to see results")


# =====================================================
# BATCH MODE
# =====================================================
else:
    st.markdown("### Batch Prediction")
    
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")
        
        if st.button("Analyze All", type="primary"):
            progress_bar = st.progress(0)
            status = st.empty()
            
            images = [(f.read(), f.name) for f in uploaded_files]
            for f in uploaded_files:
                f.seek(0)
            images = [(f.read(), f.name) for f in uploaded_files]
            
            def update(c, t):
                progress_bar.progress(c / t)
                status.text(f"Processing {c}/{t}...")
            
            results = st.session_state.api_client.predict_batch(images, disease_type, update)
            st.session_state.batch_results = results
            status.text("Complete!")
    
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(results))
        c2.metric("Successful", successful)
        c3.metric("Failed", failed)
        
        table_data = []
        for r in results:
            if r.get("success"):
                table_data.append({
                    "Filename": r.get("filename"),
                    "Prediction": r.get("predicted_class"),
                    "Confidence": format_confidence(r.get("confidence", 0)),
                    "Status": "Success"
                })
            else:
                table_data.append({
                    "Filename": r.get("filename"),
                    "Prediction": "-",
                    "Confidence": "-",
                    "Status": f"Failed: {r.get('error', '')[:25]}"
                })
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        
        csv = pd.DataFrame(table_data).to_csv(index=False)
        st.download_button("Download CSV", csv, "predictions.csv", "text/csv")


# =====================================================
# FOOTER
# =====================================================
st.markdown(render_footer(), unsafe_allow_html=True)
