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
import plotly.graph_objects as go

from api_client import (
    APIClient, 
    format_confidence, 
    validate_image, 
    get_disease_info,
    get_gradcam_interpretation,
    check_image_quality,
    get_severity_score,
    generate_pdf_report
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

st.markdown(get_custom_css(), unsafe_allow_html=True)

# Custom CSS for larger fonts
st.markdown("""
<style>
    .section-header {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
        color: #0891b2 !important;
    }
    .interpretation-text {
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
        padding: 1rem !important;
        background: rgba(8, 145, 178, 0.05) !important;
        border-radius: 8px !important;
    }
    .quality-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .quality-good { background: #10b981; color: white; }
    .quality-warn { background: #f59e0b; color: white; }
    .quality-bad { background: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)


# =====================================================
# INITIALIZE STATE
# =====================================================
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient(base_url="http://127.0.0.1:8001")

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### Settings")
    
    is_healthy, health_data = st.session_state.api_client.health_check()
    if is_healthy:
        st.success("API Online")
    else:
        st.error("API Offline")
    
    st.divider()
    
    st.markdown("### Analysis Type")
    disease_type = st.radio(
        "Select:",
        options=["brain_mri", "pneumonia", "retina"],
        format_func=lambda x: {"brain_mri": "Brain MRI", "pneumonia": "Chest X-Ray", "retina": "Retinal Scan"}[x],
        label_visibility="collapsed"
    )
    
    st.divider()
    
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
    st.warning("Cannot connect to FastAPI backend.")
    st.code("uvicorn src.api.main:app --port 8001", language="bash")
    st.stop()


# =====================================================
# SINGLE IMAGE MODE
# =====================================================
if not batch_mode:
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
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            st.session_state.uploaded_image_bytes = image_bytes
            
            # IMAGE QUALITY CHECK
            quality = check_image_quality(image_bytes)
            
            with st.expander("Image Quality Check", expanded=len(quality.get("issues", [])) > 0):
                q_col1, q_col2, q_col3 = st.columns(3)
                with q_col1:
                    st.metric("Resolution", f"{quality['resolution'][0]}x{quality['resolution'][1]}")
                with q_col2:
                    st.metric("File Size", f"{quality['file_size_mb']} MB")
                with q_col3:
                    if quality['blur_score'] is not None:
                        st.metric("Sharpness", f"{quality['blur_score']:.0f}")
                    else:
                        st.metric("Sharpness", "N/A")
                
                if quality["issues"]:
                    for issue in quality["issues"]:
                        if "too low" in issue or "exceeds" in issue or "blurry" in issue.lower():
                            st.warning(issue)
                        else:
                            st.info(issue)
                else:
                    st.success("Image quality looks good!")
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
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
    # RESULTS SECTION
    # =====================================================
    result = st.session_state.prediction_result
    
    if result:
        confidence = result.get("confidence", 0)
        predicted_class = result.get("predicted_class", "N/A")
        disease_names = {"brain_mri": "Brain MRI", "pneumonia": "Chest X-Ray", "retina": "Retinal"}
        
        # Get severity info
        disease_info = get_disease_info(disease_type)
        pred_key = predicted_class.lower().replace(" ", "_")
        medical_details = disease_info.get("medical_details", {})
        class_info = None
        for key, value in medical_details.items():
            if key.lower().replace("_", "") == pred_key.replace("_", ""):
                class_info = value
                break
        severity = class_info.get("severity", "Unknown") if class_info else "Unknown"
        severity_score = get_severity_score(severity)
        
        # -------------------------------------------------
        # SECTION 1: ANALYSIS RESULTS
        # -------------------------------------------------
        st.divider()
        st.markdown('<p class="section-header">1. Analysis Results</p>', unsafe_allow_html=True)
        
        if confidence < 0.5:
            st.warning("Low Confidence - Manual review recommended")
        
        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Condition", predicted_class)
        m2.metric("Confidence", format_confidence(confidence))
        m3.metric("Analysis Type", disease_names.get(disease_type, disease_type))
        
        # Probability table
        probs = result.get("probabilities", {})
        if probs:
            prob_data = [{"Class": k, "Probability": f"{v*100:.1f}%"} for k, v in sorted(probs.items(), key=lambda x: -x[1])]
            st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)
        
        # -------------------------------------------------
        # SEVERITY GAUGE
        # -------------------------------------------------
        st.markdown("**Severity Gauge**")
        
        gauge_colors = [(0, "#10b981"), (0.35, "#f59e0b"), (0.65, "#ef4444"), (1, "#dc2626")]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=severity_score,
            title={'text': f"Severity: {severity}"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#0891b2"},
                'steps': [
                    {'range': [0, 25], 'color': "#d1fae5"},
                    {'range': [25, 50], 'color': "#fef3c7"},
                    {'range': [50, 75], 'color': "#fed7aa"},
                    {'range': [75, 100], 'color': "#fecaca"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.8,
                    'value': severity_score
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # -------------------------------------------------
        # SECTION 2: AI INTERPRETATION
        # -------------------------------------------------
        st.divider()
        st.markdown('<p class="section-header">2. AI Interpretation</p>', unsafe_allow_html=True)
        
        if class_info:
            interpretation = f"""
<div class="interpretation-text">
<p><strong>Detection:</strong> The AI identified patterns consistent with <strong>{predicted_class}</strong> ({format_confidence(confidence)} confidence).</p>

<p><strong>What is {predicted_class}?</strong><br>{class_info.get('description', 'N/A')}</p>

<p><strong>Severity:</strong> <span style="color: {'#ef4444' if severity_score >= 65 else '#f59e0b' if severity_score >= 35 else '#10b981'}; font-weight: bold;">{severity}</span></p>

<p><strong>Recommendation:</strong><br>{class_info.get('recommendation', 'Consult a healthcare professional.')}</p>
</div>
"""
            st.markdown(interpretation, unsafe_allow_html=True)
        
        # -------------------------------------------------
        # SECTION 3: ENSEMBLE MODEL COMPARISON (Retina only)
        # -------------------------------------------------
        if disease_type == "retina" and result.get("individual_predictions"):
            st.divider()
            st.markdown('<p class="section-header">3. Ensemble Model Comparison</p>', unsafe_allow_html=True)
            
            st.caption("The retina analysis uses an ensemble of 3 EfficientNet models")
            
            ind_preds = result["individual_predictions"]
            if ind_preds:
                model_data = []
                for model_name, pred_data in ind_preds.items():
                    if isinstance(pred_data, dict):
                        model_data.append({
                            "Model": model_name.replace("efficientnet", "EfficientNet").upper(),
                            "Prediction": pred_data.get("predicted_class", "N/A"),
                            "Confidence": f"{pred_data.get('confidence', 0)*100:.1f}%"
                        })
                
                if model_data:
                    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
            else:
                st.info("Individual model predictions not available")
        
        # -------------------------------------------------
        # SECTION 4: GRAD-CAM VISUALIZATION (Comparison)
        # -------------------------------------------------
        if generate_gradcam:
            st.divider()
            section_num = "4" if disease_type == "retina" else "3"
            st.markdown(f'<p class="section-header">{section_num}. Grad-CAM Visualization</p>', unsafe_allow_html=True)
            
            # Side-by-side comparison
            gradcam_url = result.get("gradcam_url")
            comparison_url = result.get("gradcam_comparison_url")
            
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**Original Image**")
                if st.session_state.uploaded_image_bytes:
                    st.image(st.session_state.uploaded_image_bytes, width=350)
            
            with img_col2:
                st.markdown("**Grad-CAM Overlay**")
                if gradcam_url and gradcam_url.startswith("/static/"):
                    local_path = gradcam_url.replace("/static/", "outputs/")
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    full_path = os.path.join(project_root, local_path.replace("/", os.sep))
                    
                    if os.path.exists(full_path):
                        st.image(full_path, width=350)
                    else:
                        gradcam_bytes = st.session_state.api_client.get_gradcam_image(gradcam_url)
                        if gradcam_bytes:
                            st.image(gradcam_bytes, width=350)
                        else:
                            st.info("Grad-CAM not available")
                else:
                    st.info("Grad-CAM not generated")
            
            # -------------------------------------------------
            # SECTION 5: GRAD-CAM INTERPRETATION
            # -------------------------------------------------
            st.divider()
            section_num = "5" if disease_type == "retina" else "4"
            st.markdown(f'<p class="section-header">{section_num}. How to Read Grad-CAM</p>', unsafe_allow_html=True)
            
            gradcam_guide = get_gradcam_interpretation()
            guide_html = f"""
<div class="interpretation-text">
<p><strong>What is Grad-CAM?</strong><br>{gradcam_guide['description']}</p>
<p><strong>Color Legend:</strong></p>
<ul>
"""
            for color, meaning in gradcam_guide['colors'].items():
                guide_html += f"<li><strong>{color}:</strong> {meaning}</li>"
            guide_html += f"""
</ul>
<p><strong>Clinical Note:</strong> {gradcam_guide['clinical_note']}</p>
<p style="color: #6b7280; font-style: italic;">{gradcam_guide['disclaimer']}</p>
</div>
"""
            st.markdown(guide_html, unsafe_allow_html=True)
        
        # -------------------------------------------------
        # PDF EXPORT
        # -------------------------------------------------
        st.divider()
        st.markdown("### Export Report")
        
        if st.button("Generate PDF Report", type="secondary"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf_bytes = generate_pdf_report(result, disease_type)
                    st.download_button(
                        "Download PDF",
                        data=pdf_bytes,
                        file_name="medical_analysis_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
    
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
            progress = st.progress(0)
            status = st.empty()
            
            images = []
            for f in uploaded_files:
                f.seek(0)
                images.append((f.read(), f.name))
            
            def update(c, t):
                progress.progress(c / t)
                status.text(f"Processing {c}/{t}...")
            
            results = st.session_state.api_client.predict_batch(images, disease_type, update)
            st.session_state.batch_results = results
            status.text("Complete!")
    
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        
        successful = sum(1 for r in results if r.get("success"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(results))
        c2.metric("Successful", successful)
        c3.metric("Failed", len(results) - successful)
        
        table_data = []
        for r in results:
            if r.get("success"):
                table_data.append({
                    "Filename": r.get("filename"),
                    "Prediction": r.get("predicted_class"),
                    "Confidence": format_confidence(r.get("confidence", 0))
                })
            else:
                table_data.append({
                    "Filename": r.get("filename"),
                    "Prediction": "-",
                    "Confidence": "-"
                })
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        
        csv = pd.DataFrame(table_data).to_csv(index=False)
        st.download_button("Download CSV", csv, "predictions.csv", "text/csv")


# =====================================================
# FOOTER
# =====================================================
st.markdown(render_footer(), unsafe_allow_html=True)
