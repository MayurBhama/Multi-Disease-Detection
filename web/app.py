# web/app.py
"""
Multi-Disease Detection - Streamlit Frontend
=============================================
Professional medical image analysis dashboard.

Run with: streamlit run web/app.py
"""

import streamlit as st
import pandas as pd
from io import BytesIO
import base64

from api_client import (
    APIClient, 
    format_confidence, 
    validate_image, 
    get_disease_info
)
from styles import (
    get_custom_css, 
    render_header, 
    render_footer,
    render_probability_bar
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
    st.markdown("### ⚙️ Settings")
    
    # API Status
    is_healthy, health_data = st.session_state.api_client.health_check()
    if is_healthy:
        st.markdown('<span class="status-dot online"></span> API Online', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot offline"></span> API Offline', unsafe_allow_html=True)
        st.error(health_data.get("error", "Cannot connect"))
    
    st.divider()
    
    # Disease Type Selector
    st.markdown("### 🔬 Disease Type")
    disease_type = st.radio(
        "Select analysis type:",
        options=["brain_mri", "pneumonia", "retina"],
        format_func=lambda x: {
            "brain_mri": "🧠 Brain MRI",
            "pneumonia": "🫁 Chest X-Ray",
            "retina": "👁️ Retinal Scan"
        }[x],
        label_visibility="collapsed"
    )
    
    # Show disease info
    info = get_disease_info(disease_type)
    if info:
        st.caption(info["description"])
        st.markdown(f"**Classes:** {', '.join(info['classes'])}")
    
    st.divider()
    
    # Options
    st.markdown("### 🎛️ Options")
    generate_gradcam = st.toggle("Generate Grad-CAM", value=True, help="Visual explanation of model attention")
    batch_mode = st.toggle("Batch Mode", value=False, help="Upload and analyze multiple images")
    
    st.divider()
    
    # API URL config
    with st.expander("🔧 Advanced"):
        api_url = st.text_input("API URL", value="http://127.0.0.1:8001")
        if st.button("Update"):
            st.session_state.api_client = APIClient(base_url=api_url)
            st.rerun()


# =====================================================
# MAIN CONTENT
# =====================================================
# Header
st.markdown(render_header(), unsafe_allow_html=True)

# Check API status before showing upload
if not is_healthy:
    st.warning("⚠️ Cannot connect to the FastAPI backend. Please ensure the server is running.")
    st.code("uvicorn src.api.main:app --port 8001", language="bash")
    st.stop()


# =====================================================
# SINGLE IMAGE MODE
# =====================================================
if not batch_mode:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["png", "jpg", "jpeg", "bmp"],
            help="Supported formats: PNG, JPG, JPEG, BMP (max 10MB)"
        )
        
        if uploaded_file:
            # Validate
            is_valid, error_msg = validate_image(uploaded_file)
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                # Show preview
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                
                # Analyze button
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        # Get file bytes
                        uploaded_file.seek(0)
                        image_bytes = uploaded_file.read()
                        
                        # Call API
                        success, result = st.session_state.api_client.predict(
                            image_bytes=image_bytes,
                            filename=uploaded_file.name,
                            disease_type=disease_type,
                            generate_gradcam=generate_gradcam
                        )
                        
                        if success:
                            st.session_state.prediction_result = result
                        else:
                            st.error(f"❌ {result.get('error', 'Prediction failed')}")
    
    with col2:
        st.markdown("### 📊 Results")
        
        result = st.session_state.prediction_result
        
        if result:
            # Confidence level
            confidence = result.get("confidence", 0)
            if confidence >= 0.7:
                conf_level = "high"
            elif confidence >= 0.5:
                conf_level = "medium"
            else:
                conf_level = "low"
            
            # Low confidence warning
            if confidence < 0.5:
                st.warning("⚠️ **Low Confidence Prediction** - Consider manual review")
            
            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted Class", result.get("predicted_class", "N/A"))
            with col_b:
                st.metric("Confidence", format_confidence(confidence))
            
            st.divider()
            
            # Probability table
            st.markdown("**Class Probabilities**")
            probs = result.get("probabilities", {})
            predicted = result.get("predicted_class", "")
            
            prob_html = '<table class="prob-table"><tr><th>Class</th><th>Probability</th><th>Distribution</th></tr>'
            for class_name, prob in sorted(probs.items(), key=lambda x: -x[1]):
                prob_html += render_probability_bar(class_name, prob, class_name == predicted)
            prob_html += '</table>'
            st.markdown(prob_html, unsafe_allow_html=True)
            
            # Grad-CAM
            if generate_gradcam and result.get("gradcam_url"):
                st.divider()
                st.markdown("**🔥 Grad-CAM Visualization**")
                
                gradcam_bytes = st.session_state.api_client.get_gradcam_image(result["gradcam_url"])
                if gradcam_bytes:
                    st.image(gradcam_bytes, caption="Model Attention Heatmap", use_container_width=True)
                else:
                    st.info("Grad-CAM image not available")
        else:
            st.info("👈 Upload an image and click 'Analyze' to see results")


# =====================================================
# BATCH MODE
# =====================================================
else:
    st.markdown("### 📦 Batch Prediction")
    st.caption("Upload multiple images for batch analysis")
    
    uploaded_files = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) selected")
        
        if st.button("🔍 Analyze All", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare images
            images = []
            for f in uploaded_files:
                f.seek(0)
                images.append((f.read(), f.name))
            
            # Process
            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Processing {current}/{total}...")
            
            results = st.session_state.api_client.predict_batch(
                images=images,
                disease_type=disease_type,
                progress_callback=update_progress
            )
            
            st.session_state.batch_results = results
            status_text.text("✅ Complete!")
    
    # Show results
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        
        # Summary
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(results))
        col2.metric("Successful", successful)
        col3.metric("Failed", failed)
        
        st.divider()
        
        # Results table
        table_data = []
        for r in results:
            if r.get("success"):
                table_data.append({
                    "Filename": r.get("filename", "N/A"),
                    "Predicted Class": r.get("predicted_class", "N/A"),
                    "Confidence": format_confidence(r.get("confidence", 0)),
                    "Status": "✅"
                })
            else:
                table_data.append({
                    "Filename": r.get("filename", "N/A"),
                    "Predicted Class": "-",
                    "Confidence": "-",
                    "Status": f"❌ {r.get('error', 'Failed')[:30]}"
                })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )


# =====================================================
# FOOTER
# =====================================================
st.markdown(render_footer(), unsafe_allow_html=True)
