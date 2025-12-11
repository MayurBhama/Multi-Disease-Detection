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
    get_disease_info
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
        st.markdown(f"**Classes:** {', '.join(info['classes'])}")
    
    st.divider()
    
    # Options
    st.markdown("### Options")
    generate_gradcam = st.toggle("Generate Grad-CAM", value=True, help="Visual explanation of model attention")
    batch_mode = st.toggle("Batch Mode", value=False, help="Upload and analyze multiple images")
    
    st.divider()
    
    # API URL config
    with st.expander("Advanced Settings"):
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
    st.warning("Cannot connect to the FastAPI backend. Please ensure the server is running.")
    st.code("uvicorn src.api.main:app --port 8001", language="bash")
    st.stop()


# =====================================================
# SINGLE IMAGE MODE
# =====================================================
if not batch_mode:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["png", "jpg", "jpeg", "bmp"],
            help="Supported formats: PNG, JPG, JPEG, BMP (max 10MB)"
        )
        
        if uploaded_file:
            # Validate
            is_valid, error_msg = validate_image(uploaded_file)
            if not is_valid:
                st.error(error_msg)
            else:
                # Show preview (smaller size)
                st.image(uploaded_file, caption="Uploaded Image", width=300)
                
                # Analyze button
                if st.button("Analyze Image", type="primary", use_container_width=True):
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
                            st.error(result.get('error', 'Prediction failed'))
    
    with col2:
        st.markdown("### Results")
        
        result = st.session_state.prediction_result
        
        if result:
            # Confidence level
            confidence = result.get("confidence", 0)
            
            # Low confidence warning
            if confidence < 0.5:
                st.warning("Low Confidence Prediction - Consider manual review")
            
            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted Class", result.get("predicted_class", "N/A"))
            with col_b:
                st.metric("Confidence", format_confidence(confidence))
            
            st.divider()
            
            # Probability table using native Streamlit
            st.markdown("**Class Probabilities**")
            probs = result.get("probabilities", {})
            
            # Create dataframe for probabilities
            prob_data = []
            for class_name, prob in sorted(probs.items(), key=lambda x: -x[1]):
                prob_data.append({
                    "Class": class_name,
                    "Probability": f"{prob * 100:.1f}%",
                    "Score": prob
                })
            
            if prob_data:
                prob_df = pd.DataFrame(prob_data)
                # Show as bar chart
                st.bar_chart(prob_df.set_index("Class")["Score"])
                # Show as table
                st.dataframe(
                    prob_df[["Class", "Probability"]], 
                    use_container_width=True,
                    hide_index=True
                )
            
            # Grad-CAM
            if generate_gradcam and result.get("gradcam_url"):
                st.divider()
                st.markdown("**Grad-CAM Visualization**")
                
                gradcam_url = result["gradcam_url"]
                
                # Try local file first
                if gradcam_url.startswith("/static/"):
                    # Convert /static/gradcam/... to outputs/gradcam/...
                    local_path = gradcam_url.replace("/static/", "outputs/")
                    # Get project root
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    full_path = os.path.join(project_root, local_path.replace("/", os.sep))
                    
                    if os.path.exists(full_path):
                        st.image(full_path, caption="Model Attention Heatmap", width=400)
                    else:
                        # Try API
                        gradcam_bytes = st.session_state.api_client.get_gradcam_image(gradcam_url)
                        if gradcam_bytes:
                            st.image(gradcam_bytes, caption="Model Attention Heatmap", width=400)
                        else:
                            st.info(f"Grad-CAM image not found at: {full_path}")
                else:
                    st.info("Grad-CAM URL format not recognized")
            elif generate_gradcam:
                st.info("Grad-CAM not generated. Check API response.")
        else:
            st.info("Upload an image and click 'Analyze' to see results")


# =====================================================
# BATCH MODE
# =====================================================
else:
    st.markdown("### Batch Prediction")
    st.caption("Upload multiple images for batch analysis")
    
    uploaded_files = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")
        
        if st.button("Analyze All", type="primary"):
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
            status_text.text("Complete!")
    
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
                    "Status": "Success"
                })
            else:
                table_data.append({
                    "Filename": r.get("filename", "N/A"),
                    "Predicted Class": "-",
                    "Confidence": "-",
                    "Status": f"Failed: {r.get('error', 'Unknown')[:30]}"
                })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Results (CSV)",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )


# =====================================================
# FOOTER
# =====================================================
st.markdown(render_footer(), unsafe_allow_html=True)
