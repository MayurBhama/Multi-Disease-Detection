# web/styles.py
"""
Custom CSS Styles for Streamlit Medical Dashboard
=================================================
Professional medical UI with teal/blue theme.
"""


def get_custom_css() -> str:
    """Return custom CSS for the Streamlit app."""
    return """
    <style>
    /* ===== Global Styles ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ===== Header ===== */
    .main-header {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* ===== Cards ===== */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%);
        border-left: 4px solid #14b8a6;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fef9c3 100%);
        border-left: 4px solid #f59e0b;
    }
    
    .error-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fef2f2 100%);
        border-left: 4px solid #ef4444;
    }
    
    /* ===== Metrics ===== */
    .metric-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        flex: 1;
        min-width: 150px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0891b2;
        margin-top: 0.25rem;
    }
    
    .metric-value.high {
        color: #10b981;
    }
    
    .metric-value.medium {
        color: #f59e0b;
    }
    
    .metric-value.low {
        color: #ef4444;
    }
    
    /* ===== Probability Table ===== */
    .prob-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    
    .prob-table th, .prob-table td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .prob-table th {
        background: #f9fafb;
        font-weight: 600;
        color: #374151;
    }
    
    .prob-bar {
        height: 8px;
        background: #e5e7eb;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #14b8a6 0%, #0891b2 100%);
        border-radius: 4px;
    }
    
    /* ===== Sidebar ===== */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    .sidebar-section {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-section h3 {
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ===== Upload Area ===== */
    .uploadedFile {
        border: 2px dashed #0891b2 !important;
        border-radius: 12px !important;
        background: #f0fdfa !important;
    }
    
    /* ===== Buttons ===== */
    .stButton > button {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(8, 145, 178, 0.4);
    }
    
    /* ===== Footer ===== */
    .footer {
        background: #f1f5f9;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 3rem;
        text-align: center;
        color: #64748b;
        font-size: 0.875rem;
    }
    
    .footer-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    /* ===== Disease Selector ===== */
    .disease-option {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .disease-option:hover {
        border-color: #0891b2;
        background: #f0fdfa;
    }
    
    .disease-option.selected {
        border-color: #0891b2;
        background: #ecfeff;
    }
    
    .disease-icon {
        font-size: 2rem;
    }
    
    /* ===== Responsive ===== */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.75rem;
        }
        
        .metric-container {
            flex-direction: column;
        }
        
        .metric-box {
            min-width: 100%;
        }
    }
    
    /* ===== Status Indicator ===== */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-dot.online {
        background: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
    }
    
    .status-dot.offline {
        background: #ef4444;
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.2);
    }
    </style>
    """


def render_header():
    """Return HTML for the main header."""
    return """
    <div class="main-header">
        <h1>🏥 Multi-Disease Detection System</h1>
        <p>AI-powered medical image analysis for Brain MRI, Chest X-Ray, and Retinal Scans</p>
    </div>
    """


def render_footer():
    """Return HTML for the footer disclaimer."""
    return """
    <div class="footer">
        <p class="footer-warning">⚠️ IMPORTANT DISCLAIMER</p>
        <p>This system is for <strong>research and educational purposes only</strong> and is not intended as a medical diagnosis tool.</p>
        <p>Always consult qualified healthcare professionals for medical advice and diagnosis.</p>
        <p style="margin-top: 1rem; color: #94a3b8;">© 2024 Multi-Disease Detection System | Built with Streamlit & FastAPI</p>
    </div>
    """


def render_metric(label: str, value: str, level: str = "normal") -> str:
    """Render a metric box."""
    level_class = {"high": "high", "medium": "medium", "low": "low"}.get(level, "")
    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value {level_class}">{value}</div>
    </div>
    """


def render_probability_bar(class_name: str, probability: float, is_predicted: bool = False) -> str:
    """Render a probability bar for a class."""
    pct = probability * 100
    highlight = "font-weight: 700; color: #0891b2;" if is_predicted else ""
    return f"""
    <tr>
        <td style="{highlight}">{class_name}</td>
        <td style="{highlight}">{pct:.1f}%</td>
        <td style="width: 50%;">
            <div class="prob-bar">
                <div class="prob-bar-fill" style="width: {pct}%;"></div>
            </div>
        </td>
    </tr>
    """
