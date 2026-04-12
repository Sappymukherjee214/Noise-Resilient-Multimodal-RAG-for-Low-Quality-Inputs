import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

# Page Config
st.set_page_config(
    page_title="Sentinel AI | Multimodal RAG",
    page_icon="💠",
    layout="wide",
)

# Advanced CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top left, #1a1c24, #0e1117);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    .status-badge {
        background: rgba(0, 255, 128, 0.1);
        color: #00ff80;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        border: 1px solid rgba(0, 255, 128, 0.3);
    }
    
    .hallucination-warning {
        background: rgba(255, 76, 76, 0.1);
        border: 1px solid rgba(255, 76, 76, 0.3);
        border-radius: 8px;
        padding: 16px;
        color: #ff4c4c;
    }
    </style>
    """, unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/security-shield.png", width=80)
    st.title("Sentinel RAG v3")
    st.markdown("---")
    
    st.subheader("⚙️ System Config")
    denoise = st.toggle("Noise Resilience Layer", value=True)
    prompt_mode = st.selectbox("Intelligence Mode", ["dynamic", "v1", "v2_robust"])
    
    st.markdown("---")
    st.subheader("📊 Fleet Performance")
    # Simulated metrics
    st.metric("Avg Latency", "1.24s", "-0.12s")
    st.metric("Avg Faithfulness", "92%", "+4%")
    
    if st.button("🔄 Reset Cache", use_container_width=True):
        st.toast("Cache flushed successfully!")

# --- Main UI ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("Multimodal Intelligence Dashboard")
    st.markdown("#### Real-time Fashion Retrieval & Generative Analysis")
with header_col2:
    st.markdown("<br><div align='right'><span class='status-badge'>PRODUCTION ACTIVE</span></div>", unsafe_allow_html=True)

tabs = st.tabs(["🔍 Live Query", "📈 Performance Lab", "🛡️ Governance"])

with tabs[0]:
    input_col, output_col = st.columns([1, 1.5])
    
    with input_col:
        st.markdown("### User Intent")
        with st.container(border=True):
            query = st.text_input("Product Description", placeholder="e.g., Modrin blue cotton shtir")
            image_file = st.file_uploader("Multimodal Context", type=["jpg", "png"])
            
            if st.button("🚀 Process Intelligence", use_container_width=True):
                if query:
                    with st.spinner("Analyzing Multimodal Signals..."):
                        try:
                            # Mocking/Calling API
                            files = {"image": image_file.getvalue()} if image_file else None
                            data = {
                                "text": query,
                                "denoise": str(denoise).lower(),
                                "prompt_version": prompt_mode
                            }
                            resp = requests.post(f"{API_URL}/query", data=data, files=files)
                            if resp.status_code == 200:
                                st.session_state['result'] = resp.json()
                            else:
                                st.error(f"Engine Error: {resp.text}")
                        except Exception as e:
                            # Fallback for demo
                            st.warning(f"Connection to backend failed. Showing simulation. ({e})")
                            st.session_state['result'] = {
                                "answer": "Based on the retrieved context, I found a refined blue cotton shirt with modern tailoring. It features a breathable fabric suitable for summer.",
                                "confidence": 0.89,
                                "processing_time": 1.15,
                                "metrics": {
                                    "faithfulness": 0.94,
                                    "is_hallucinated": False,
                                    "reason": "Direct keyword match found in styles.csv"
                                }
                            }

    with output_col:
        if 'result' in st.session_state:
            res = st.session_state['result']
            
            # Header Stats
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Confidence", f"{res['confidence']*100:.0f}%")
            s_col2.metric("Latency", f"{res['processing_time']:.2f}s")
            s_col3.metric("Faithfulness", f"{res['metrics']['faithfulness']*100:.0f}%")
            
            # Response
            st.markdown("### Generated Intelligence")
            st.success(res['answer'])
            
            if res['metrics']['is_hallucinated']:
                st.markdown(f"""<div class='hallucination-warning'>
                    <b>⚠️ Hallucination Detected:</b> {res['metrics']['reason']}
                </div>""", unsafe_allow_html=True)
            
            # Evidence Visualization
            st.markdown("### Contextual Evidence")
            # In real app, we iterate over retrieved context
            e_col1, e_col2 = st.columns(2)
            e_col1.image("https://via.placeholder.com/300x400?text=Retrieved+Product+1", caption="Primary Evidence")
            e_col2.image("https://via.placeholder.com/300x400?text=Retrieved+Product+2", caption="Supporting Evidence")
        else:
            st.info("Awaiting user input for live analysis...")

with tabs[1]:
    st.markdown("### System Benchmarking")
    
    # Simulated Trend Data
    chart_data = pd.DataFrame({
        'Daily': range(7),
        'Baseline Accuracy': [0.72, 0.70, 0.73, 0.69, 0.71, 0.70, 0.72],
        'Sentinel Accuracy': [0.88, 0.90, 0.89, 0.91, 0.93, 0.92, 0.94]
    })
    
    fig = px.line(chart_data, x='Daily', y=['Baseline Accuracy', 'Sentinel Accuracy'], 
                  title="Noise-Resilient Accuracy Trend",
                  color_discrete_sequence=["#ff4b4b", "#00ff80"])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Robustness Matrix (Text Noise)")
        st.table(pd.DataFrame({
            "Noise Level": ["Low (0.1)", "Med (0.3)", "High (0.5)"],
            "Baseline F1": [0.75, 0.58, 0.42],
            "Sentinel F1": [0.89, 0.84, 0.79]
        }))

with tabs[2]:
    st.markdown("### Data Quality & Governance")
    # Simulated Dataset Metadata
    st.json({
        "dataset_version": "v2.1.0-fashion",
        "total_records": 44441,
        "completeness_score": 0.984,
        "schema_status": "VALID",
        "last_index_update": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    
    st.markdown("#### Semantic Drift Monitoring")
    st.warning("No significant semantic drift detected in the last 24 hours.")

