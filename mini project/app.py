import streamlit as st
import text
import image
from PIL import Image
import os
import time
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Forensic Tool 2026", 
    page_icon="🛡️", 
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4259; }
    div[data-testid="stExpander"] { border: 1px solid #3e4259; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ AI Forensic Analysis Engine")
st.caption("Utilizing DeepDetect-2026 Neural Patterns to identify machine-generated content.")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Settings")
    option = st.radio("Analysis Mode", ("📝 Text Content", "🖼️ Image Artifacts"))
    st.divider()
    st.info("System Status: Online 🟢")

# --- TEXT ANALYSIS UI ---
if option == "📝 Text Content":
    st.subheader("Text Fingerprinting")
    user_text = st.text_area("Paste content for forensic scanning:", height=250, placeholder="Enter at least 50 words...")
    
    col_btn, _ = st.columns([1, 3])
    if col_btn.button("🚀 Start Deep Scan"):
        if user_text.strip():
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            results = text.predict_text(user_text)
            verdict = max(results, key=results.get)
            
            st.divider()
            if verdict == "Human":
                st.balloons()
                st.success(f"### ✅ VERDICT: HUMAN AUTHENTIC ({results['Human']:.1%})")
            elif verdict == "AI":
                st.error(f"### ⚠️ VERDICT: AI GENERATED ({results['AI']:.1%})")
            else:
                st.warning(f"### 🤔 VERDICT: AI-ASSISTED / EDITED ({results['Edited']:.1%})")

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Human Confidence", f"{results['Human']:.1%}")
            m_col2.metric("AI Confidence", f"{results['AI']:.1%}")
            m_col3.metric("Neural Editing", f"{results['Edited']:.1%}")

            # --- FIXING THE NUMPY ERRORS HERE ---
            b_features = text.get_burstiness_features(user_text)
            
            # Ensure burstiness is a single float value (scalar)
            # We use .item() to pull the value out of the numpy array
            raw_burstiness = b_features[0]
            burstiness_val = float(np.mean(raw_burstiness)) 
            
            st.write("---")
            st.write(f"**Burstiness Score: `{burstiness_val:.4f}`**")
            
            # Use the scalar value for the progress bar
            st.progress(min(burstiness_val, 1.0), text="Rhythm Variation")
            
            with st.expander("🔬 View Linguistic Metadata"):
                st.json({
                    "Burstiness (CV)": round(b_features[0], 4),
                    "Avg Word Length": round(b_features[1], 2),
                    "Sentence Count": int(b_features[2])
                })
        else:
            st.warning("Please enter text to analyze.")

# --- IMAGE ANALYSIS UI ---
elif option == "🖼️ Image Artifacts":
    st.subheader("Visual Artifact Detection")
    uploaded_file = st.file_uploader("Upload image (PNG/JPG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(img, caption="Original Asset", use_container_width=True)
        
        with c2:
            if st.button("🔍 Scan Pixels"):
                with st.spinner('Deconstructing image layers...'):
                    temp_path = "temp_scan.png"
                    img.save(temp_path)
                    res = image.predict_image(temp_path)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    st.write(f"### {res['Verdict']}")
                    # Ensure probability is between 0.0 and 1.0 for progress bar
                    st.progress(res['Fake Probability'] / 100, text="AI Probability Indicator")
                    
                    if res['Fake Probability'] > 50:
                        st.error(f"High risk detected: {res['Fake Probability']}%")
                    else:
                        st.success(f"Low risk detected: {res['Real Probability']}% Real")
                    
                    st.metric("Neural Artifact Score", f"{res['Fake Probability']:.1f}")