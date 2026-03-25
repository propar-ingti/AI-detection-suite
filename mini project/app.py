import streamlit as st
import text
import image
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="AI Forensic Tool 2026", layout="centered")

st.title("🛡️ AI vs Human: Forensic Analysis")
st.markdown("Detect AI-generated text and images using DeepDetect-2025 technology.")

# Sidebar for Navigation
option = st.sidebar.selectbox("Select Analysis Type", ("Text Analysis", "Image Analysis"))

# --- TEXT ANALYSIS UI ---
if option == "Text Analysis":
    st.header("📝 Text Content Detection")
    user_text = st.text_area("Paste the text you want to analyze here:", height=200)
    
    if st.button("Analyze Text"):
        if user_text.strip():
            with st.spinner('Analyzing patterns...'):
                results = text.predict_text(user_text)
                burstiness = text.get_burstiness(user_text)
                
                # Display Results
                col1, col2, col3 = st.columns(3)
                col1.metric("Human Score", f"{results['Human']:.1%}")
                col2.metric("AI Score", f"{results['AI']:.1%}")
                col3.metric("Edited AI", f"{results['Edited']:.1%}")
                
                st.info(f"**Burstiness Score:** {burstiness} (Lower scores often indicate AI generation)")
        else:
            st.warning("Please enter some text first!")

# --- IMAGE ANALYSIS UI ---
elif option == "Image Analysis":
    st.header("🖼️ Image Authenticity Check")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Scan for AI Artifacts"):
            with st.spinner('Scanning pixels...'):
                # Save temp file to pass to engine
                temp_path = "temp_upload.png"
                img.save(temp_path)
                
                try:
                    ai_chance = image.predict_image(temp_path)
                    
                    # Visual Gauge/Result
                    if ai_chance > 50:
                        st.error(f"⚠️ High Probability of AI Generation: {ai_chance:.2f}%")
                    else:
                        st.success(f"✅ Likely Authentic/Human: {100 - ai_chance:.2f}% Real")
                    
                    st.progress(ai_chance / 100)
                except Exception as e:
                    st.error(f"Error: {e}. Make sure you ran the training script first!")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)