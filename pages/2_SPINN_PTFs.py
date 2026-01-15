import streamlit as st
import sys
from pathlib import Path

# Add project root to the path
sys.path.append(str(Path(__file__).parent.parent))
from src.spinn_modules import H2, H3_simultaneous, H3_separate

st.header('SPINN-PTFs: Soil Physics-Informed Neural Networks')
st.markdown(
    """
    ### 📄 Paper Information
    **Title:** Soil Physics‐Informed Neural Networks to Estimate Bimodal Soil Hydraulic Properties  
    **Authors:** Jieliang Zhou, Yunquan Wang *, ...  
    **Journal:** *Water Resources Research*, 2025  
    **DOI:** [10.1029/2024WR039337](https://doi.org/10.1029/2024WR039337)
    """
)

# Link to the local PDF in assets folder
pdf_filename = "Water Resources Research - 2025 - Zhou - Soil Physics‐Informed Neural Networks to Estimate Bimodal Soil Hydraulic.pdf"
pdf_path = Path(__file__).parent.parent / "assets" / pdf_filename

if pdf_path.exists():
    with open(pdf_path, "rb") as f:
         st.download_button(
             label="📥 Download Full Paper (PDF)",
             data=f,
             file_name="SPINN_Paper_2025.pdf",
             mime="application/pdf"
         )

st.markdown(
    """
    ---
    
    ### 💡 Introduction
    This program allows you to utilize **Soil Physics-Informed Neural Networks (SPINN)** to predict soil hydraulic parameters. 
    SPINN embeds soil hydraulic models into the training process. The loss function consists of moisture content and hydraulic conductivity.
    We have provided the prediction parameters of three models, namely VGM, FXW-M3, and B-FXW.
    **What you can get:**
    By inputting basic soil properties, you will obtain parameters for:
    *   **VGM Model**: The standard unimodal model.
    *   **FXW-M3 & B-FXW Models**: Advanced bimodal models that describe the soil hydraulic properties over the full saturation range.

    ### 🔍 Model  Guide
    Please choose the model that best fits your available data and needs:

    | Model Name | Required Inputs | Characteristics |
    | :--- | :--- | :--- |
    | **H2** | **Texture** (Sand, Silt, Clay) | **Basic Model**. Suitable when Bulk Density data is **unavailable**. |
    | **H3-Separate** | **Texture + Bulk Density** | **Advanced (Step-wise)**. Optimizes Water Retention and Conductivity separately. |
    | **H3-Simultaneous** | **Texture + Bulk Density** | **Advanced (Joint)**. Optimizes Water Retention and Conductivity simultaneously for better consistency. |
    
    """
)
st.markdown("---")


st.markdown(
    '<p style="color:rgb(163,42,42); font-size:20px; font-weight:bold;">Here, we provide three PTFs for selection:</p>', 
    unsafe_allow_html=True
)
# Initialize state
if 'view' not in st.session_state:
    st.session_state.view = 'selection'

# --- Render content based on state ---
if st.session_state.view == 'selection':
    option = st.radio(
        'Please select a model:',
        ('H2', 'H3-separate', 'H3-simultaneous'),
        horizontal=True
    )
    if st.button("Confirm"):
        st.session_state.view = option # Set state to 'H2', 'H3-simu', or 'H3-SP'
        st.rerun() # Force an immediate rerun to show the new view

elif st.session_state.view == 'H2':
    if st.button("← Back to Selection(currently H2)"):
        st.session_state.view = 'selection'
        st.rerun()
    H2.render_page()

elif st.session_state.view == 'H3-separate':
    if st.button("← Back to Selection(currently H3-separate)"):
        st.session_state.view = 'selection'
        st.rerun()
    H3_separate.render_page()


elif st.session_state.view == 'H3-simultaneous':
    if st.button("← Back to Selection(currently H3-simultaneous)"):
        st.session_state.view = 'selection'
        st.rerun()
    H3_simultaneous.render_page()


