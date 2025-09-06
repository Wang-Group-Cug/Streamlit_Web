import streamlit as st
import sys
from pathlib import Path

# Add project root to the path
sys.path.append(str(Path(__file__).parent.parent))
from SPINN_content import H2, H3_simultaneous, H3_separate

st.header('This is a program that uses soil texture to predict soil hydraulic parameters.')
st.markdown('For more details, please refer to <Soil Physics Informed Neural Networks to Estimate Bimodal Soil Hydraulic Properties>.')
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


