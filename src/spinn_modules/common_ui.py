import streamlit as st
import pandas as pd
import numpy as np
import importlib
from pathlib import Path

def render_spinn_page(model_type, header, description, needs_bd=False):
    """
    Renders the SPINN PTF page dynamically.
    
    Args:
        model_type (str): The model type identifier (e.g., 'H2', 'H3_separate', 'H3_simultaneous').
        header (str): The subheader text to display.
        description (str): Additional description text.
        needs_bd (bool): Whether Bulk Density input is required.
    """
    
    # Dynamic Import
    try:
        models_pkg = f"src.spinn_models.{model_type}"
        VGM_PTF = getattr(importlib.import_module(f"{models_pkg}.VGM_PTF"), "VGM_PTF")
        M3_PTF = getattr(importlib.import_module(f"{models_pkg}.M3_PTF"), "M3_PTF")
        B_FXW_PTF = getattr(importlib.import_module(f"{models_pkg}.B_FXW_PTF"), "B_FXW_PTF")
    except ImportError as e:
        st.error(f"Model import failed for {model_type}: {str(e)}")
        st.stop()

    # UI Rendering
    st.subheader(header)
    if description:
        st.text(description)
        
    st.subheader(':blue[Single sample need to be predicted:]')
    st.text('Please enter Soil Texture' + (' and Bulk density.' if needs_bd else '.'))
    
    with st.form('Texture & BD'):
        sand = st.number_input('Sand×100%, (eg 0.41)')
        silt = st.number_input('Silt×100%, (eg 0.32)')
        clay = st.number_input('Clay×100%, (eg 0.27)')
        bd = 0.0
        if needs_bd:
            bd = st.number_input('Bulk Density(g/cm^3)')
            
        submitted = st.form_submit_button("PTF calculates hydraulic parameter")

    if submitted:
        input_valid = False
        if 0.99 <= (sand + silt + clay) <= 1.01:
            input_valid = True
            if needs_bd:
                # Basic check for BD if needed, though mostly just texture sum is critical
                pass
        
        if input_valid:
            input_data = [[sand, silt, clay, bd]] if needs_bd else [[sand, silt, clay]]
            
            # VGM
            [VGM_Para, nan] = VGM_PTF(input_data)
            VGM_Para = np.round(VGM_Para, 4)
            st.write("VGM Model Parameters:", 
                     "alpha=", VGM_Para[0], 'n=', VGM_Para[1], 
                     'θr=', VGM_Para[2], 'θs=', VGM_Para[3], 'Ks=', VGM_Para[4]) 
            
            # M3
            [M3_Para, nan] = M3_PTF(input_data)
            M3_Para = np.round(M3_Para, 4)
            st.write("FXW_M3 Model Parameters:",
                     "alpha=", M3_Para[0], 'n=', M3_Para[1], 'm=', M3_Para[2],
                     'θs=', M3_Para[3], 'K(ha)=', M3_Para[4], 'Ks=', M3_Para[5]) 
            
            # B_FXW
            [B_FXW_Para, nan] = B_FXW_PTF(input_data)
            B_FXW_Para = np.round(B_FXW_Para, 4)
            st.write("B_FXW Model Parameters:",
                     "alpha=", B_FXW_Para[2], 'n=', B_FXW_Para[3], 'm=', B_FXW_Para[0],
                     'θs=', B_FXW_Para[1], 'nc=', B_FXW_Para[4], 
                     'K(ha)=', B_FXW_Para[5], 'Ks=', B_FXW_Para[6])       

        else:
            msg = 'input error (sand+silt+clay should equal to 1)'
            if needs_bd:
                msg += ' & The unit of bulk density is g/cm^3'
            st.write(msg)

    # File Upload Section
    st.markdown("---")
    st.subheader(':blue[Multiple samples need to be predicted:]')
    st.text('If a large number of soil need to be predicted, you can download the "Example.csv" and rewrite it. And then Browse and upload the file.')
    
    # Path to assets/texture1.csv
    # Assuming this file is at src/spinn_modules/common_ui.py
    # assets is at ../../assets
    csv_path = Path(__file__).resolve().parent.parent.parent / "assets" / "texture1.csv"
    
    if csv_path.exists():
        with open(csv_path, "rb") as file:
            st.download_button(
                label="Download Soil Texture Example.csv",
                data=file,
                file_name="texture1.csv",
                mime="text/csv",
            )
    else:
        st.error(f"Example file not found at {csv_path}")

    uploaded_file = st.file_uploader('Please Upload the .csv file', type='csv')

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        Texture = np.array(df)
        
        # H2 uses only first 3 columns (sand, silt, clay), H3 uses 4 (plus BD)
        # We need to ensure input dimensions match what the model expects
        if not needs_bd:
            Texture = Texture[:, :3]
        
        [nan, VGM_Para] = VGM_PTF(Texture)
        [nan, M3_Para] = M3_PTF(Texture)
        [nan, B_FXW_Para] = B_FXW_PTF(Texture)

        st.download_button(
            label="↓ Download VGM Parameter",
            data=VGM_Para.to_csv().encode("utf-8"),
            file_name='VGM_Parameter.csv',
            mime='text/csv'
        )

        st.download_button(
            label="↓ Download FXW-M3 Parameter",
            data=M3_Para.to_csv().encode("utf-8"),
            file_name='FXW_M3_Parameter.csv',
            mime='text/csv'
        )

        st.download_button(
            label="↓ Download B-FXW Parameter",
            data=B_FXW_Para.to_csv().encode("utf-8"),
            file_name='B_FXW_Parameter.csv',
            mime='text/csv'
        )
