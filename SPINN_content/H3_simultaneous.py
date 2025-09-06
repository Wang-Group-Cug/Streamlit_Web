import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def render_page():
    try:
        from SPINN_model.H3_simultaneous.VGM_PTF import VGM_PTF
        from SPINN_model.H3_simultaneous.M3_PTF  import M3_PTF
        from SPINN_model.H3_simultaneous.B_FXW_PTF import B_FXW_PTF
    except ImportError as e:
        st.error(f"Model import failed: {str(e)}")
        st.stop()


    # @st.cache


    #   ''' Single samles '''     ''' Single samles '''           
    #st.subheader('Single sample need to be predicted hydraulic parameters:')
    #st.markdown(
    #    "<h2 style='color: blue;'>Single sample need to be predicted hydraulic parameters:</h2>", 
    #    unsafe_allow_html=True
    #)
    st.subheader('H3-PTFs: Need to know the soil texture and bulk density information (sand, silt, clay [×100%] & Bulk Density)')   
    st.text('H3-simultaneous: When training PTF, consider both SWRC and HCC simultaneously.')
    st.subheader(':blue[Single sample need to be predicted:]')
    st.text('Please enter Soil Texture and Bulk density.')
    with st.form('Texture & BD'):
        sand = st.number_input('Sand×100%, (eg 0.41)')
        silt = st.number_input('Silt×100%, (eg 0.32)')
        clay = st.number_input('Clay×100%, (eg 0.27)')
        bd    = st.number_input('Bulk Density(g/cm^3)')
        submitted=st.form_submit_button("PTF calculates hydraulic parameter")

    if submitted:
       if sand+silt+clay<=1.01 and sand+silt+clay>=0.99:
          

          [VGM_Para,nan]=VGM_PTF([[sand,silt,clay,bd]])
          VGM_Para = np.round(VGM_Para,4)
          st.write("VGM Model Parameters:","alpha=",VGM_Para[0],'n=',VGM_Para[1],'θr=',VGM_Para[2],'θs=',VGM_Para[3],'Ks=',VGM_Para[4]) 
          
          [M3_Para,nan]=M3_PTF([[sand,silt,clay,bd]])
          M3_Para = np.round(M3_Para,4)
          st.write("FXW_M3 Model Parameters:","alpha=",M3_Para[0],'n=',M3_Para[1],'m=',M3_Para[2],'θs=',M3_Para[3],'K(ha)=',M3_Para[4],'Ks=',M3_Para[5]) 

          [B_FXW_Para,nan]=B_FXW_PTF([[sand,silt,clay,bd]])
          B_FXW_Para = np.round(B_FXW_Para,4)
          st.write("B_FXW Model Parameters:","alpha=",B_FXW_Para[2],'n=',B_FXW_Para[3],'m=',B_FXW_Para[0],'θs=',B_FXW_Para[1],'nc=',B_FXW_Para[4],'K(ha)=',B_FXW_Para[5],'Ks=',B_FXW_Para[6])       

       else:
          st.write('input error (sand+silt+clay should equal to 1 & The unit of bulk density is g/cm^3)')

    ###### Upload input .csv
    st.markdown("---")
    st.subheader(':blue[Multiple samples need to be predicted:]')
    st.text('If a large number of soil need to be predicted, you can download the "Example.csv" and rewrite it. And then Browse and upload the file.')
    csv_path = Path(__file__).parent.parent / "texture1.csv"
    with open(csv_path, "rb") as file:
        btn = st.download_button(
            label="Download Soil Texture Example.csv",
            data=file,
            file_name="texture1.csv",
            mime="text/csv",
        )


    uploaded_file=st.file_uploader('Please Upload the .csv file',type='csv')

    if uploaded_file:

        Texture=np.array(pd.read_csv(uploaded_file))

        [nan,VGM_Para]=VGM_PTF(Texture);  [nan,M3_Para] =M3_PTF(Texture);  [nan,B_FXW_Para]=B_FXW_PTF(Texture)

        st.download_button(
        label="↓ Download VGM Parameter",
        data=VGM_Para.to_csv().encode("utf-8"),
        file_name='VGM_Parameter.csv',
        mime='text/csv')

        st.download_button(
        label="↓ Download FXW-M3 Parameter",
        data= M3_Para.to_csv().encode("utf-8"),
        file_name='FXW_M3_Parameter.csv',
        mime='text/csv')

        st.download_button(
        label="↓ Download B-FXW Parameter",
        data= B_FXW_Para.to_csv().encode("utf-8"),
        file_name='B_FXW_Parameter.csv',
        mime='text/csv')