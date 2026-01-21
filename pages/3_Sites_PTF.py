import streamlit as st
from pathlib import Path
import sys
from SPTF_src.SPTF_forwardfunc import forward_func as SPTFfunc
import pandas as pd


st.title("Site-Specific Pedotransfer Functions (SPTFs)")
# st.markdown("*Integrating Deep Learning with Physics-Aware Soil Hydrological Modeling*")

st.markdown("---")

# 主要内容部分
st.markdown("""
**Pedotransfer functions (PTFs)** are widely used to estimate soil hydraulic parameters from basic soil properties, 
playing a critical role in parameterizing earth system models.  
However, traditional PTFs — often developed from limited soil samples — tend to introduce substantial 
uncertainty and variability when applied to field-scale hydrological simulations.
To overcome these challenges, we introduce **Site-Specific Pedotransfer Functions (SPTFs)**, 
a novel approach that integrates deep learning with physics-aware modeling of soil water movement.
SPTFs utilize time-series input data and directly optimize simulated 
soil moisture by coupling the 1‑D Richardson–Richards equation with observational records. 
This results in significantly improved accuracy and reliability under real-world field conditions.
Developed and validated using two years of soil moisture data from **1,181 sites** 
in the International Soil Moisture Network, SPTFs demonstrate strong performance 
in simulating soil water content.
""")




st.caption("Performance metrics on independent test set (n = 179 sites)")

st.markdown("---")

# 使用说明部分
st.subheader("📌 Usage Requirements")

st.warning("""
**Prior to use, please note the following input requirements:**
""")

input_details = """
**Core soil properties (required):**
- Sand, silt, clay content
- Bulk density

**Time-series data (required, minimum 60 days recommended):**
- Shallow soil moisture dynamics (**SM**)
- Precipitation (**P**, cm/day)
- Potential evapotranspiration (**PET**, cm/day)
- Leaf area index (**LAI**)
- binarized Land Surface Temperature (**bLST**) –  1 if LST > 0 °C, 0 if LST < 0 °C.
"""

st.markdown(input_details)

st.markdown("""
**Important:** Given that collecting all categories of time-series data may be difficult,  
             it is essential that the **SM** data strives to be as comprehensive as possible.
""")
###################################

st.markdown("---")
st.subheader("📚 Further Information")

st.markdown("""
For complete methodological details, validation results, and implementation guidelines, 
please refer to the published paper:
""")

st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50;'>
        <strong>Reference:</strong> Physics-Informed Neural Networks to Develop Site-Specific Pedotransfer Functions<br>
        <a href="https://doi.org/10.1029/2025WR041265" style='color: #0066cc;'>https://doi.org/10.1029/2025WR041265</a>
    </div>
    """,
    unsafe_allow_html=True
)
# File Upload Section
st.markdown("---")
st.subheader(':blue[**Paramater Prediction**]')
st.text('Please upload "XXX.csv" file including data for all predictors. ' )

csv_path = Path(__file__).resolve().parent.parent / "assets" / "Sptf_sample.csv"
if csv_path.exists():
    with open(csv_path, "rb") as file:
        st.download_button(
            label="Click here Download Upload Example.csv",
            data=file,
            file_name="upload_sample.csv",
            mime="text/csv",
        )
else:
    st.error(f"Example file not found at {csv_path}")


uploaded_file = st.file_uploader('Please Upload the .csv file', type='csv')
# uploaded_file = csv_path


if uploaded_file:
    # try:
        vgmpara,fxwpara = SPTFfunc(uploaded_file)
        st.text("VGM Parameters")
        df1 = pd.DataFrame(list(vgmpara.items()), columns=['Parameters','Value'])
        st.dataframe(df1, )
        df2 = pd.DataFrame(list(fxwpara.items()), columns=['Parameters','Value'])
        st.text("FXW-M3 Parameters")
        st.dataframe(df2,)
    # except:
    #     st.error("Please verify the file format of your uploaded file.")

