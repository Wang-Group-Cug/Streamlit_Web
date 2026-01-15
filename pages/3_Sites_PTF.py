import streamlit as st
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import sys
# # current_path = Path(__file__).parent.absolute()
# # print(current_path)

st.title('Coming soon ')
# st.image('./Global_Map/Global_Parameter_Map_FXW-M3.jpg', caption='Global hydraulic parameter map estimated by FXW-M3 PTF (Zhou et al. 2025)')

# st.markdown('1. If you need globally distributed soil hydraulic parameters, please visit (www.***.com). Due to the large size of the maps, this page does not offer the tif file downloads.')
# st.markdown('2. If you only need soil hydraulic parameters for multiple latitude and longitude points, we recommend extracting soil texture from SoilGrid (Hengl et al. 2017) and predicting soil hydraulic parameters through PTFs. The extraction method is as follows:')
# st.text('Please Download the "Station Exeample.csv" and rewrite it.\nAnd then Browse and upload the file.')

# #  Give Link to Download Example.csv
# with open("station_point.csv") as file:
#     btn = st.download_button(
#         label="Download Station Example.csv",
#         data=file,
#         file_name="Station Example.csv",
#         mime="text/csv",
#     )

# #  Give Link to Browse and Upload csv file
# uploaded_file_map=st.file_uploader('Please Upload the csv file, and wait 10-20 minutes.',type='csv')


# if uploaded_file_map:
#     from Global_Map.Global_Lati_Longti_texture import Global_data 
#     station = pd.read_csv(uploaded_file_map)  # Read csv file
#     tex1 = Global_data(station)             # extract data
#     tex1= tex1.to_csv(index=False).encode("utf-8") # write to csv
#     st.download_button(
#     label="Download Soil Texture & Bulk density",
#     data=tex1,
#     file_name='Soil Texture.csv',
#     mime='text/csv',)
