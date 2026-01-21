import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt
# from sor_model_sum import*
from SPTF_src.sor_model_sum import*
import torch
from torch import nn
from torch import optim
device="cpu"

# ['历史观测','降雨','蒸散发','LST','LAI','Sand,silt,clay,bd']


def forward_func(path):
    ptf_fxw=tptf_FXW20250610()
    ptf_vgm=tptf_VGM20250610()
    testinput = torch.tensor(pd.read_csv(path,index_col=0).to_numpy(),dtype=torch.float32)

    testinput = torch.unsqueeze(testinput,dim=1)
    para_fxw =ptf_fxw(testinput)[0,0,:].numpy()
    para_vgm =ptf_vgm(testinput)[0,0,:].numpy()
    paravgm_dic={ "θ_r":para_vgm[0]/3,
                "θ_s":para_vgm[1],
                "α (1/cm)":10.0**(-para_vgm[2]),
                "n":np.exp(para_vgm[3]),
                "K_s (cm/d)":10**(para_vgm[4]),
                "h_50 (cm)":-1*10**(para_vgm[5]),}
    parfxw_dic={ "θ_s":para_fxw[0],
                "α (1/cm)":10.0**(-para_fxw[1]),
                "n":np.exp(para_fxw[2]),
                "K_s (cm/d)":10**(para_fxw[3]),
                "m":para_fxw[4],
                "h_50 (cm)":-1*10**(para_vgm[5]),}
    return paravgm_dic,parfxw_dic