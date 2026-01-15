import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
# from scipy.io import loadmat
from torch.utils.data import DataLoader,TensorDataset 
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device='cpu'
print(f'Current_Device:  {device}')
torch.set_default_dtype(torch.float64)

def B_FXW_PTF(Texture):
    
    ''' PTF-SWRC'''
    class FNN_test(nn.Module):
        def __init__(self,input_size,hidden_size,output_size,layers,**kwargs) -> None:
            super(FNN_test,self).__init__(**kwargs)
            self.input_size=input_size
            self.hideen_size=hidden_size
            self.seq=nn.Sequential()
            for i in range(layers):
                input_size=self.input_size if i ==0 else hidden_size
                self.seq.append(nn.Linear(input_size,self.hideen_size))
                self.seq.append(nn.ReLU())
            self.seq.append(nn.Linear(self.hideen_size,output_size))
            self.seq.append(nn.Sigmoid())
        
        def forward(self,x):
            '''x[batch_size,x_num]'''
            out=self.seq(x)
            out2=torch.zeros(out.shape)
            out2[:,[0]]=-2+2.17609125905568*out[:,[0]] # log10()m
            out2[:,[1]]=-0.698970004336019+0.628388930050312*out[:,[1]] # log10()ths
            out2[:,[2]]=-3+2.3010*out[:,[2]]  # log10()alpha
            out2[:,[3]]=0.0043+1.17176988527304*out[:,[3]] # log10()n
            out2[:,[4]] = 0 + 1.1761 * out[:, [4]]      # n_c
            return out2
    FNN=FNN_test(input_size=4,hidden_size=226,output_size=5,layers=3)
    FNN.to(device=device)

    model_path_swrc = Path(__file__).parent / "B_FXW_H3_SWRC.pth"
    FNN.load_state_dict(torch.load(model_path_swrc))


    Tex_list=torch.tensor(np.array(Texture),dtype=torch.float64)  
    with torch.no_grad():  # Predict
        FNN.eval()
        out_para=FNN(Tex_list)
    out_para_SWRC=pd.DataFrame(10**out_para)

    '''PTF-HCC'''
    class FNN_test(nn.Module):
        def __init__(self,input_size,hidden_size,output_size,layers,**kwargs) -> None:
            super(FNN_test,self).__init__(**kwargs)
            self.input_size=input_size
            self.hideen_size=hidden_size
            self.seq=nn.Sequential()
            for i in range(layers):
                input_size=self.input_size if i ==0 else hidden_size
                self.seq.append(nn.Linear(input_size,self.hideen_size))
                self.seq.append(nn.ReLU())
            self.seq.append(nn.Linear(self.hideen_size,output_size))
            self.seq.append(nn.Sigmoid())
        
        def forward(self,x):
            '''x[batch_size,x_num]'''
            out=self.seq(x)
            out2=torch.zeros(out.shape)
            out2[:, [0]] = -4 + 8 * out[:, [0]]      # Kha
            out2[:, [1]] = -4 + 8.5 * out[:, [1]]      # Ks
            return out2
    FNN=FNN_test(input_size=4,hidden_size=226,output_size=2,layers=3)
    FNN.to(device=device)

    model_path_k = Path(__file__).parent / "B_FXW_H3_K.pth"
    FNN.load_state_dict(torch.load(model_path_k))

    # Texture=np.array(pd.read_csv('texture.csv'))
    Tex_list=torch.tensor(np.array(Texture),dtype=torch.float64)  
    with torch.no_grad():  # Predict
        FNN.eval()
        out_para=FNN(Tex_list)
    out_para_K=pd.DataFrame(10**out_para)
    mask = out_para_K[0] > out_para_K[1]  
    out_para_K.loc[mask, 0] = out_para_K.loc[mask, 1]  # Kha & Ks

    # pd.concat([pd.DataFrame(out_para_SWRC),pd.DataFrame(out_para_K)],axis=1).to_csv('B_FXW_Para.csv',header=None) # Save
    pd_Para=pd.concat([pd.DataFrame(out_para_SWRC),pd.DataFrame(out_para_K)],axis=1)
    pd_Para.columns=['m','ths','alpha','n','nc','K(ha)','Ks']

    np_Para=pd.concat([pd.DataFrame(out_para_SWRC),pd.DataFrame(out_para_K)],axis=1)
    np_Para=np.array(np_Para)
    np_Para=np_Para[0]
    return np_Para,pd_Para