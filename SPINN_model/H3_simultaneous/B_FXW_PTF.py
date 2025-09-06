import numpy as np
import pandas as pd 
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader,TensorDataset 
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device='cpu'
print(f'Current_Device:  {device}')
torch.set_default_dtype(torch.float64)

def B_FXW_PTF(Texture):
    
    ''' PTF'''
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
            out2[:,[0]]=0.01 +1.49*out[:,[0]] # m\n",
            out2[:,[1]]=0.2  +0.65*out[:,[1]] # ths\n",
            out2[:,[2]]=0.001+0.2*out[:,[2]]   # alpha\n",
            out2[:,[3]]=1.01+14*out[:,[3]]  # n\n",
            out2[:,[4]]=1+14*out[:,[4]]      # nc\n",
            out2[:,[5]]=1e-4 + 1e4*out[:,[5]]   # Kha\n",
            out2[:,[6]]=1e-4 + 2e4*out[:,[6]]  # Ks\n",
            return out2
    FNN=FNN_test(input_size=4,hidden_size=226,output_size=7,layers=3)
    FNN.to(device=device)

    model_path_swrc = Path(__file__).parent / "B_FXW_H3.pth"
    FNN.load_state_dict(torch.load(model_path_swrc))


    Tex_list=torch.tensor(np.array(Texture),dtype=torch.float64)  
    with torch.no_grad():  # Predict
        FNN.eval()
        out_para=FNN(Tex_list)
    # out_para=pd.DataFrame(10**out_para)


    # pd.concat([pd.DataFrame(out_para_SWRC),pd.DataFrame(out_para_K)],axis=1).to_csv('B_FXW_Para.csv',header=None) # Save
    pd_Para=pd.concat([pd.DataFrame(out_para)],axis=1)
    pd_Para.columns=['m','ths','alpha','n','nc','K(ha)','Ks']

    np_Para=pd.concat([pd.DataFrame(out_para)],axis=1)
    np_Para=np.array(np_Para)
    np_Para=np_Para[0]
    return np_Para,pd_Para