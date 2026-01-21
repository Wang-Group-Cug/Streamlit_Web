import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import sys
from pathlib import Path

# %%
def surrogate_model_FXW():
#     ('''共计9个输入'''+"\n"
# '''P,PET,LAI,ths,-log10(alpha),log(n),log10(ks),m,log10(-P50)''')
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)


        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            outputi=outputi*inputs[:,:,[3]].repeat(1,1,1)*100
            return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    lstm_layer=nn.LSTM(input_size=9,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=1)
    lstm_model.load_state_dict(
        torch.load(
            r"D:\替代模型训练集20240902\sorfor_fxw_thslimtestloss0.10600734.pth"
            )
        )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model

def surrogate_model_VGM():
    '''P,PET,LAI,thr*3,ths,-log10(alpha),log(n),log10(ks),-log10(P50)'''
    class Rnn_model(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)


        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            outputi=100*outputi*inputs[:,:,[4]].repeat(1,1,1)

            # outputi=outputi*(inputs[:,:,[3]]-inputs[:,:,[2]])+inputs[:,:,[2]]
            return output,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    lstm_layer=nn.LSTM(input_size=9,hidden_size=256,num_layers=1)
    lstm_model=Rnn_model(lstm_layer,output_size=1)
    lstm_model.load_state_dict(
        torch.load(
            r"D:\替代模型训练集20240902\sorfor_vgm_thslim0.09317286312580109.pth"
            )
        )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model



def tptf_FXW():
    '''ptf:输入'''
    '''ptf 输入：sm0.05,pointer,LAI,sand,silt,clay,bd'''
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            output2i[:,:,0]=outputi[:,:,0]*(
                torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
                                            ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            '''alpha'''
            output2i[:,:,1]=outputi[:,:,1]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(13)'''
            output2i[:,:,2]=outputi[:,:,2]*(torch.log(torch.tensor(13))-torch.log(torch.tensor(1.1)))+torch.log(torch.tensor(1.1))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''m     0.2~1.6'''
            output2i[:,:,4]=outputi[:,:,4]*(1.6-0.2)+0.2
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,4]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=7,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"D:\时序PTF训练\PTFfor_FXWM3_SD_16674.080078125.pth"
        # r'D:\时序PTF训练\PTFfor_FXWM3_SD_testfunc_59.017154693603516.pth'
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model

def tptf_FXW2025():
    '''ptf:输入'''
    '''ptf 输入：sm0.05,pointer,LAI,sand,silt,clay,bd'''
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            output2i[:,:,0]=outputi[:,:,0]*(
                torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
                                            ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            '''alpha'''
            output2i[:,:,1]=outputi[:,:,1]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(13)'''
            output2i[:,:,2]=outputi[:,:,2]*(torch.log(torch.tensor(13))-torch.log(torch.tensor(1.1)))+torch.log(torch.tensor(1.1))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''m     0.2~1.6'''
            output2i[:,:,4]=outputi[:,:,4]*(1.6-0.2)+0.2
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,4]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=7,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"D:\时序PTF训练\SPTFfor_FXWM3_SD_2025_17673.82421875.pth"
        # r'D:\时序PTF训练\PTFfor_FXWM3_SD_testfunc_59.017154693603516.pth'
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model

def tptf_FXW2025_P():
    '''ptf:输入'''
    '''ptf 输入：sm0.05,pointer,LAI,sand,silt,clay,bd'''
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            output2i[:,:,0]=outputi[:,:,0]*(
                torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
                                            ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            '''alpha'''
            output2i[:,:,1]=outputi[:,:,1]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(13)'''
            output2i[:,:,2]=outputi[:,:,2]*(torch.log(torch.tensor(13))-torch.log(torch.tensor(1.1)))+torch.log(torch.tensor(1.1))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''m     0.2~1.6'''
            output2i[:,:,4]=outputi[:,:,4]*(1.6-0.2)+0.2
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,4]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=8,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"D:\时序PTF训练\SPTFfor_FXWM3_SDaddP_2025_16912.08203125.pth"
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model


def tptf_FXW20250610():
    '''ptf:输入'''
    '''ptf 输入：sm0.05,pointer,LAI,sand,silt,clay,bd'''
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            output2i[:,:,0]=outputi[:,:,0]*(
                torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
                                            ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            '''alpha'''
            output2i[:,:,1]=outputi[:,:,1]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(13)'''
            output2i[:,:,2]=outputi[:,:,2]*(torch.log(torch.tensor(13))-torch.log(torch.tensor(1.1)))+torch.log(torch.tensor(1.1))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''m     0.2~1.6'''
            output2i[:,:,4]=outputi[:,:,4]*(1.6-0.2)+0.2
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,4]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
    # current_path = 
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=9,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    current_path = Path(__file__).parent / "SPTFfor_FXWM3_SDaddPPET_2025_17374.046875.pth"
    lstm_model.load_state_dict(
    torch.load(
        current_path,
        map_location=torch.device('cpu'))
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model




def tptf_VGM():
    '''ptf:输入'''
    '''ptf 输入：sm0.05,pointer,LAI,sand,silt,clay,bd'''
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            '''thr 3*(0~0.15)'''

            output2i[:,:,0]=outputi[:,:,0]*(torch.tensor(0.15))*3

            '''ths 0.32~max(0.41,max(obs))'''
            output2i[:,:,1]=outputi[:,:,1]*(
                torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
                                            ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            '''alpha '''
            output2i[:,:,2]=outputi[:,:,2]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(6)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log(torch.tensor(6))-torch.log(torch.tensor(1.05)))+torch.log(torch.tensor(1.05))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,4]=outputi[:,:,4]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,5]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=7,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"PTFfor_VGM_SD_17448.3203125.pth"
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model

def tptf_VGM2025():
    '''ptf:输入'''
    '''ptf 输入：sm0.05,pointer,LAI,sand,silt,clay,bd'''
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            '''thr 3*(0~0.15)'''

            output2i[:,:,0]=outputi[:,:,0]*(torch.tensor(0.15))*3

            '''ths 0.32~max(0.41,max(obs))'''
            output2i[:,:,1]=outputi[:,:,1]*(
                torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
                                            ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            '''alpha '''
            output2i[:,:,2]=outputi[:,:,2]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(6)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log(torch.tensor(6))-torch.log(torch.tensor(1.05)))+torch.log(torch.tensor(1.05))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,4]=outputi[:,:,4]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,5]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=7,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"D:\时序PTF训练\SPTFfor_VGM_SD_2025_18619.93359375.pth"
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model

def tptf_VGM2025_P():
    '''ptf:输入'''
    '''ptf 输入：sm0.05,pointer,LAI,sand,silt,clay,bd'''
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            '''thr 3*(0~0.15)'''

            output2i[:,:,0]=outputi[:,:,0]*(torch.tensor(0.15))*3

            '''ths 0.32~max(0.41,max(obs))'''
            output2i[:,:,1]=outputi[:,:,1]*(
                torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
                                            ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            '''alpha '''
            output2i[:,:,2]=outputi[:,:,2]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(6)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log(torch.tensor(6))-torch.log(torch.tensor(1.05)))+torch.log(torch.tensor(1.05))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,4]=outputi[:,:,4]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,5]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=8,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"SPTFfor_VGM_SD_2025_17694.365234375.pth"
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model


def tptf_VGM20250610():
    '''ptf:输入'''
    '''ptf 输入：sm0.05,pointer,LAI,sand,silt,clay,bd'''
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            '''thr 3*(0~0.15)'''

            output2i[:,:,0]=outputi[:,:,0]*(torch.tensor(0.15))*3

            '''ths 0.32~max(0.41,max(obs))'''
            output2i[:,:,1]=outputi[:,:,1]*(
                torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
                                            ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            '''alpha '''
            output2i[:,:,2]=outputi[:,:,2]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(6)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log(torch.tensor(6))-torch.log(torch.tensor(1.05)))+torch.log(torch.tensor(1.05))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,4]=outputi[:,:,4]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,5]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
            
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=9,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    current_path = Path(__file__).parent/"SPTFfor_VGM_SDaddPPET_2025_17675.349609375.pth"
    lstm_model.load_state_dict(
    torch.load(
        current_path
        ,map_location=torch.device('cpu')
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model


def ptf_notimeseries_FXW():
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            # output2i[:,:,0]=outputi[:,:,0]*(
            #     torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
            #                                 ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            output2i[:,:,0]=0.32+outputi[:,:,0]*0.33
            '''alpha'''
            output2i[:,:,1]=outputi[:,:,1]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(13)'''
            output2i[:,:,2]=outputi[:,:,2]*(torch.log(torch.tensor(13))-torch.log(torch.tensor(1.1)))+torch.log(torch.tensor(1.1))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''m     0.2~1.6'''
            output2i[:,:,4]=outputi[:,:,4]*(1.6-0.2)+0.2
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,5]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=4,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"D:\时序PTF训练\PTFfor(notimeseries)_FXWM3_1031_SD_25597.69921875.pth"
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model

def SPTFstatic2025_FXW():
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            # output2i[:,:,0]=outputi[:,:,0]*(
            #     torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
            #                                 ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            output2i[:,:,0]=0.32+outputi[:,:,0]*0.33
            '''alpha'''
            output2i[:,:,1]=outputi[:,:,1]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(13)'''
            output2i[:,:,2]=outputi[:,:,2]*(torch.log(torch.tensor(13))-torch.log(torch.tensor(1.1)))+torch.log(torch.tensor(1.1))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''m     0.2~1.6'''
            output2i[:,:,4]=outputi[:,:,4]*(1.6-0.2)+0.2
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,5]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
    torch.manual_seed(6) # 固定随机数
    lstm_layer=nn.LSTM(input_size=4,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"D:\时序PTF训练\SPTFstatic_forFXWM3_SD_33040.29296875.pth"
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model

def ptf_notimeseries_VGM():
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            '''thr 3*(0~0.15)'''

            output2i[:,:,0]=outputi[:,:,0]*(torch.tensor(0.15))*3

            # '''ths 0.32~max(0.41,max(obs))'''
            # output2i[:,:,1]=outputi[:,:,1]*(
            #     torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
            #                                 ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            output2i[:,:,1]=0.32+outputi[:,:,1]*0.33
            '''alpha '''
            output2i[:,:,2]=outputi[:,:,2]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(6)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log(torch.tensor(6))-torch.log(torch.tensor(1.05)))+torch.log(torch.tensor(1.05))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,4]=outputi[:,:,4]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,5]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
    torch.manual_seed(9) # 固定随机数
    lstm_layer=nn.LSTM(input_size=4,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"D:\时序PTF训练\PTFfor(notimeseries)_VGM_1031_SD_25986.373046875.pth"
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model

def SPTFstatic2025_VGM():
    class Rnn_model_test(nn.Module):
        def __init__(self,rnn_layer,output_size,**kwargs) -> None:
            super(Rnn_model_test,self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.output_size=output_size
            self.num_hiddens=self.rnn.hidden_size
            self.relu_t=nn.Sigmoid()
            # self.relu_t=nn.ReLU()
            # 如果RNN是双向的，num_directions应该是2，否则应该是1
            if not self.rnn.bidirectional:
                self.num_directions = 1
                self.linear = nn.Linear(self.num_hiddens, self.output_size)
            else:
                self.num_directions = 2
                self.linear = nn.Linear(self.num_hiddens * 2, self.output_size)

        def forward(self,inputs,state=None):
            Yi,state=self.rnn(inputs,state)
            output=self.linear(Yi)
            outputi=self.relu_t(output)
            output2i=torch.ones_like(outputi)

            '''thr 3*(0~0.15)'''

            output2i[:,:,0]=outputi[:,:,0]*(torch.tensor(0.15))*3

            # '''ths 0.32~max(0.41,max(obs))'''
            # output2i[:,:,1]=outputi[:,:,1]*(
            #     torch.max(inputs[:,:,0].max(dim=0)[0],torch.tensor(0.41))-0.32 # ths
            #                                 ) +0.32# 这样设置ths为0.32~max(0.41,max(obs))
            output2i[:,:,1]=0.32+outputi[:,:,1]*0.33
            '''alpha '''
            output2i[:,:,2]=outputi[:,:,2]*(-np.log10(0.0015)+np.log10(0.2))-np.log10(0.2) # alpha -log10(0.0015)~-log10(0.2)
            '''n     log(1.1)~log(6)'''
            output2i[:,:,3]=outputi[:,:,3]*(torch.log(torch.tensor(6))-torch.log(torch.tensor(1.05)))+torch.log(torch.tensor(1.05))
            '''ks    log10(0.5)~log10(8000)'''
            output2i[:,:,4]=outputi[:,:,4]*(torch.log10(torch.tensor(8000))-torch.log10(torch.tensor(0.5)))+torch.log10(torch.tensor(0.5))
            '''p50   log10(50)~log10(2000)'''
            output2i[:,:,5]=outputi[:,:,5]*(torch.log10(torch.tensor(2000))-torch.log10(torch.tensor(55)))+torch.log10(torch.tensor(55))

            # outputi=outputi*inputs[:,:,[2]].repeat(1,1,1)*100
            return torch.mean(output2i,dim=0).unsqueeze(0).repeat(365,1,1)
            # return outputi,state
        
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens),
                    device=device)
            else :
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device))
    torch.manual_seed(9) # 固定随机数
    lstm_layer=nn.LSTM(input_size=4,hidden_size=128,num_layers=1)
    lstm_model=Rnn_model_test(lstm_layer,output_size=6)
    lstm_model.load_state_dict(
    torch.load(
        r"D:\时序PTF训练\SPTFstatic_forvgm_SD_33774.76953125.pth"
        )
    )
    for parameter in lstm_model.parameters():
        parameter.requires_grad=False
    return lstm_model