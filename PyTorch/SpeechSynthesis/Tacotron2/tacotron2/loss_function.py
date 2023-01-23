# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from torch import nn
from torch import tensor
import pandas as pd

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, loss_function):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        
        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)

        nonpadded_f=[0]*len(mel_target)

        for i, mel in enumerate(mel_target):
            nonzero=[]
            for f in range(mel.size()[1]):
                count=mel[range(len(mel_target[i])),f].count_nonzero().item()
                if count>=1: value=1
                else: value=0
                nonzero.append(value)
                nonpadded_f[i]=nonzero.count(1)
        
        if loss_function=='mse':
            mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                nn.MSELoss()(mel_out_postnet, mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
            return mel_loss + gate_loss

        else:
            mel_loss_i=[0]*len(mel_target)
            mel_postnet_loss_i=[0]*len(mel_target)
            gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
            total_loss_i=[0]*len(mel_target)
            for i in range(0,len(mel_target)):
                mel_loss_i[i]=nn.MSELoss(reduction='none')(mel_out[i], mel_target[i]).sum()/tensor(nonpadded_f[i]*len(mel_target[i]))
                mel_postnet_loss_i[i]=nn.MSELoss(reduction='none')(mel_out_postnet[i], mel_target[i]).sum()/tensor(nonpadded_f[i]*len(mel_target[i]))
                total_loss_i[i]=mel_loss_i[i]+mel_postnet_loss_i[i]+gate_loss
            total_loss=sum(total_loss_i)/len(total_loss_i)
            return total_loss

class Tacotron2Loss_passage(nn.Module):
    def __init__(self):
        super(Tacotron2Loss_passage, self).__init__()
    def forward(self, model_output, targets, len_x, text_padded, loss_function):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        #mel_loss = nn.MSELoss(reduction='none')(mel_out, mel_target) + \
            #nn.MSELoss(reduction='none')(mel_out_postnet, mel_target)
        #gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        lossdf=pd.DataFrame({'i':[], 'count_i':[],'count_non0_i':[], 'nonpadded_f':[],'mel_loss_i':[],'mel_postnet_loss_i':[],'gate_loss':[],'total_loss_i':[]})
        
        #return print(mel_target[0].size())

        nonpadded_f=[0]*len(mel_target)

        for i, mel in enumerate(mel_target):
            nonzero=[]
            for f in range(mel.size()[1]):
                count=mel[range(len(mel_target[i])),f].count_nonzero().item()
                if count>=1: value=1
                else: value=0
                nonzero.append(value)
                nonpadded_f[i]=nonzero.count(1)
        
        print('melbincount: '+str(len(mel_target[i])))
        
        print(gate_target.size())
        #print(gate_target)
        
        #return print(nonzero[0])

        #mel_target[i].count_nonzero().cpu().detach().item()

        for i in range(0,len(mel_target)):
            gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target).cpu().detach().item()
            if loss_function=='mse':
                mel_loss_i=nn.MSELoss()(mel_out[i], mel_target[i]).cpu().detach().item()
                mel_postnet_loss_i=nn.MSELoss()(mel_out_postnet[i], mel_target[i]).cpu().detach().item()
            else:
                mel_loss_i=nn.MSELoss(reduction='none')(mel_out[i], mel_target[i]).sum().cpu().detach().item()/(nonpadded_f[i]*len(mel_target[i]))
                mel_postnet_loss_i=nn.MSELoss(reduction='none')(mel_out_postnet[i], mel_target[i]).sum().cpu().detach().item()/(nonpadded_f[i]*len(mel_target[i]))
            total_loss_i=mel_loss_i+mel_postnet_loss_i+gate_loss
            len_i_0=len(mel_target[i][0])
            count_i=mel_target[i].numel()
            count_non0_i=mel_target[i].count_nonzero().cpu().detach().item()
            lossdf=pd.concat([lossdf,pd.DataFrame({'i':[i], 'count_i':[count_i], 'count_non0_i':[count_non0_i], 'nonpadded_f':[nonpadded_f[i]], 'mel_loss_i':[mel_loss_i],'mel_postnet_loss_i':[mel_postnet_loss_i],'gate_loss':[gate_loss],'total_loss_i':[total_loss_i]})],axis=0)
        return lossdf            
