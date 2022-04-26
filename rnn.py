import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


device = "cuda:3" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using {device} device")


class RNN_scratch(nn.Module):
    'Simple Recurrent Network'
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(RNN_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        self.U = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.W = Parameter(torch.randn(self.hidden_size, self.input_size))
        self.b = Parameter(torch.randn(self.hidden_size, 1))
        self.V = Parameter(torch.randn(self.output_size, self.hidden_size))

    def forward(self, input):
        H = torch.tensor([]).to(device)
        Y = torch.randn(self.output_size, 1).to(device)
        H = torch.cat((H, input[:, 0].reshape(-1, 1)), dim=1) # FIXME: H[0]取值可能有问题，若input_size、hidden_size不一致，H[0]设置成x[0]会有问题
        for i in range(input.shape[1]):
            cur_h = torch.sigmoid(torch.mm(self.U, H[:, i].reshape(-1, 1)) + torch.mm(self.W, input[:, i].reshape(-1, 1)) + self.b)
            H = torch.cat((H, cur_h.reshape(-1, 1)), dim=1)
            cur_y = torch.mm(self.V, cur_h)
            Y = torch.cat((Y, cur_y.reshape(-1, 1)), dim=1)
        return Y[:, 1:], H[:, 1:]


class LSTM_scratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(LSTM_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        for suf in ['c', 'o', 'i', 'f']:
            setattr(self, 'U' + suf, Parameter(torch.randn(self.hidden_size, self.hidden_size)))
            setattr(self, 'W' + suf, Parameter(torch.randn(self.hidden_size, self.input_size)))
            setattr(self, 'b' + suf, Parameter(torch.randn(self.hidden_size, 1)))
    
    def forward(self, input):
        H = input[:, 0].reshape(-1, 1).to(device)
        C = input[:, 0].reshape(-1, 1).to(device) # 目前c的维度设置为hidden_size
        names = self.state_dict()
        for i in range(input.shape[1]):
            c_hat = torch.tanh(torch.mm(names['Uc'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wc'], input[:, i].reshape(-1, 1)) + names['bc'])
            gate_o = torch.sigmoid(torch.mm(names['Uo'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wo'], input[:, i].reshape(-1, 1)) + names['bo'])
            gate_i = torch.sigmoid(torch.mm(names['Ui'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wi'], input[:, i].reshape(-1, 1)) + names['bi'])
            gate_f = torch.sigmoid(torch.mm(names['Uf'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wf'], input[:, i].reshape(-1, 1)) + names['bf'])

            cur_c = gate_f * C[:, i].reshape(-1, 1) + gate_i * c_hat
            cur_h = gate_o * torch.tanh(cur_c)
            C = torch.cat((C, cur_c.reshape(-1, 1)), dim=1)
            H = torch.cat((H, cur_h.reshape(-1, 1)), dim=1)
        return H[:, 1:],


class GRU_scratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(GRU_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        for suf in ['z', 'r', 'h']:
            setattr(self, 'U' + suf, Parameter(torch.randn(self.hidden_size, self.hidden_size)))
            setattr(self, 'W' + suf, Parameter(torch.randn(self.hidden_size, self.input_size)))
            setattr(self, 'b' + suf, Parameter(torch.randn(self.hidden_size, 1)))
    
    def forward(self, input):
        H = input[:, 0].reshape(-1, 1).to(device)
        names = self.state_dict()
        for i in range(input.shape[1]):
            gate_z = torch.sigmoid(torch.mm(names['Uz'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wz'], input[:, i].reshape(-1, 1)) + names['bz'])
            gate_r = torch.sigmoid(torch.mm(names['Ur'], H[:, i].reshape(-1, 1)) + torch.mm(names['Wr'], input[:, i].reshape(-1, 1)) + names['br'])
            h_hat = torch.tanh(torch.mm(names['Uh'], gate_r * H[:, i].reshape(-1, 1)) + torch.mm(names['Wh'], input[:, i].reshape(-1, 1)) + names['bh'])

            cur_h = gate_z * H[:, i].reshape(-1, 1) + (1 - gate_z) * h_hat
            H = torch.cat((H, cur_h.reshape(-1, 1)), dim=1)
        return H[:, 1:],


class SRNN_scratch(nn.Module):
    '''Stacked Recurrent Nerual Network'''
    def __init__(self, input_size, hidden_size, stacked_layer_nums, output_size) -> None:
        super(SRNN_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.stacked_layer_nums = stacked_layer_nums
        self.init_weights()

    def init_weights(self):
        for i in range(self.stacked_layer_nums):
            setattr(self, 'U' + str(i), Parameter(torch.randn(self.hidden_size, self.hidden_size)))
            setattr(self, 'W' + str(i), Parameter(torch.randn(self.hidden_size, self.input_size)))
            setattr(self, 'b' + str(i), Parameter(torch.randn(self.hidden_size, 1)))
        self.V = Parameter(torch.randn(self.output_size, self.hidden_size))

    def forward(self, input):
        # input.shape: (input_size, sequence_len)
        Y = torch.randn(self.output_size, 1).to(device)
        # H.shape: (sequence_len+1, hidden_size, stacked_layer_nums+1)
        H = torch.zeros(input.shape[1]+1, self.hidden_size, self.stacked_layer_nums+1).to(device)
        H[0] = input[:, 0].reshape(-1, 1).repeat(1, 4)
        for i in range(1, input.shape[1]):
            H[i, :, 0] = input[:, i]
        names = self.state_dict()
        for i in range(input.shape[1]):
            for j in range(1, self.stacked_layer_nums+1):
                cur_h = torch.sigmoid(torch.mm(names['U' + str(j-1)], H[i, :, j].reshape(-1, 1)) + torch.mm(names['W' + str(j-1)], H[i+1, :, j-1].reshape(-1, 1)) + names['b' + str(j-1)])
                H[i+1, :, j] = cur_h.reshape(1, -1).squeeze(0)
            cur_y = torch.mm(self.V, cur_h)
            Y = torch.cat((Y, cur_y.reshape(-1, 1)), dim=1)
        return Y[:, 1:],


class BI_RNN_scratch(nn.Module):
    'Simple Recurrent Network'
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(BI_RNN_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self):
        for suf in ['l', 'r']:
            setattr(self, 'U' + suf, Parameter(torch.randn(self.hidden_size, self.hidden_size)))
            setattr(self, 'W' + suf, Parameter(torch.randn(self.hidden_size, self.input_size)))
            setattr(self, 'b' + suf, Parameter(torch.randn(self.hidden_size, 1)))
        self.V = Parameter(torch.randn(self.output_size, 2 * self.hidden_size))
    
    def unidirectional_forward(self, input, direc='l'):
        names = self.state_dict()
        H = torch.tensor([]).to(device)
        H = torch.cat((H, input[:, 0].reshape(-1, 1)), dim=1)
        for i in range(input.shape[1]):
            cur_h = torch.sigmoid(torch.mm(names['U' + direc], H[:, i].reshape(-1, 1)) + torch.mm(names['W' + direc], input[:, i].reshape(-1, 1)) + names['b' + direc])
            H = torch.cat((H, cur_h.reshape(-1, 1)), dim=1)
        return H[:, 1:]

    def forward(self, input):
        H = self.unidirectional_forward(input)
        input_reverse = torch.flip(input, dims=[1])
        H_reverse = self.unidirectional_forward(input_reverse, direc='r')
        H_concat = torch.cat((H, H_reverse), dim=0)

        Y = torch.randn(self.output_size, 1).to(device)
        for i in range(H_concat.shape[1]):
            cur_y = torch.mm(self.V, H_concat[:, i].reshape(-1, 1))
            Y = torch.cat((Y, cur_y.reshape(-1, 1)), dim=1)
        
        return Y[:, 1:],


if __name__ == '__main__':
    # input.shape: (sequence_len, input_size)
    input = torch.randn(10, 8).to(device)
    # model = RNN_scratch(10, 10, 5)
    # model = LSTM_scratch(10, 10, 5)
    # model = GRU_scratch(10, 10, 5)
    model = SRNN_scratch(10, 10, 3, 5)
    # model = BI_RNN_scratch(10, 10, 5)
    model.to(device)
    print(model(input)[0])
