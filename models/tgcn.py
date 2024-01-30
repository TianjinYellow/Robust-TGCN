import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop


class RTGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0,lamda:float=1.0):
        super(RTGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.lamda=lamda
        self.register_buffer(
            "laplacian0", calculate_laplacian_with_self_loop(torch.FloatTensor(adj),alpha=-0.5)
        )
        self.register_buffer(
            "laplacian1", calculate_laplacian_with_self_loop(torch.FloatTensor(adj),alpha=-1.0)
        )
        self.weights_mean = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases_mean = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.weights_var = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases_var = nn.Parameter(torch.FloatTensor(self._output_dim))

        
        self.inputs_weights_mean=nn.Parameter(torch.FloatTensor(1, 1))
        self.inputs_weights_var=nn.Parameter(torch.FloatTensor(1, 1))       
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_mean)
        nn.init.xavier_uniform_(self.weights_var)
        nn.init.xavier_uniform_(self.inputs_weights_mean)
        nn.init.xavier_uniform_(self.inputs_weights_var)
        nn.init.constant_(self.biases_var, self._bias_init_value)
        nn.init.constant_(self.biases_mean, self._bias_init_value)
    def forward(self, inputs, hidden_state):

        hidden_mean,hidden_var=torch.chunk(hidden_state,chunks=2,dim=-1)
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        input_mean=torch.bmm(inputs,self.inputs_weights_mean.view(1,1,1).repeat((batch_size,1,1)))
        input_var=torch.bmm(inputs,self.inputs_weights_var.view(1,1,1).repeat((batch_size,1,1)))
        
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state_mean = hidden_mean.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        hidden_state_var = hidden_var.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation_mean = torch.cat((input_mean, hidden_state_mean), dim=2)
        concatenation_var=torch.cat((input_var,hidden_state_var),dim=2)
        


        ###############linear operation################################
        
        # [x, h] (batch_size * num_nodes, num_gru_units + 1)
        concatenation_mean = concatenation_mean.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        concatenation_var = concatenation_var.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        concatenation_mean = torch.relu(concatenation_mean @ self.weights_mean + self.biases_mean)
        concatenation_var = torch.relu(concatenation_var @ self.weights_var + self.biases_var)
        #concatenation_var=torch.abs(concatenation_var)
        ##create alpha##########
        node_weights=torch.exp(-1*concatenation_var*self.lamda).reshape((batch_size,num_nodes,self._num_gru_units))
        node_weights=node_weights.transpose(0,1).transpose(1,2).reshape((num_nodes, (self._num_gru_units ) * batch_size))


        ###########Aggregation#######################################
        concatenation_mean=concatenation_mean.reshape((batch_size,num_nodes,self._num_gru_units))        
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation_mean = concatenation_mean.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation_mean = concatenation_mean.reshape(
            (num_nodes, (self._num_gru_units ) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        outputs_mean = self.laplacian0 @ (concatenation_mean*node_weights)
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        outputs_mean = outputs_mean.reshape(
            (num_nodes, self._num_gru_units , batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        outputs_mean = outputs_mean.transpose(0, 2).transpose(1, 2)


        ######varaiance####
        concatenation_var=concatenation_var.reshape((batch_size,num_nodes,self._num_gru_units))        
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation_var = concatenation_var.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation_var = concatenation_var.reshape(
            (num_nodes, (self._num_gru_units ) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        outputs_var = self.laplacian1 @ (concatenation_var*node_weights*node_weights)
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        outputs_var = outputs_var.reshape(
            (num_nodes, self._num_gru_units, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        outputs_var = outputs_var.transpose(0, 2).transpose(1, 2)

        """
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        """
        outputs=torch.cat((outputs_mean,outputs_var),dim=-1)
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        #outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int,lamda=1.0,N_layers=2):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )

        self.graph_conv1_1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
        )
        self.N_layers=N_layers
        if N_layers==4:
            self.graph_conv3 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
        elif N_layers==6:
            self.graph_conv3 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
            self.graph_conv4 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
        elif N_layers==8:
            self.graph_conv3 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
            self.graph_conv4 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
            self.graph_conv5 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
        elif N_layers==10:
            self.graph_conv3 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
            self.graph_conv4 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
            self.graph_conv5 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )
            self.graph_conv6 = RTGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim,lamda=lamda
            )



    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        hidden_mean,hidden_var=torch.chunk(hidden_state,chunks=2,dim=-1)
        self.mean=hidden_mean
        self.var=hidden_var
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_mean))
        concatenation1 = torch.sigmoid(self.graph_conv1_1(inputs, hidden_mean))
        #  r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=-1)
        r1, u1 = torch.chunk(concatenation1, chunks=2, dim=-1)
        #r=r.reshape()

        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        #print(r.shape)
        #print(hidden_state.shape)
        #print(hidden_mean.shape)
        #print(r.shape)
        #hidden_state[:,:,:hidden_mean.shape[2]]=hidden_state[:,:,:hidden_mean.shape[2]]*r
        #hidden_state[:,:,hidden_mean.shape[2]:]=hidden_state[:,:,hidden_mean.shape[2]:]*r*r
        r_concat=torch.cat((r,r1),dim=2)
        hidden_state=hidden_state*r_concat
        c = self.graph_conv2(inputs,  hidden_state)
        if self.N_layers==4:
            c=self.graph_conv3(inputs,c)
        elif self.N_layers==6:
            c=self.graph_conv3(inputs,c)
            c=self.graph_conv4(inputs,c)
        elif self.N_layers==8:
            c=self.graph_conv3(inputs,c)
            c=self.graph_conv4(inputs,c)
            c=self.graph_conv5(inputs,c)
        elif self.N_layers==10:
            c=self.graph_conv3(inputs,c)
            c=self.graph_conv4(inputs,c)
            c=self.graph_conv5(inputs,c)
            c=self.graph_conv6(inputs,c)

        

        c_mean,c_var=torch.chunk(c,chunks=2,dim=-1)
        c_mean=torch.relu(c_mean)
        c_var=torch.nn.functional.relu(c_var)
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_mean = u *hidden_mean + (1.0 - u) * c_mean
        new_hidden_var = u1*hidden_var + (1.0 - u1)*c_var
        new_hidden_state=torch.cat((new_hidden_mean,new_hidden_var),dim=-1)
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int,lamda:float,dim_input:int,N_layers=2, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim

        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim,lamda,N_layers=N_layers)

    def forward(self, inputs):
        #print(inputs.shape)
        if len(inputs.shape)==4:
            #batch_size, seq_len, num_nodes,_ = inputs.shape
            #inputs=self.input_transform(inputs).squeeze()
            assert False 
            #print(inputs.shape)
        else:
            batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes , self._hidden_dim*2).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            #output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
