import numpy as np
import pandas as pd
import torch


def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True,noise=True,noise_ratio=0.2,noise_sever=1,noise_ratio_node=0.2,noise_type='gaussian',noise_ratio_test=0.2,noise_ratio_node_test=0.2,noise_test=True,adj=None):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    print('noise',noise,"noise_ratio_train",noise_ratio,"noise_ratio_test",noise_ratio_test,"noise_sever",noise_sever,"noise_ratio_node_train",noise_ratio_node,"noise_ratio_node_test",noise_ratio_node_test,'noise_type',noise_type)
    if noise==2:        
        indexes=np.arange(train_size)
        indexes_node=np.arange(data.shape[1])
        temp_len=int(train_size*noise_ratio)
        temp_len_node=int(data.shape[1]*noise_ratio_node)
        np.random.shuffle(indexes)
        np.random.shuffle(indexes_node)
        noise_indexes=indexes[:temp_len]
        noise_indexes_node=indexes_node[:temp_len_node]
        mask1=np.zeros_like(train_data).astype(np.bool_)
        mask2=np.zeros_like(train_data).astype(np.bool_)
        mask1[noise_indexes,:]=True
        #print(mask2.shape)
        if len(mask2.shape)==3:
            mask2=mask2.transpose((1,0,2))
        else:
            mask2=mask2.transpose((1,0))
        mask2[noise_indexes_node,:]=True
        if len(mask2.shape)==3:
            mask2=mask2.transpose((1,0,2))
        else:
            mask2=mask2.transpose((1,0))
        mask=mask2&mask1
        max_value=np.max(train_data)-np.min(train_data)
        origin_shape=train_data.shape
        #train_data=train_data.reshape((train_size*data.shape[1],-1))
        #noise_shape=train_data[noise_indexes].shape
        if noise_type=='missing':
            train_data[mask]=0
        elif noise_type=='gaussian':
            train_data[mask]=train_data[mask]+np.random.randn(*origin_shape)[mask]*max_value*noise_sever*0.1
        else:
            print("Error!")
            assert False


        train_data=train_data.reshape(origin_shape)


    #   shape=train_data.shape


    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))

    
    test_size=seq_len
    indexes=np.arange(test_size)
    indexes_node=np.arange(test_data.shape[1])
    
    temp_len=int(test_size*noise_ratio_test)
    temp_len_node=int(data.shape[1]*noise_ratio_node_test)
    
    if noise_test:
        test_data = data[train_size:time_len]
        for i in range(len(test_data) - seq_len - pre_len):
            temp=np.array(test_data[i : i + seq_len])
            
            np.random.shuffle(indexes)
            np.random.shuffle(indexes_node)
            noise_indexes=indexes[:temp_len]
            #print(noise_indexes)
            if 'neighbor' in noise_type:
                noise_indexes_node=[]
                #if temp_len >0:
                #    noise_indexes_node.append[indexes_node[0]]
                n=0
                while  len(noise_indexes_node)< temp_len_node:
                    temp_indexes=np.nonzero(adj[indexes_node[n]])[0].tolist()
                    #print(len(noise_indexes_node),temp_len_node)
                    diff=temp_len_node-len(noise_indexes_node)
                    if len(indexes)>diff:
                        temp_indexes=temp_indexes[:diff]

                    noise_indexes_node1=noise_indexes_node+temp_indexes
                    noise_indexes_node=list(set(noise_indexes_node1))
                    n=n+1

            else:
                noise_indexes_node=indexes_node[:temp_len_node]
                

            #noise_indexes_node=indexes_node[:temp_len_node]
            mask1=np.zeros_like(temp).astype(np.bool_)
            mask2=np.zeros_like(temp).astype(np.bool_)
            mask1[noise_indexes,:]=True
            if len(mask2.shape)==3:
                mask2=mask2.transpose((1,0,2))
            else:
                mask2=mask2.transpose((1,0))
            mask2[noise_indexes_node,:]=True
            if len(mask2.shape)==3:
                mask2=mask2.transpose((1,0,2))
            else:
                mask2=mask2.transpose((1,0))
            mask=mask2&mask1
            max_value=np.max(temp)-np.min(temp)
            origin_shape=temp.shape
            #train_data=train_data.reshape((train_size*data.shape[1],-1))
            #noise_shape=train_data[noise_indexes].shape
            if 'missing' in noise_type:
                temp[mask]=0
            elif 'gaussian' in noise_type:
                temp[mask]=temp[mask]+np.random.randn(*origin_shape)[mask]*max_value*noise_sever*0.1
            else:
                print("Error!")
                assert False
            test_X.append(temp)
            test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    else:
        for i in range(len(test_data) - seq_len - pre_len):
            test_X.append(np.array(test_data[i : i + seq_len]))
            test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    
    np.save("./test_X.npy",np.array(test_X))
    np.save("./test_y.npy",np.array(test_Y))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True,noise=True,noise_ratio=0.2,noise_sever=1,noise_ratio_node=0.2,noise_type='gaussian',noise_ratio_test=0.2,noise_ratio_node_test=0.2,noise_test=True,adj=None
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
        noise=noise,
        noise_ratio=noise_ratio,
        noise_sever=noise_sever,
        noise_ratio_node=noise_ratio_node,
        noise_type=noise_type,
        noise_ratio_test=noise_ratio_test,
        noise_ratio_node_test=noise_ratio_node_test,
        noise_test=noise_test,
        adj=adj
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
