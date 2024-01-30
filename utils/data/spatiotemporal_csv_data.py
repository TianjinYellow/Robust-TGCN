import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions
import argparse
import numpy as np
import pandas as pd
import pickle


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def get_adjacency_matrix(args):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    if 'pkl' in args.sensor_ids_filename:
        sensor_ids, sensor_id_to_ind, adj_mx=load_graph_data(args.sensor_ids_filename)
        return sensor_ids, sensor_id_to_ind, adj_mx
    with open(args.sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})

    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < args.normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx

def generate_graph_seq2seq_io_data(args, add_time_in_day=True, add_day_in_week=False):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    df = pd.read_hdf(args.traffic_df_filename)   

    # 0 is the latest observed sample.
    # x_offsets = np.sort(
    #     # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
    #     np.concatenate((np.arange(-11, 1, 1),))
    # )
    # # Predict the next one hour
    # y_offsets = np.sort(np.arange(1, 13, 1))

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    return data


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        adj_path: str,
        batch_size: int = 64,
        seq_len: int = 12,
        pre_len: int = 3,
        split_ratio: float = 0.8,
        normalize: bool = True,
        noise=True,
        noise_ratio=0.2,
        noise_sever=1,
        noise_ratio_node=0.2,
        noise_type='gaussian',
        noise_ratio_test=0.2,
        noise_ratio_node_test=0.2,
        noise_test=True,
        args=None,
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self.noise=noise
        self.noise_test=noise_test
        self.noise_ratio_test=noise_ratio_test
        self.noise_ratio_node_test=noise_ratio_node_test
        self.noise_ratio=noise_ratio
        self.noise_sever=noise_sever
        self.noise_ratio_node=noise_ratio_node
        self.noise_type=noise_type
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        if args.data=='MeterLA' or args.data=='PemsBAY':
            self._feat=generate_graph_seq2seq_io_data(args)[:,:,0].squeeze()
            #print(self._feat.shape)
            self._feat_max_val = np.max(self._feat)
            _,_,self._adj=get_adjacency_matrix(args)
            #print(self._adj.shape)
        else:
            self._feat = utils.data.functions.load_features(self._feat_path)
            #print(self._feat.shape,flush=True)
            self._feat_max_val = np.max(self._feat)
            self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)
            #print(self._adj.shape)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--seq_len", type=int, default=12)
        parser.add_argument("--dim_input", type=int, default=2)
        parser.add_argument("--pre_len", type=int, default=3)
        parser.add_argument("--traffic_df_filename",type=str,default="data/metr-la.h5",help="Raw traffic readings.")
        parser.add_argument('--sensor_ids_filename', type=str, default='data/sensor_graph/graph_sensor_ids.txt',help='File containing sensor ids separated by comma.')
        parser.add_argument('--distances_filename', type=str, default='data/sensor_graph/distances_la_2012.csv',help='CSV file containing sensor distances with three columns: [from, to, distance].')
        parser.add_argument('--normalized_k', type=float, default=0.1,help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
        parser.add_argument("--split_ratio", type=float, default=0.8)      
        parser.add_argument("--normalize", type=bool, default=True)
        return parser

    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
            noise=self.noise,
            noise_ratio=self.noise_ratio,
            noise_sever=self.noise_sever,
            noise_ratio_node=self.noise_ratio_node,
            noise_type=self.noise_type,
            noise_ratio_test=self.noise_ratio_test,
            noise_ratio_node_test=self.noise_ratio_node_test,
            noise_test=self.noise_test,
            adj=self._adj
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj
