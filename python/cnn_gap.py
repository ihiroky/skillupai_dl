#
# このスクリプトを実行すると学習開始
# 学習データは '~/skillupai/DAY1_vr2_1_0/4_kadai/1_data/train' にある想定
#
import numpy as np
from datetime import datetime
from time import time
import os
import pickle
from nn import BaseNet
from layers import Affine, ReLU, BatchNormalization, SoftmaxWithLoss, GlobalAveragePooling
from optimizers import RMSProp, Adam
from data import load_data, create_cross_validation_data, load_data_path, create_cross_validation_data_path
from cnn import ConvNet


class ConvGapNet(BaseNet):
    """
    CNN with Global Average Pooling
    (Convolution -> BatchNormalization -> ReLU [-> MaxPooling]) * n
      -> GlobalAveragePooling -> BatchNormalization -> ReLU -> Affine -> Softmax
    """

    def __init__(self, input_shape, output_size, params_list, weight_decay_lambda=0.0):
        layers = []
        next_input_shape = input_shape
        for params in params_list:
            next_input_shape = ConvNet.append_conv_pool(layers, next_input_shape, params)

        gap = GlobalAveragePooling()
        layers.append(gap)
        gap_output_size = next_input_shape[0]  # channel
        gamma = np.ones(gap_output_size)
        beta = np.zeros(gap_output_size)
        bn = BatchNormalization(gamma, beta)
        layers.append(bn)
        layers.append(ReLU())

        a1_input_size = gap_output_size
        a1_w = np.random.rand(a1_input_size, output_size) * np.sqrt(2 / a1_input_size)  # He initialization
        a1_b = np.zeros(output_size)
        layers.append(Affine(a1_w, a1_b))

        super().__init__(layers, SoftmaxWithLoss(), weight_decay_lambda)
        self.input_shape = input_shape
        self.output_size = output_size
        self.params_list = params_list

    def save_params(self):
        save_path = datetime.now().strftime('convgapnet_params_%Y%m%d-%H%M%S.dat')
        params = [self.input_shape, self.output_size, self.params_list, self.weight_decay_lambda]
        for layer in self.layers:
            params.append(layer.params())
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
        return save_path


def load_conv_net(path) -> ConvGapNet:
    """
    保存済みモデルからニューラルネットワークを復元する
    :param path:保存済みモデルへのパス
    :return: 復元されたモデル
    """
    print('Load {}'.format(path))
    with open(path, 'rb') as f:
        params = pickle.load(f)
    net = ConvGapNet(params[0], params[1], params[2], params[3])

    i = 4
    for layer in net.layers:
        layer.set_params(params[i])
        i += 1

    return net


def main():
    """
    学習用main関数
    :return:
    """

    np.random.seed(12345)

    optimizer = Adam(0.001)
    weight_decay_lambda = 0.00001
    params_list = [
        {'fn': 16, 'fh': 5, 'fw': 5, 'cs': 1, 'cp': 0, 'ph': 2, 'pw': 2, 'ps': 2, 'pp': 0},  # 28-> 24 -> 12
        {'fn': 32, 'fh': 3, 'fw': 3, 'cs': 1, 'cp': 0, 'ph': 0, 'pw': 0, 'ps': 0, 'pp': 0},  # 12 -> 10
        {'fn': 32, 'fh': 3, 'fw': 3, 'cs': 1, 'cp': 0, 'ph': 2, 'pw': 2, 'ps': 2, 'pp': 0},  # 10 -> 8 -> 4
    ]
    net = ConvGapNet(input_shape=(1, 28, 28), output_size=5, params_list=params_list,
                     weight_decay_lambda=weight_decay_lambda)
    epochs = 100
    batch_size = 30

    ''' データセット作成 '''
    # data_root = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/1_data/train')
    # data, label = load_data(data_root, True)
    # print('The number of data:{}'.format(len(data)))
    ''' パラメータ調整用 '''
    # train_set_list, test_set_list = create_cross_validation_data(data, label, 10)
    # start_time = time()
    # net.cross_validation(train_set_list, test_set_list, optimizer, epochs, batch_size, net.train_file)
    # print('Elapsed: {:.6f}'.format(time() - start_time))
    ''' モデル生成 '''
    # net.train(data, label, None, None, epochs, batch_size, optimizer)
    # model_file = net.save_params()
    # print('The model was saved into {}.'.format(model_file))

    ''' データセット作成 '''
    data_root = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/1_data/train')
    data_path_list, label_path_list = load_data_path(data_root, './tmp')
    print('The number of data:{}'.format(len(data_path_list)))
    ''' パラメータ調整用 '''
    # train_set_list, test_set_list = create_cross_validation_data(data_path_list, label_path_list, 10)
    # start_time = time()
    # net.cross_validation_file(train_set_list, test_set_list, optimizer, epochs, batch_size)
    # print('Elapsed: {:.6f}'.format(time() - start_time))
    ''' モデル生成ファイル読み込み版 '''
    start = time()
    net.train_file(data_path_list, label_path_list, None, None, epochs, batch_size, optimizer)
    print('Elapsed: {} sec.'.format(time() - start))
    model_file = net.save_params()
    print('The model was saved into {}.'.format(model_file))


if __name__ == '__main__':
    main()

