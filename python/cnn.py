#
# このスクリプトを実行すると学習開始
# 学習データは '~/skillupai/DAY1_vr2_1_0/4_kadai/1_data/train' にある想定
#
import numpy as np
from datetime import datetime
import os
import pickle
from nn import BaseNet
from layers import Affine, ReLU, SoftmaxWithLoss, BatchNormalization, Convolution, MaxPooling
from optimizers import SGD, RMSProp
from data import load_data, create_cross_validation_data


class ConvNet(BaseNet):
    """
    CNN
    (Convolution -> BatchNormalization -> ReLU -> MaxPooling) * n -> Affine -> Softmax
    """

    def __init__(self, input_shape, output_size, params_list):
        layers = []
        next_input_shape = input_shape
        for params in params_list:
            next_input_shape = ConvNet.append_conv_pool(layers, next_input_shape, params)

        a_input_size = np.array(next_input_shape).prod()
        a_w = np.random.rand(a_input_size, output_size) * np.sqrt(2 / a_input_size)
        a_b = np.zeros(output_size)
        layers.append(Affine(a_w, a_b))

        super().__init__(layers, SoftmaxWithLoss())
        self.input_shape = input_shape
        self.output_size = output_size
        self.params_list = params_list

    @staticmethod
    def append_conv_pool(layers, input_shape, params):
        c, h, w = input_shape
        fn, fh, fw = params['fn'], params['fh'], params['fw']
        conv_w = np.random.rand(fn, c, fh, fw) * np.sqrt(2 / (c * h * w))  # He initialization
        conv_b = np.zeros(fn)
        conv_stride = params['cs']
        conv_padding = params['cp']
        conv = Convolution(conv_w, conv_b, conv_stride, conv_padding)
        conv_out_h = ConvNet.out_size(h, fh, conv_stride, conv_padding)
        conv_out_w = ConvNet.out_size(w, fw, conv_stride, conv_padding)

        gamma = np.ones((fn, conv_out_h, conv_out_w)).flatten()
        beta = np.zeros((fn, conv_out_h, conv_out_w)).flatten()
        bn = BatchNormalization(gamma, beta)

        layers.append(conv)
        layers.append(bn)
        layers.append(ReLU())
        out_h = conv_out_h
        out_w = conv_out_w

        ph = params['ph']
        pw = params['pw']
        if ph > 0 and pw > 0:
            ps = params['ps']
            pp = params['pp']
            pool = MaxPooling(height=ph, width=pw, stride=ps, padding=pp)
            out_h = ConvNet.out_size(conv_out_h, ph, ps, pp)
            out_w = ConvNet.out_size(conv_out_w, pw, ps, pp)
            layers.append(pool)

        return fn, out_h, out_w

    @staticmethod
    def out_size(x, f, s, p):
        return (x - f + 2 * p) // s + 1

    def save_params(self):
        save_path = datetime.now().strftime('convnet_params_%Y%m%d-%H%M%S.dat')
        params = [self.input_shape, self.output_size, self.params_list]
        for layer in self.layers:
            params.append(layer.params())
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
        print('Parameters are saved into {}.'.format(save_path))
        return save_path


class ConvNet2(BaseNet):
    """
    CNN
    (Conv -> BatchNorm -> ReLU -> Conv -> BatcNorm -> ReLU -> MaxPooling) * n -> Affine -> Softmax
    """

    def __init__(self, input_shape, output_size, params_list):
        layers = []
        weight_init_std = 0.01
        next_input_shape = input_shape
        for params in params_list:
            next_input_shape = ConvNet2.append_conv_conv_pool(layers, weight_init_std, next_input_shape, params)

        a_input_size = np.array(next_input_shape).prod()
        a_w = weight_init_std * np.random.rand(a_input_size, output_size)
        a_b = np.zeros(output_size)
        layers.append(Affine(a_w, a_b))

        super().__init__(layers, SoftmaxWithLoss())
        self.input_shape = input_shape
        self.output_size = output_size
        self.params_list = params_list

    @staticmethod
    def append_conv_conv_pool(layers, weight_init_std, input_shape, params):
        conv_list = []
        bn_list = []
        for i in range(params['convs']):
            def key(s):
                return s + str(i)

            c, h, w = input_shape
            fn, fh, fw = params[key('fn')], params[key('fh')], params[key('fw')]
            conv_w = weight_init_std * np.random.rand(fn, c, fh, fw)
            conv_b = np.zeros(fn)
            conv_stride = params[key('cs')]
            conv_padding = params[key('cp')]
            conv = Convolution(conv_w, conv_b, conv_stride, conv_padding)
            conv_out_h = ConvNet.out_size(h, fh, conv_stride, conv_padding)
            conv_out_w = ConvNet.out_size(w, fw, conv_stride, conv_padding)
            input_shape = fn, conv_out_h, conv_out_w
            conv_list.append(conv)

            gamma = np.ones(input_shape).flatten()
            beta = np.zeros(input_shape).flatten()
            bn = BatchNormalization(gamma, beta)
            bn_list.append(bn)

        ph = params['ph']
        pw = params['pw']
        ps = params['ps']
        pp = params['pp']
        pool = MaxPooling(height=ph, width=pw, stride=ps, padding=pp)
        pool_out_h = ConvNet.out_size(input_shape[1], ph, ps, pp)
        pool_out_w = ConvNet.out_size(input_shape[2], pw, ps, pp)

        for conv, bn in zip(conv_list, bn_list):
            layers.append(conv)
            layers.append(bn)
            layers.append(ReLU())
        layers.append(pool)

        return fn, pool_out_h, pool_out_w

    def save_params(self):
        save_path = datetime.now().strftime('convnet2_params_%Y%m%d-%H%M%S.dat')
        params = [self.input_shape, self.output_size, self.params_list]
        for layer in self.layers:
            params.append(layer.params())
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
        print('Parameters are saved into {}.'.format(save_path))
        return save_path


def load_conv_net(path) -> ConvNet:
    """
    保存済みモデルからニューラルネットワークを復元する
    :param path:保存済みモデルへのパス
    :return: 復元されたモデル
    """
    print('Load {}'.format(path))
    with open(path, 'rb') as f:
        params = pickle.load(f)
    net = ConvNet(params[0], params[1], params[2])

    i = 3
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

    # データセット作成
    data_root = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/1_data/train')
    data, label = load_data(data_root, True)
    print('The number of data:{}'.format(len(data)))

    # optimizer = SGD(0.01)
    optimizer = RMSProp(0.01)
    params_list = [
        {'fn': 16, 'fh': 3, 'fw': 3, 'cs': 1, 'cp': 1, 'ph': 4, 'pw': 4, 'ps': 4, 'pp': 0},
        {'fn': 32, 'fh': 3, 'fw': 3, 'cs': 1, 'cp': 1, 'ph': 2, 'pw': 2, 'ps': 2, 'pp': 0},
    ]
    net = ConvNet(input_shape=(1, 28, 28), output_size=5, params_list=params_list)
    '''
    params_list = [
        {
            'convs': 2,
            'fn0': 16, 'fh0': 3, 'fw0': 3, 'cs0': 1, 'cp0': 1,
            'fn1': 16, 'fh1': 3, 'fw1': 3, 'cs1': 1, 'cp1': 1,
            'ph': 4, 'pw': 4, 'ps': 4, 'pp': 0
        },
    ]
    net = ConvNet2(input_shape=(1, 28, 28), output_size=5, params_list=params_list)
    '''
    epochs = 20
    batch_size = 20

    # パラメータ調整用
    # train_set_list, test_set_list = create_cross_validation_data(data, label, 5)
    # net.cross_validation(train_set_list, test_set_list, optimizer, epochs, batch_size)

    # モデル生成
    net.train(data, label, None, None, epochs, batch_size, optimizer)
    model_file = net.save_params()
    print('The model was saved into {}.'.format(model_file))


if __name__ == '__main__':
    main()

