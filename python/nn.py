#
# このスクリプトを実行すると学習開始
# 学習データは '~/skillupai/DAY1_vr2_1_0/4_kadai/1_data/train' にある想定
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import copy
import glob
import os
import pickle
from layers import Affine, ReLU, SoftmaxWithLoss, BatchNormalization
from optimizers import SGD


def numerical_diff(f, x):
    """
    数値微分
    :param f: 関数
    :param x: 微分する点
    :return: xにおけるfの微分値
    """

    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val

        it.iternext()

    return grad


class BaseNet:
    def __init__(self, layers, last_layer, weight_decay_lambda=0.0):
        self.layers = layers
        self.last_layer = last_layer
        self.weight_decay_lambda = weight_decay_lambda

    def predict(self, x, train_flg=False):
        """
        推論関数
        :param x: 入力データ
        :param train_flg: 訓練時はTrue, 推論時はFalse
        :return: スコア
        """
        for layer in self.layers:
            x = layer.forward(x, train_flg)
        return x

    def loss(self, x, t, train_flg=False):
        """
        誤差関数
        :param x: 入力値
        :param t: 正解ラベル
        :param train_flg: 訓練時はTrue, 推論時はFalse
        :return: 誤差
        """
        y = self.predict(x, train_flg)

        weight_square_sum = 0
        for layer in self.layers:
            weight_square_sum += layer.weight_square_sum()
        weight_decay = 0.5 * self.weight_decay_lambda * weight_square_sum

        return self.last_layer.forward(y, t) + weight_decay

    def gradient(self, x, t):
        """
        微分
        重み・バイアスの微分値は各レイヤに保持
        :param x: 入力値
        :param t: 正解ラベル
        :return: 入力値に対する微分
        """

        # 順伝播
        self.loss(x, t, True)

        # 逆伝播
        dout = self.last_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout, self.weight_decay_lambda)

        return dout

    def accuracy(self, x, t):
        """
        精度
        :param x: 入力値
        :param t: 正解ラベル
        :return: 精度
        """

        y = self.predict(x)
        y = np.argmax(y, axis=1)  # one_hot -> label index
        if t.ndim != 1:
            # one hot なら label index へ変換
            t = np.argmax(t, axis=1)
        return float(np.sum(y == t)) / x.shape[0]

    def train(self, train_data, train_label, test_data, test_label, epochs=50, batch_size=20, optimizer=SGD(),
              log=True, save_model_per_epoch=False):
        """
        学習
        学習時の誤差・精度をグラフ表示し、重み・バイアスをファイルへ出力する
        :param train_data: 学習データ
        :param train_label: 学習データに対するラベル
        :param test_data: テストデータ
        :param test_label: テストデータに対するラベル
        :param epochs: エポック数
        :param batch_size: ミニバッチのサイズ
        :param optimizer: 最適化方法
        :param log: 途中経過を表示する場合はTrue
        :param save_model_per_epoch:
        :return:
        """
        input_size = train_data.shape[0]
        iterations = np.ceil(input_size / batch_size).astype(np.int)
        validate_test = test_data is not None and test_label is not None

        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []

        for e in range(epochs):
            if log:
                print('Epoch: {}'.format(e), end='')
            idx = np.arange(input_size)
            np.random.shuffle(idx)

            for i in range(iterations):
                mask = idx[batch_size * i:batch_size * (i + 1)]
                dmb = train_data[mask]
                lmb = train_label[mask]
                self.gradient(dmb, lmb)
                optimizer.next_iteration()
                for layer in self.layers:
                    layer.update(optimizer)

            loss = self.loss(train_data, train_label)
            accuracy = self.accuracy(train_data, train_label)
            train_loss.append(loss)
            train_accuracy.append(accuracy)
            if log:
                print(' loss:{:.6f}, accuracy:{:.3f}'.format(loss, accuracy))
            if validate_test:
                test_loss.append(self.loss(test_data, test_label))
                test_accuracy.append(self.accuracy(test_data, test_label))

            if save_model_per_epoch:
                model_file = self.save_params()
                epoch_model_file = 'epoch_{}_{}'.format(e, model_file)
                os.rename(model_file, epoch_model_file)
                print('Save model into {}'.format(epoch_model_file))

        if validate_test:
            def plot(dic, ylabel):
                df = pd.DataFrame(dic)
                df.plot()
                plt.xlabel('epochs')
                plt.ylabel(ylabel)
                plt.show()
            plot({'train_loss': train_loss, 'test_loss': test_loss}, 'loss')
            plot({'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}, 'accuracy')

    def train_file(self, train_data, train_label, test_data, test_label, epochs=50, batch_size=20, optimizer=SGD(),
                   log=True, save_model_per_epoch=False):
        """
        学習
        学習時の誤差・精度をグラフ表示し、重み・バイアスをファイルへ出力する
        :param train_data: 学習データ
        :param train_label: 学習データに対するラベル
        :param test_data: テストデータ
        :param test_label: テストデータに対するラベル
        :param epochs: エポック数
        :param batch_size: ミニバッチのサイズ
        :param optimizer: 最適化方法
        :param log: 途中経過を表示する場合はTrue
        :param save_model_per_epoch:
        :return:
        """
        input_size = train_data.shape[0]
        iterations = np.ceil(input_size / batch_size).astype(np.int)
        validate_test = test_data is not None and test_label is not None

        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []

        for e in range(epochs):
            idx = np.arange(input_size)
            np.random.shuffle(idx)

            for i in range(iterations):
                mask = idx[batch_size * i:batch_size * (i + 1)]
                dmb = np.array([np.load(path) for path in train_data[mask]])
                lmb = np.array([np.load(path) for path in train_label[mask]])
                self.gradient(dmb, lmb)
                optimizer.next_iteration()
                for layer in self.layers:
                    layer.update(optimizer)

            # 全部乗らないので先頭2000個
            train_mask_data = np.array([np.load(path) for path in train_data[0:2000]])
            train_mask_label = np.array([np.load(path) for path in train_label[0:2000]])
            loss = self.loss(train_mask_data, train_mask_label)
            accuracy = self.accuracy(train_mask_data, train_mask_label)
            train_loss.append(loss)
            train_accuracy.append(accuracy)

            if log:
                print('Epoch: {:03d} loss:{:.15f}, accuracy:{:.3f}'.format(e, loss, accuracy))
            else:
                print('.', end='')

            if validate_test:
                test_data_np = np.array([np.load(path) for path in test_data])
                test_label_np = np.array([np.load(path) for path in test_label])
                test_loss.append(self.loss(test_data_np, test_label_np))
                test_accuracy.append(self.accuracy(test_data_np, test_label_np))

            if save_model_per_epoch:
                model_file = self.save_params()
                epoch_model_file = 'epoch_{}_{}'.format(e, model_file)
                os.rename(model_file, epoch_model_file)
                print('Save model into {}'.format(epoch_model_file))

        if validate_test:
            def plot(dic, ylabel):
                df = pd.DataFrame(dic)
                df.plot()
                plt.xlabel('epochs')
                plt.ylabel(ylabel)
                plt.show()
            plot({'train_loss': train_loss, 'test_loss': test_loss}, 'loss')
            plot({'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}, 'accuracy')

    def cross_validation(self, train_set_list, test_set_list, optimizer, epochs, batch_size):
        """
        交差検証
        :param train_set_list: 訓練データセットのリスト
        :param test_set_list: テストデータセットのリスト
        :param optimizer: オプティマイザ
        :param epochs: エポック数
        :param batch_size: バッチサイズ
        :return:
        """
        initial_layers = self.layers
        initial_last_layer = self.last_layer
        loss_list = []
        accuracy_list = []
        for i, (train_set, test_set) in enumerate(zip(train_set_list, test_set_list)):
            train_data, train_label = train_set
            test_data, test_label = test_set
            self.layers = copy.deepcopy(initial_layers)
            self.last_layer = copy.deepcopy(initial_last_layer)
            self.train(train_data, train_label, test_data, test_label, epochs, batch_size, optimizer, False)
            loss = self.loss(test_data, test_label)
            accuracy = self.accuracy(test_data, test_label)
            print('Dataset {} - loss: {}'.format(i, loss))
            print('Dataset {} - accuracy: {}'.format(i, accuracy))
            loss_list.append(loss)
            accuracy_list.append(accuracy)

        print('Average loss: {}'.format(np.average(loss_list)))
        print('Average accuracy: {}'.format(np.average(accuracy_list)))

    def cross_validation_file(self, train_set_list, test_set_list, optimizer, epochs, batch_size):
        """
        交差検証
        :param train_set_list: 訓練データセットのリスト
        :param test_set_list: テストデータセットのリスト
        :param optimizer: オプティマイザ
        :param epochs: エポック数
        :param batch_size: バッチサイズ
        :return:
        """
        initial_layers = self.layers
        initial_last_layer = self.last_layer
        loss_list = []
        accuracy_list = []
        for i, (train_set, test_set) in enumerate(zip(train_set_list, test_set_list)):
            train_data, train_label = train_set
            test_data, test_label = test_set
            self.layers = copy.deepcopy(initial_layers)
            self.last_layer = copy.deepcopy(initial_last_layer)
            self.train_file(train_data, train_label, test_data, test_label, epochs, batch_size, optimizer, False)
            test_data_np = np.array([np.load(p) for p in test_data])
            test_label_np = np.array([np.load(p) for p in test_label])
            loss = self.loss(test_data_np, test_label_np)
            accuracy = self.accuracy(test_data_np, test_label_np)
            print('')
            print('Dataset {} - loss: {}'.format(i, loss))
            print('Dataset {} - accuracy: {}'.format(i, accuracy))
            loss_list.append(loss)
            accuracy_list.append(accuracy)

        print('Average loss: {}'.format(np.average(loss_list)))
        print('Average accuracy: {}'.format(np.average(accuracy_list)))

    def save_params(self):
        return ''


class MultiLayerNet(BaseNet):
    """
    多層ニューラルネットワーク
    """
    def __init__(self, layer_sizes, batch_normalization=True, weight_init_std=0.01, last_layer=SoftmaxWithLoss()):
        """
        コンストラクタ
        :param layer_sizes: 入力層、中間層１、…、中間層N、出力層のサイズ
        :param weight_init_std: 重み初期化時に使う正規分布の標準偏差
        """
        layers = []
        n = len(layer_sizes)
        for i in range(n-1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            w = weight_init_std * np.random.randn(in_size, out_size)
            b = np.zeros(out_size)
            layers.append(Affine(w, b))
            print('Affine Layer {}: {}'.format(i, w.shape))
            if batch_normalization:
                gamma = np.ones(out_size)
                beta = np.zeros(out_size)
                layers.append(BatchNormalization(gamma, beta))
                print('Batch Normalization Layer {}'.format(i))
            if i != n - 2:  # 最後は last_layer
                layers.append(ReLU())
                print('ReLU Layer {}'.format(i))

        super().__init__(layers, last_layer)
        self.layer_sizes = layer_sizes
        self.batch_normalization = batch_normalization

    def save_params(self):
        save_path = datetime.now().strftime('params_%Y%m%d-%H%M%S.dat')
        params = [self.layer_sizes, self.batch_normalization]
        for layer in self.layers:
            params.append(layer.params())
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
        print('Parameters are saved into {}.'.format(save_path))
        return save_path


def load_net(path) -> MultiLayerNet:
    """
    保存済みモデルからニューラルネットワークを復元する
    :param path:保存済みモデルへのパス
    :return: 復元されたモデル
    """
    print('Load {}'.format(path))
    with open(path, 'rb') as f:
        params = pickle.load(f)
    layer_sizes = params[0]
    batch_normalization = params[1]
    net = MultiLayerNet(layer_sizes, batch_normalization)

    i = 2
    for layer in net.layers:
        layer.set_params(params[i])
        i += 1

    return net


def load_data(dir_path):
    """
    データセット読み込み
    :param dir_path: データセットが格納されたディレクトリへのパス
    :return: データセット
    """
    aiueo_to_num = {
        'a': 0,
        'i': 1,
        'u': 2,
        'e': 3,
        'o': 4,
    }
    data = []
    label = []
    li_fpath = glob.glob(os.path.join(dir_path, "*", "*.png"))
    for i, p in enumerate(li_fpath):
        # データ
        img = Image.open(p)
        img = np.array(img).astype(np.float32) / 255
        img = img.reshape(-1)
        data.append(img)

        # ラベル
        path_elements = p.split(os.path.sep)
        label_str = path_elements[3]  # a,i,u,e,o
        label_num = aiueo_to_num[label_str]
        lbl = np.zeros(len(aiueo_to_num))
        lbl[label_num] = 1  # one hot
        label.append(lbl)

    # シャッフル
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    np_data = np.array(data)[idx]
    np_label = np.array(label)[idx]

    return np_data, np_label


def main():
    """
    学習用main関数
    :return:
    """

    np.random.seed(12345)

    # データセット作成
    cwd = os.getcwd()
    wd = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/2_notebook/')
    os.chdir(wd)
    data, label = load_data('../1_data/train')
    data_size = data.shape[0]
    train_size = int(data_size * 0.7)
    dev_size = (data_size - train_size) // 2

    train_data = data[:train_size]
    train_label = label[:train_size]
    dev_data = data[train_size:train_size+dev_size]
    dev_label = label[train_size:train_size+dev_size]
    test_data = data[train_size+dev_size:]
    test_label = label[train_size+dev_size:]
    print('train:{}, dev:{}, test:{}'.format(len(train_data), len(dev_data), len(test_data)))

    os.chdir(cwd)

    optimizer = SGD()
    # optimizer = RMSProp()
    net = MultiLayerNet((784, 50, 5))
    epochs = 20
    batch_size = 5
    # net = MultiLayerNN((784, 100, 10, 5))
    # epochs = 60
    # batch_size = 10

    # パラメータ調整用
    # net.train(train_data, train_label, dev_data, dev_label, epochs, batch_size, optimizer)

    # 評価
    net.train(train_data, train_label, test_data, test_label, epochs, batch_size, optimizer)
    model_file = net.save_params()
    print('Loss: {}'.format(net.loss(test_data, test_label)))
    print('Accuracy: {}'.format(net.accuracy(test_data, test_label)))

    net0 = load_net(model_file)
    print('Loss: {}'.format(net0.loss(data, label)))
    print('Accuracy: {}'.format(net0.accuracy(data, label)))


if __name__ == '__main__':
    main()

