import numpy as np


class Affine:
    """
    Affineレイヤ
    """

    def __init__(self, w, b):
        """
        コンストラクタ
        :param w: 重み行列
        :param b: バイアス
        """

        self.w = w
        self.b = b

        self.x = None   # 逆伝播時計算用
        self.dw = None  # dL/dw
        self.db = None  # dL/db
        self.input_shape = None  # 入力値のshape

    def __str__(self):
        return 'Affine [{}, {}]'.format(*self.w.shape);

    def forward(self, x, train_flg):
        """
        順伝播
        :param x: 入力データ
        :param train_flg: 未使用
        :return: 出力データ
        """

        self.input_shape = x.shape
        if x.ndim != 2:
            # (n, c, h, w) に備えて2次元にする
            x = np.reshape(x, (x.shape[0], -1))

        # 逆伝播時に利用
        self.x = x

        return np.dot(x, self.w) + self.b

    def backward(self, dout, weight_decay_lambda):
        """
        逆伝播
        :param dout: 入力勾配
        :param weight_decay_lambda:
        :return: 出力勾配
        """
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout) + weight_decay_lambda * self.w
        self.db = np.sum(dout, axis=0)

        # (n, c, h, w) に備えて入力値のshapeに戻す
        dx = np.reshape(dx, self.input_shape)

        return dx

    def weight_square_sum(self):
        return np.sum(self.w ** 2)

    def update(self, optimizer):
        """
        パラーメタ更新
        :param optimizer: オプティマイザ
        :return:
        """
        self.w = optimizer.update((self, 'w'), self.w, self.dw)
        self.b = optimizer.update((self, 'b'), self.b, self.db)

    def params(self):
        return self.w, self.b

    def set_params(self, params):
        self.w = params[0]
        self.b = params[1]


class ReLU:
    """
    ReLUレイヤ
    """
    def __init__(self):
        """
        コンストラクタ
        """

        # 入力値が負になる部分を表すマスク
        self.mask = None

    def __str__(self):
        return 'ReLU'

    def forward(self, x, train_flg):
        """
        順伝播
        :param x: 入力データ
        :param train_flg: 未使用
        :return: 出力データ
        """

        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0  # 負値要素を0にする
        return out

    def backward(self, dout, weight_decay_lambda):
        """
        逆伝播
        :param dout: 入力勾配
        :param weight_decay_lambda:
        :return: 出力勾配
        """

        # 順伝播時負だった要素は逆伝播しない
        dout[self.mask] = 0
        return dout

    def weight_square_sum(self):
        return 0

    def update(self, optimizer):
        """
        パラーメタ更新
        何もしない
        :param optimizer: オプティマイザ
        :return:
        """
        pass

    def params(self):
        pass

    def set_params(self, params):
        pass


class SoftmaxWithLoss:
    """
    Softmaxレイヤ
    """
    def __init__(self):
        """
        コンストラクタ
        """

        # 逆伝播計算用
        self.t = None
        self.y = None

    def __str__(self):
        return 'Softmax'

    @staticmethod
    def softmax(x):
        x -= np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def cross_entropy_error(y, t):
        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)

        delta = 1e-7  # log(0)防止
        n = y.shape[0]
        return - np.sum(t * np.log(y + delta)) / n

    def forward(self, x, t):
        """
        順伝播
        :param x: 入力データ
        :return: 出力データ
        """

        self.t = t
        self.y = SoftmaxWithLoss.softmax(x)
        return SoftmaxWithLoss.cross_entropy_error(self.y, t)

    def backward(self, dout=1):
        """
        逆伝播
        :param dout: 入力勾配
        :return: 出力勾配
        """

        n = self.y.shape[0]
        return (self.y - self.t) / n


class Dropout:
    """
    ドロップアウトレイヤ
    """
    def __init__(self, dropout_ratio=0.5):
        """
        コンストラクタ
        :param dropout_ratio: ドロップアウトするノードの割合
        """
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def __str__(self):
        return 'Dropout'

    def forward(self, x, train_flg=True):
        """
        順伝播
        :param x: 入力値
        :param train_flg: 学習時はTrue, 推論時はFalse
        :return: 出力値
        """
        if train_flg:
            # ドロップアウトしないノードは伝播させる
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # ドロップアウトしていないノードの割合をかけて出力を調整
            return (1 - self.dropout_ratio) * x

    def backward(self, dout, weight_decay_lambda):
        """
        逆伝播
        :param dout: 入力勾配
        :param weight_decay_lambda:
        :return: 出力勾配
        """
        # 順伝播させたノードだけ逆伝播させる
        return dout * self.mask

    def weight_square_sum(self):
        return 0

    def update(self, optimizer):
        """
        パラーメタ更新
        何もしない
        :param optimizer: オプティマイザ
        :return:
        """
        pass

    def params(self):
        pass

    def set_params(self, params):
        pass


class BatchNormalization:
    """
    バッチ正規化
    """
    def __init__(self, gamma, beta, rho=0.9):
        """
        コンストラクタ
        :param gamma: 正規化後のスケール量
        :param beta: 正規化後のシフト量
        :param rho: 平均・分散移動平均の減衰率
        """
        self.gamma = gamma
        self.beta = beta
        self.rho = rho

        self.moving_mean = None
        self.moving_var = None
        self.batch_size = None
        self.mu = None
        self.x_minus_mu = None
        self.std = None
        self.x_std = None
        self.dgamma = None
        self.dbeta = None

        self.input_shape = None

    def __str__(self):
        return 'BatchNormalization [{}]'.format(gamma.shape)

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            x = np.reshape(x, (x.shape[0], -1))

        out = self.__forward(x, train_flg)

        out = np.reshape(out, self.input_shape)
        return out

    def __forward(self, x, train_flg=True, epsilon=1e-8):
        """
        順伝播
        :param x: 入力データ
        :param train_flg: 訓練時はTrue, 推論時はFalse
        :param epsilon: ゼロ除算防止用定数
        :return:
        """
        if (self.moving_mean is None) or (self.moving_var is None):
            out_size = x.shape[1]
            self.moving_mean = np.zeros(out_size)
            self.moving_var = np.zeros(out_size)

        if train_flg:
            mu = np.mean(x, axis=0)
            x_minus_mu = x - mu
            var = np.mean(x_minus_mu ** 2, axis=0)
            std = np.sqrt(var + epsilon)
            x_std = x_minus_mu / std  # 正規化された入力値

            self.batch_size = x.shape[0]
            self.mu = mu
            self.x_minus_mu = x_minus_mu
            self.std = std
            self.x_std = x_std
            self.moving_mean = self.rho * self.moving_mean + (1 - self.rho) * mu
            self.moving_var = self.rho * self.moving_var + (1 - self.rho) * var
        else:
            # 推論時は、学習時の平均・分散の移動平均を使う
            x_std = (x - self.moving_mean) / np.sqrt(self.moving_var + epsilon)

        return self.gamma * x_std + self.beta

    def backward(self, dout, weight_decay_lambda):
        if dout.ndim != 2:
            dout = np.reshape(dout, (dout.shape[0], -1))

        dx = self.__backward(dout)

        dx = np.reshape(dx, self.input_shape)
        return dx

    def __backward(self, dout):
        """
        逆伝播
        :param dout: 入力勾配
        :return: 出力勾配
        """
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * self.x_std, axis=0)
        dx_std = self.gamma * dout
        dx_minus_mu = dx_std / self.std
        dstd_inv = np.sum(dx_std * self.x_minus_mu, axis=0)
        dstd = - dstd_inv / self.std ** 2
        dsqrt_std = dstd / (2 * self.std)
        dvar_avg = dsqrt_std / self.batch_size
        dvar_twice = 2 * self.x_minus_mu * dvar_avg
        dx_minus_mean_left = dx_minus_mu + dvar_twice
        dx_minus_mean_right = - (dx_minus_mu + dvar_twice)
        dx_mean = np.sum(dx_minus_mean_right, axis=0) / self.batch_size
        dx = dx_minus_mean_left + dx_mean

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx

    def weight_square_sum(self):
        return 0

    def update(self, optimizer):
        """
        パラーメタ更新
        :param optimizer: オプティマイザ
        :return:
        """
        self.gamma = optimizer.update((self, 'g'), self.gamma, self.dgamma)
        self.beta = optimizer.update((self, 'b'), self.beta, self.dbeta)

    def params(self):
        # 推論時にmoving_mean, moving_varも必要
        return self.gamma, self.beta, self.moving_mean, self.moving_var

    def set_params(self, params):
        self.gamma = params[0]
        self.beta = params[1]
        self.moving_mean = params[2]
        self.moving_var = params[3]


def im2col(im, filter_height, filter_width, stride=1, padding=0, padding_value=0):
    """
    ミニバッチイメージを行列に変換
    :param im: 入力データ
    :param filter_height: フィルタ高
    :param filter_width: フィルタ幅
    :param stride: ストライド
    :param padding: パディングサイズ
    :param padding_value: パディング値
    :return: 行列化された入力データ
    """

    n, c, h, w = im.shape
    output_h = (h - filter_height + 2 * padding) // stride + 1
    output_w = (w - filter_width + 2 * padding) // stride + 1
    padded_im = np.pad(im, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                      'constant', constant_values=padding_value)

    col = np.zeros((n, c, filter_height, filter_width, output_h, output_w))
    for fh in range(filter_height):
        fh_end = fh + output_h * stride
        for fw in range(filter_width):
            fw_end = fw + output_w * stride
            # フィルタの各要素毎に、stride間隔の要素コピーを一括で行う
            col[:, :, fh, fw, :, :] = padded_im[:, :, fh:fh_end:stride, fw:fw_end:stride]

    # データ数, チャネル数, フィルタ高, フィルタ幅, 出力高, 出力幅
    #  -> データ数, 出力高, 出力幅, チャネル数, フィルタ高, フィルタ幅
    #  -> (前3つ, 後3つ)
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(n * output_h * output_w, -1)


def col2im(col, im_shape, filter_height, filter_width, stride=1, padding=0):
    """
    行列をミニバッチイメージに変換
    :param col: ミニバッチイメージをim2colで行列に変換したデータ
    :param im_shape: ミニバッチイメージのshape
    :param filter_height: フィルタ高
    :param filter_width: フィルタ幅
    :param stride: ストライド
    :param padding: パディングサイズ
    :return:
    """

    n, c, h, w = im_shape
    output_h = (h - filter_height + 2 * padding) // stride + 1
    output_w = (w - filter_width + 2 * padding) // stride + 1
    col = col.reshape(n, output_h, output_w, c, filter_height, filter_width).transpose(0, 3, 4, 5, 1, 2)
    im = np.zeros((n, c, h + 2 * padding, w + 2 * padding))
    for fh in range(filter_height):
        fh_end = fh + output_h * stride
        for fw in range(filter_width):
            fw_end = fw + output_w * stride
            # stride間隔でのフィルタ要素コピー
            im[:, :, fh:fh_end:stride, fw:fw_end:stride] += col[:, :, fh, fw, :, :]
            # im[:, :, fh:fh_end:stride, fw:fw_end:stride] = col[:, :, fh, fw, :, :]  # im2colの逆変換

    # パディング部分除去
    return im[:, :, padding:h+padding, padding:w+padding]


class Convolution:
    def __init__(self, w, b, stride=1, padding=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.padding = padding

        self.x = None
        self.col = None
        self.col_w = None
        self.db = None
        self.dw = None

    def forward(self, x, train_flg):
        """
        順伝播
        :param x: 入力値（データ数, チャネル数, 画像高, 画像幅）
        :param train_flg: 未使用
        :return: 出力値（データ数, フィルタ数, 畳み込み後高, 畳み込み後幅), フィルタ数は後のチャネル数
        """

        fn, c, fh, fw = self.w.shape
        n, c, h, w = x.shape
        output_h = (h - fh + 2 * self.padding) // self.stride + 1
        output_w = (w - fh + 2 * self.padding) // self.stride + 1
        col = im2col(x, fh, fw, self.stride, self.padding)  # (n*oh*ow, c*fh*fw)
        col_w = self.w.reshape(fn, -1).T  # (c*fh*fw, fn)
        out = np.dot(col, col_w) + self.b  # (n*oh*ow, fn)
        out = out.reshape(n, output_h, output_w, fn).transpose(0, 3, 1, 2)  # (n, oh, ow, fn) -> (n, fn, oh, ow)

        self.x = x
        self.col = col
        self.col_w = col_w

        return out

    def backward(self, dout, weight_decay_lamba):
        """
        逆伝播
        :param dout: 入力勾配 (データ数, チャネル数（フィルタ数）, 畳み込み後高, 畳み込み後幅)
        :param weight_decay_lamba:
        :return: 出力勾配 (データ数, チャネル数, 画像高, 画像幅)
        """
        fn, c, fh, fw = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, fn)  # (n*oh*ow, fn)
        db = np.sum(dout, axis=0)
        dcol_w = np.dot(self.col.T, dout)  # (c*fh*fw, fn)
        dw = dcol_w.T.reshape(fn, c, fh, fw) + weight_decay_lamba * self.w
        dcol = np.dot(dout, self.col_w.T)  # (n*oh*ow, c*fh*fw)
        dx = col2im(dcol, self.x.shape, fh, fw, self.stride, self.padding)

        self.db = db
        self.dw = dw

        return dx

    def weight_square_sum(self):
        return np.sum(self.w ** 2)

    def update(self, optimizer):
        """
        パラーメタ更新
        :param optimizer: オプティマイザ
        :return:
        """
        self.w = optimizer.update((self, 'w'), self.w, self.dw)
        self.b = optimizer.update((self, 'b'), self.b, self.db)

    def params(self):
        return self.w, self.b, self.stride, self.padding

    def set_params(self, params):
        self.w = params[0]
        self.b = params[1]
        self.stride = params[2]
        self.padding = params[3]


class MaxPooling:
    def __init__(self, height, width, stride=1, padding=0):
        self.height = height
        self.width = width
        self.stride = stride
        self.padding = padding

        self.x = None
        self.col = None
        self.arg_max = None

    def forward(self, x, train_flg):
        """
        順伝播
        :param x: 入力値
        :param train_flg: 未使用
        :return: 出力値
        """
        n, c, h, w = x.shape
        output_h = (h - self.height + 2 * self.padding) // self.stride + 1
        output_w = (w - self.width + 2 * self.padding) // self.stride + 1
        col = im2col(x, self.height, self.width, self.stride, self.padding, -np.inf)  # (n, oh, ow, c, fh, hw)
        col = col.reshape(-1, self.height * self.width)
        arg_max = np.argmax(col, axis=1)
        col_max = np.max(col, axis=1)
        out = col_max.reshape(n, output_h, output_w, c).transpose(0, 3, 1, 2)  # (n, c, oh, ow)

        self.x = x
        self.col = col
        self.arg_max = arg_max

        return out

    def backward(self, dout, weight_decay_lambda):
        """
        逆伝播
        :param dout: 入力勾配 (データ数, チャネル数, データ高, データ幅）
        :param weight_decay_lambda:
        :return: 出力勾配 (データ数, チャネル数, pooling前データ高, pooling前データ幅）
        """
        dout = dout.transpose(0, 2, 3, 1)  # (n, oh, ow, c)
        dcol = np.zeros((dout.size, self.height * self.width))
        dcol[np.arange(dout.size), self.arg_max.flatten()] = dout.flatten()  # 最大値部分へ伝播
        dx = col2im(dcol, self.x.shape, self.height, self.width, self.stride, self.padding)
        return dx

    def weight_square_sum(self):
        return 0

    def update(self, optimizer):
        """
        パラーメタ更新　何もしない
        :param optimizer: オプティマイザ
        :return:
        """
        pass

    def params(self):
        return self.height, self.width, self.stride, self.padding

    def set_params(self, params):
        self.height = params[0]
        self.width = params[1]
        self.stride = params[2]
        self.padding = params[3]


class AveragePooling:
    def __init__(self, height, width, stride=1, padding=0):
        self.height = height
        self.width = width
        self.stride = stride
        self.padding = padding

        self.x = None
        self.col = None

    def forward(self, x, train_flg):
        n, c, h, w = x.shape
        output_h = (h - self.height + 2 * self.padding) // self.stride + 1
        output_w = (w - self.width + 2 * self.padding) // self.stride + 1
        col = im2col(x, self.height, self.width, self.stride, self.padding)  # (n, oh, ow, c, fh, fw)
        col = col.reshape(-1, self.height * self.width)
        col_avg = col.mean(axis=1)
        out = col_avg.reshape(n, output_h, output_w, c).transpose(0, 3, 1, 2)  # (n, c, oh, ow)

        self.x = x

        return out

    def backward(self, dout, weight_decay_lamba):
        dout = dout.transpose(0, 2, 3, 1)  # (n, oh, ow, c)
        dcol = np.tile(dout.reshape(-1, 1), self.height * self.width)  # 全体に伝播
        dx = col2im(dcol, self.x.shape, self.height, self.width, self.stride, self.padding)
        dx /= self.height * self.width
        return dx

    def weight_square_sum(self):
        return 0

    def update(self, optimizer):
        pass

    def params(self):
        return self.height, self.width, self.stride, self.padding

    def set_params(self, params):
        self.height = params[0]
        self.width = params[1]
        self.stride = params[2]
        self.padding = params[3]


class GlobalAveragePooling:
    def __init__(self):
        self.x = None

    def forward(self, x, train_flg):
        out = x.mean(axis=(2, 3))
        self.x = x
        return out

    def backward(self, dout, weight_decay_lambda):
        n, c, h, w = self.x.shape
        area = h * w
        dx = dout.repeat(area).reshape(n, c, h, w)
        dx /= area
        return dx

    def weight_square_sum(self):
        return 0

    def update(self, optimizer):
        pass

    def params(self):
        pass

    def set_params(self, params):
        pass
