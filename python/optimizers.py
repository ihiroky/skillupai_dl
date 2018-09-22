import numpy as np


class SGD:
    """
    Stochastic Gradient Decent
    """
    def __init__(self, lr=0.01):
        """
        コンストラクタ
        :param lr: 学習率
        """
        self.lr = lr

    def next_iteration(self):
        pass

    def update(self, key, param, grad):
        """
        パラメータ更新
        :param key: 未使用
        :param param: パラメータ
        :param grad: パラメータの微分値
        :return: 更新後のパラメータ
        """
        return param - self.lr * grad


class RMSProp:
    """
    RMSProp
    """
    def __init__(self, lr=0.01, rho=0.9):
        """
        コンストラクタ
        :param lr: 学習率
        :param rho: 減衰率
        """
        self.lr = lr
        self.rho = rho

        self.epsilon = 1e-6  # 0除算防止
        self.h = {}

    def next_iteration(self):
        pass

    def update(self, key, param, grad):
        """
        パラメータ更新
        :param key: パラメータを位置に判別する識別子
        :param param: パラメータ
        :param grad: パラメータの微分値
        :return: 更新後のパラメータ
        """

        h = self.h.get(key)
        if h is None:
            h = np.zeros_like(param)

        # h = self.rho * self.h[key] + (1 - self.rho) * (grad * grad)
        h = self.rho * h + (1 - self.rho) * (grad * grad)
        param -= self.lr * grad / np.sqrt(h + self.epsilon)
        self.h[key] = h
        return param


class Adam:

    def __init__(self, lr=0.001, rho1=0.9, rho2=0.999):
        self.lr = lr
        self.rho1 = rho1
        self.rho2 = rho2
        self.t = 0
        self.m = {}
        self.v = {}
        self.epsilon = 1e-8

    def next_iteration(self):
        self.t += 1

    def update(self, key, param, grad):
        m = self.m.get(key)
        if m is None:
            m = np.zeros_like(param)
        v = self.v.get(key)
        if v is None:
            v = np.zeros_like(param)

        m = self.rho1 * m + (1 - self.rho1) * grad
        v = self.rho2 * v + (1 - self.rho2) * (grad ** 2)
        m_adjusted = m / (1 - self.rho1 ** self.t)
        v_adjusted = v / (1 - self.rho2 ** self.t)
        param -= self.lr * m_adjusted / (np.sqrt(v_adjusted) + self.epsilon)

        self.m[key] = m
        self.v[key] = v

        return param


class Momentum:

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def next_iteration(self):
        pass

    def update(self, key, param, grad):
        v = self.v.get(key)
        if v is None:
            v = np.zeros_like(param)

        v = self.momentum * v - self.lr * grad
        param += v

        self.v[key] = v

        return param
