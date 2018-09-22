import numpy as np
from PIL import Image
import os
import glob
import shutil


def load_data(dir_path, augmented=False):
    """
    データセット読み込み
    :param dir_path: データセットが格納されたディレクトリへのパス
    :param augmented: 拡張したデータセットを読む場合はTrue
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
    if augmented:
        li_fpath += glob.glob(os.path.join(dir_path, "*", "*", "*.png"))

    for i, p in enumerate(li_fpath):
        # データ
        img = Image.open(p)
        img = np.array(img).astype(np.float32) / 255
        img = img.reshape(1, img.shape[0], img.shape[1])
        data.append(img)

        # ラベル
        path_elements = p.split(os.path.sep)
        label_str = path_elements[-1][0]  # a,i,u,e,o
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


def load_data_path(dir_path, tmp_dir):
    """
    データセット読み込み
    :param dir_path: データセットが格納されたディレクトリへのパス
    :param augmented: 拡張したデータセットを読む場合はTrue
    :return: データへのパス
    """

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    aiueo_to_num = {
        'a': 0,
        'i': 1,
        'u': 2,
        'e': 3,
        'o': 4,
    }
    li_fpath = glob.glob(os.path.join(dir_path, "*", "*.png"))
    li_fpath += glob.glob(os.path.join(dir_path, "*", "*", "*.png"))

    image_path_list = []
    label_path_list = []
    for i, p in enumerate(li_fpath):
        # データ
        img = Image.open(p)
        img = np.array(img).astype(np.float32) / 255
        img = img.reshape(1, img.shape[0], img.shape[1])

        # ラベル
        path_elements = p.split(os.path.sep)
        label_str = path_elements[-1][0]  # a,i,u,e,o
        label_num = aiueo_to_num[label_str]
        lbl = np.zeros(len(aiueo_to_num))
        lbl[label_num] = 1  # one hot

        out_dir = os.path.join(tmp_dir, label_str)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path_img = os.path.join(out_dir, '{}_{:05d}-image.npy'.format(label_str, i))
        np.save(out_path_img, img)
        image_path_list.append(out_path_img)
        out_path_lbl = os.path.join(out_dir, '{}_{:05d}-label.npy'.format(label_str, i))
        np.save(out_path_lbl, lbl)
        label_path_list.append(out_path_lbl)

    # シャッフル
    idx = np.arange(len(image_path_list))
    np.random.shuffle(idx)
    np_data = np.array(image_path_list)[idx]
    np_label = np.array(label_path_list)[idx]

    return np_data, np_label


def create_cross_validation_data(data, label, div=5):
    """
    交差検証用データセット作成
    :param data: データ
    :param label: ラベル
    :param div: 分割数
    :return: (訓練データ, テストデータ) のリスト
    """
    total = len(data)
    test_num = total // div

    train_set_list = []
    dev_set_list = []
    for i in range(div):
        dev_start = i * test_num
        dev_end = dev_start + test_num

        dev_data = data[dev_start:dev_end]
        dev_label = label[dev_start:dev_end]
        dev_set_list.append((dev_data, dev_label))

        train_data = np.concatenate((data[0:dev_start], data[dev_end:]), axis=0)
        train_label = np.concatenate((label[0:dev_start], label[dev_end:]), axis=0)
        train_set_list.append((train_data, train_label))

    return train_set_list, dev_set_list


def create_cross_validation_data_path(data, label, div=5):
    """
    交差検証用データセット作成
    :param data: データ
    :param label: ラベル
    :param div: 分割数
    :return: (訓練データ, テストデータ) のリスト
    """
    total = len(data)
    test_num = total // div

    train_set_list = []
    dev_set_list = []
    for i in range(div):
        dev_start = i * test_num
        dev_end = dev_start + test_num

        dev_data = data[dev_start:dev_end]
        dev_label = label[dev_start:dev_end]
        dev_set_list.append((dev_data, dev_label))

        train_data = data[0:dev_start].extend(data[dev_end:])
        train_label = label[0:dev_start].extend(label[dev_end:])
        train_set_list.append((train_data, train_label))

    return train_set_list, dev_set_list
