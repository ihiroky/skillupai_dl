from PIL import Image, ImageOps
import numpy as np
import Augmentor
import glob
import os
import shutil
import sys

# 訓練データの拡張
# １ピクセルずらしたデータは精度低下を招いたのでやめた

in_dir = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/1_data/train')
out_dir27 = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/1_data/small27/')
out_dir27_last = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/1_data/train/small27/')
out_dir26 = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/1_data//small26/')
out_dir26_last = os.path.join(os.path.expanduser('~'), 'skillupai/DAY1_vr2_1_0/4_kadai/1_data/train/small26/')

def ensure(s):
    if os.path.exists(s):
        shutil.rmtree(s)
    os.makedirs(s)

ensure(out_dir27)
ensure(out_dir27_last)
ensure(out_dir26)
ensure(out_dir26_last)

fpaths = glob.glob(os.path.join(in_dir, '*', '*.png'))
fpaths += glob.glob(os.path.join(in_dir, '*', '*', '*.png'))
for i, path in enumerate(fpaths):
    dir_name, file_name = os.path.split(path)
    _, parent_name = os.path.split(dir_name)
    base_name, ext = os.path.splitext(file_name)
    d27 = os.path.join(out_dir27, parent_name)
    if not os.path.exists(d27):
        os.makedirs(d27)
    d26 = os.path.join(out_dir26, parent_name)
    if not os.path.exists(d26):
        os.makedirs(d26)

    img = Image.open(path)
    '''
    img27h = img.resize((28, 27))
    img27u = np.pad(img27h, ((1, 0), (0, 0)), 'constant', constant_values=255)
    Image.fromarray(img27u).save(os.path.join(out_dir27, parent_name, base_name + '-27u.png'))
    img27d = np.pad(img27h, ((0, 1), (0, 0)), 'constant', constant_values=255)
    Image.fromarray(img27d).save(os.path.join(out_dir27, parent_name, base_name + '-27d.png'))

    img27w = img.resize((27, 28))
    img27l = np.pad(img27w, ((0, 0), (1, 0)), 'constant', constant_values=255)
    Image.fromarray(img27l).save(os.path.join(out_dir27, parent_name, base_name + '-27l.png'))
    img27r = np.pad(img27w, ((0, 0), (0, 1)), 'constant', constant_values=255)
    Image.fromarray(img27r).save(os.path.join(out_dir27, parent_name, base_name + '-27r.png'))
    '''
    img26  = img.resize((26, 26))
    img26  = np.pad(img26, ((1, 1), (1, 1)), 'constant', constant_values=255)
    Image.fromarray(img26).save(os.path.join(out_dir26, parent_name, file_name + '-26.png'))

# os.rename(out_dir27, out_dir27_last)
os.rename(out_dir26, out_dir26_last)
