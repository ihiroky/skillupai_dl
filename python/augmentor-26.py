import Augmentor
import os
import shutil
from PIL import Image
import glob

# 縮小データの拡張

home = os.path.expanduser('~')
data_root = '{}/skillupai/DAY1_vr2_1_0/4_kadai/1_data/train/small26'.format(home)
dir_out_last = os.path.join(data_root, 'augmented')
if os.path.exists(dir_out_last):
    shutil.rmtree(dir_out_last)
os.makedirs(dir_out_last)

for c in ('a', 'i', 'u', 'e', 'o'):
    dir_in = os.path.join(data_root, c)
    dir_out = os.path.join(dir_in, 'output')
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)

    def sample(g, m, s):
        p = Augmentor.Pipeline(dir_in)
        p.random_distortion(probability=1.0, grid_height=g, grid_width=g, magnitude=m)
        p.sample(s)
        out = os.path.join(dir_out_last, os.path.basename(dir_out) + '_' + c + '_g' + str(g) + '_m' + str(m))
        os.rename(dir_out, out)

    sample(4, 1, 200)
    sample(4, 2, 200)
    # sample(6, 1, 200)
    # sample(8, 1, 450)
