from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 当前文件夹的母文件夹加入系统路径
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from tf_pose.runner import infer, Estimator, get_estimator
