from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from data.BinaryDbReader import *
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, EvalUtil, load_weights_from_snapshot
import matplotlib.pyplot as plt
# flag that allows to load a retrained snapshot(original weights used in the paper are used otherwise)
USE_RETRAINED = False
PATH_TO_SNAPSHOTS = './snapshots_posenet/'  # only used when USE_RETRAINED is true

# build network
evaluation = tf.placeholder_with_default(True, shape=())
net = ColorHandPose3DNetwork()
image_crop = cv2.imread("data/00006.png")
image_crop = cv2.resize(image_crop, (256, 256))
image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
image_crop_ori = image_crop / 255.0 - 0.5

image_crop = tf.convert_to_tensor(image_crop_ori[np.newaxis])
keypoints_scoremap = net.inference_pose2d(image_crop)
keypoints_scoremap = keypoints_scoremap[-1]

# upscale to original size

keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (256,256))

# Start TF
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# initialize network weights
if USE_RETRAINED:
    # retrained version
    last_cpt = tf.train.latest_checkpoint(PATH_TO_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess,
                               last_cpt,
                               discard_list=['Adam', 'global_step', 'beta'])
else:
    # load weights used in the paper
    net.init(sess,
             weight_files=['./weights/posenet-rhd-stb.pickle'],
             exclude_var_list=['PosePrior', 'ViewpointNet'])


# get prediction
keypoints_scoremap_v = sess.run([keypoints_scoremap])

keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)

# detect keypoints
coord_hw_pred_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))

plt.figure()
plt.subplot(121)
plt.imshow(image_crop_ori + 0.5)
plt.scatter(
    coord_hw_pred_crop[:, 1],
    coord_hw_pred_crop[:, 0],
)
plt.subplot(122)
hms = np.sum(keypoints_scoremap_v, axis=2)
plt.imshow(hms)
plt.show()