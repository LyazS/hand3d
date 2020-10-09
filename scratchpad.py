from data.BinaryDbReader import BinaryDbReader
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
dataset = BinaryDbReader(mode='training',
                         batch_size=8,
                         shuffle=True,
                         hand_crop=True,
                         use_wrist_coord=False,
                         sigma=10,
                         coord_uv_noise=False,
                         crop_center_noise=False,
                         crop_offset_noise=False,
                         crop_scale_noise=False)
data = dataset.get()
print(data)
# Start TF
gpu_options = tf.GPUOptions(allow_growth=True, )
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)
sess.run(tf.global_variables_initializer())
while 1:
    d = sess.run(data)
    # print(d)
    hm_sum = np.sum(d["scoremap"][0], axis=2)
    plt.figure(figsize=(12, 12))
    plt.subplot(121)
    plt.imshow(hm_sum)
    plt.subplot(122)
    img = d["image_crop"][0] + 0.5
    img[:, :, 0] += hm_sum
    plt.imshow(img)
    plt.show()