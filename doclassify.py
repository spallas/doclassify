import os

import tensorflow as tf

# run on CPU
# comment this part if you want to run it on GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def installation_test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(str(sess.run(hello)))
    print("INFO: Installed version: " + str(tf.VERSION))
    print("INFO: GPU found: ", tf.test.gpu_device_name())


def main(_):
    installation_test()


if __name__ == "__main__":
    tf.app.run()
