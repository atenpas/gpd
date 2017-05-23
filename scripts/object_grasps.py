import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image

from grasp_candidates_classifier.msg import GraspConfigList

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
caffe_root = '/home/andreas/software/caffe/'  # replace this with the location of your Caffe folder
sys.path.insert(0, caffe_root + 'python')
import caffe


# global variable to store grasps
grasps = []

# global variable to store RGB image
image = None


# Callback function to receive grasps.
def graspsCallback(msg):
    global grasps
    grasps = msg.grasps


# Callback function to receive images.
def imageCallback(msg):
    global image
    bridge = CvBridge()
    try:
        image = bridge.imgmsg_to_cv2(msg, "rgb8")
    except CvBridgeError, e:
        print(e)


# ==================== MAIN ====================
# Create a ROS node.
rospy.init_node('get_grasps')

# Subscribe to the ROS topic that contains the grasps.
grasps_sub = rospy.Subscriber('/detect_grasps/clustered_grasps', GraspConfigList, graspsCallback)

# Subscribe to the ROS topic that contains the RGB images.
image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, imageCallback)

# Wait for grasps to arrive.
rate = rospy.Rate(1)

while not rospy.is_shutdown():
    print '.'
    if len(grasps) > 0 and image != None:
        rospy.loginfo('Received %d grasps.', len(grasps))
        break
    rate.sleep()

# Detect mug.
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'

caffe.set_mode_cpu()

# Load the network and its weights.
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

# Load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
plt.show()

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]