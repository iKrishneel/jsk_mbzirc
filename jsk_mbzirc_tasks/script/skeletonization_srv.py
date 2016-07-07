#!/usr/bin/env python

import roslib
roslib.load_manifest("jsk_mbzirc_task")

import rospy
import sys

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from jsk_mbzirc_tasks.srv import *

service_name_ = 'gpu_skeletonization'

def skeletonize_image(image):
    rospy.wait_for_service(service_name_)
    try:
        skeletonize = rospy.ServiceProxy(service_name_, Skeletonization)
        skel_image = skeletonize_gpu(image)
        return skel_image.image
    except rospy.ServiceException, e:
        rospy.logfatal("service named gpu_skeletonization not found")
    
