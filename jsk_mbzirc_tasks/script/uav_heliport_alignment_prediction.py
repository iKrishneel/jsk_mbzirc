#/usr/bin/env python

import roslib
roslib.load_manifest("jsk_mbzirc_tasks")

import rospy
from cv_bridge import CvBridge
import message_filters
import tf

from sklearn.neighbors import NearestNeighbors, KDTree, DistanceMetric
from scipy.spatial import distance

from sensor_msgs.msg import Image, PointCloud2, Imu
from jsk_mbzirc_msgs.msg import ProjectionMatrix
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

import numpy as np
from  collections import Counter
import sys
import math
import cv2
import random
import time

sub_mask_ = '/track_region_mapping/output/track_mask'
sub_odo_ = '/ground_truth/state'
sub_imu_ = '/raw_imu'
sub_matrix_ = '/projection_matrix'

sub_image_ = '/downward_cam/camera/image'
sub_point3d_ = '/uav_landing_region/output/point'
sub_pose_ = '/uav_landing_region/output/pose'

pub_image_ = None
pub_topic_ = '/track_region_segmentation/output/track_mask'

ALTITUDE_THRESH_ = 5.0  ## for building map

class MapInfo:
    image = None
    indices = []
    point3d = []
    odometry = None
    imu = None

class HeliportAlignmentAndPredictor:
    def __init__(self):
        self.pub_image_ = rospy.Publisher(pub_topic_, Image, queue_size=10) 

        self.subscribe()
        self.map_info = MapInfo()
        self.proj_matrix = None
        self.is_initalized = False
        self.kdtree = None
        self.broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

    def subscribe(self):
        mask_sub = message_filters.Subscriber(sub_mask_, Image)
        odom_sub = message_filters.Subscriber(sub_odo_, Odometry)
        imu_sub = message_filters.Subscriber(sub_imu_, Imu)
        pmat_sub = message_filters.Subscriber(sub_matrix_, ProjectionMatrix)
        init_ats = message_filters.ApproximateTimeSynchronizer((mask_sub, odom_sub, imu_sub, pmat_sub), 10, 1)
        init_ats.registerCallback(self.init_callback)
        
        sub_image = message_filters.Subscriber(sub_image_, Image)
        sub_point3d = message_filters.Subscriber(sub_point3d_, PointStamped)
        sub_pose = message_filters.Subscriber(sub_pose_, PoseStamped)

        ats = message_filters.ApproximateTimeSynchronizer((sub_image, sub_point3d, sub_pose), 10, 10)
        ats.registerCallback(self.callback)

    def callback(self, image_msg, point_msg, pose_msg):
    
        if not self.is_initalized:
            rospy.logerr("-- vehicle track map info is not availible")
            return        
        
        quat_base = self.map_info.imu.orientation
        quat_now = pose_msg.pose.orientation
        quaternion_now = (quat_now.x, quat_now.y, quat_now.z, quat_now.w)
        quaternion_base = (quat_base.x, quat_base.y, quat_base.z, quat_base.w)        

        pos_now = pose_msg.pose.position
        pos_base = self.map_info.odometry.pose.pose.position
        position_now = (pos_now.x, pos_now.y, pos_now.z)
        position_base = (pos_base.x, pos_base.y, pos_base.z)
        
        time = rospy.Time.now()
        self.broadcaster.sendTransform(position_now, quaternion_now, time, "/now", "/world")
        self.broadcaster.sendTransform(position_base, quaternion_base, time, "/base", "/world")
        
        translation = None
        rotation = None
        try:
            (translation,rotation) = self.listener.lookupTransform('/now', '/base', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("transformation lookup failed")
            return
        
        
    
        current_point = np.array((point_msg.point.x, point_msg.point.y, point_msg.point.z))
        current_point = current_point.reshape(1, -1)
        distances, indices = self.kdtree.kneighbors(current_point)
        
        ##? add condition to limit neigbors
        
        
        

        ## debug view
        im_color = cv2.cvtColor(self.map_info.image, cv2.COLOR_GRAY2BGR)
        x, y = self.map_info.indices[indices]
        cv2.circle(im_color, (x, y), 10, (0, 255, 0), -1)
        
        del translation
        del rotation
        del time
        del quat_base
        del quat_now
        del quaternion_base
        del quaternion_now
        del pos_now
        del pos_base
        del position_now
        del position_base

        # self.plot_image("input", self.map_info.image)
        self.plot_image("plot", im_color)
        cv2.waitKey(3)
        

    def init_callback(self, mask_msg, odometry_msgs, imu_msg, projection_msg):
        self.proj_matrix = np.reshape(projection_msg.data, (3, 4))
        if (self.is_initalized):
            return

        altit = odometry_msgs.pose.pose.position.z
        if altit < ALTITUDE_THRESH_:
            rospy.logwarn("cannot build the map at this altitude: "+ str(altit))
            return

        image = self.convert_image(mask_msg, "mono8")        
        world_points = []
        indices = []
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if image[y, x] == 255:
                    point3d = self.projection_to_world_coords(x, y, 0.0)
                    world_points.append(point3d)
                    indices.append([x, y])
                    
        self.map_info.indices = indices
        self.map_info.point3d = world_points
        self.map_info.odometry = odometry_msgs
        self.map_info.imu = imu_msg
        self.map_info.image = image
        self.is_initalized = True

        self.kdtree = NearestNeighbors(1, algorithm = "kd_tree", leaf_size = 30, \
                                       metric='euclidean', n_jobs=8).fit(np.array(world_points))
        
        rospy.loginfo("-- map initialized")

        del image
        del world_points
        del indices
        del altit

        # self.plot_image("input", image)
        # cv2.waitKey(3)
        
    def convert_image(self, image_msg, encoding):
        bridge = CvBridge()
        cv_img = None
        try:
            cv_img = bridge.imgmsg_to_cv2(image_msg, str(encoding))
        except Exception as e:
            print (e)
        return cv_img

    def plot_image(self, name, image):
        cv2.namedWindow(str(name), cv2.WINDOW_NORMAL)
        cv2.imshow(str(name), image)

    def projection_to_world_coords(self, x, y, ground_z = 0.0):    
        a00 = x * self.proj_matrix[2, 0] - self.proj_matrix[0, 0]
        a01 = x * self.proj_matrix[2, 1] - self.proj_matrix[0, 1]
        a10 = y * self.proj_matrix[2, 0] - self.proj_matrix[1, 0]
        a11 = y * self.proj_matrix[2, 1] - self.proj_matrix[1, 1]
        bv0 = self.proj_matrix[0, 2] * ground_z + self.proj_matrix[0, 3] -  \
              x * self.proj_matrix[2, 2] * ground_z - x * self.proj_matrix[2, 3]
        bv1 = self.proj_matrix[1, 2] * ground_z + self.proj_matrix[1, 3] -  \
              y * self.proj_matrix[2, 2] * ground_z - y * self.proj_matrix[2, 3]
        denom = a11 * a00 - a01 * a10
        pos_x = (a11 * bv0 - a01 * bv1) / denom
        pos_y = (a00 * bv1 - a10 * bv0) / denom
        return (pos_x, pos_y, ground_z)


def main():
    rospy.init_node('uav_heliport_alignment_prediction', anonymous=True)
    hpp = HeliportAlignmentAndPredictor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logwarn("SHUT DOWN COMMAND RECEIVED")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

