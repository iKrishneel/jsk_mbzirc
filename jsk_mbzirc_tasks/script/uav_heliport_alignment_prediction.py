#!/usr/bin/env python

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
import scipy


#sub_mask_ = '/track_region_mapping/output/track_mask'
sub_mask_ = '/skeletonization/output/image'  # skeletonized
sub_odo_ = '/ground_truth/state'
sub_imu_ = '/raw_imu'
sub_matrix_ = '/projection_matrix'

sub_image_ = '/downward_cam/camera/image'
sub_point3d_ = '/uav_landing_region/output/point'
sub_pose_ = '/uav_landing_region/output/pose'

pub_image_ = None
pub_topic_ = '/track_region_segmentation/output/track_mask'

ALTITUDE_THRESH_ = 5.0  ## for building map
DISTANCE_THRESH_ = 4.0  ## incase of FP
VEHICLE_SPEED_ = 3.0  ## assume fast seed of 15km/h
BEACON_POINT_DIST_ = 1.0 ## distances between beacon points in m

class MapInfo:
    image = None
    indices = []
    point3d = []
    odometry = None
    imu = None

class DijkstraShortestPath:
    def __init__(self, adjacency_matrix):
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            rospy.logfatal("the input adjacent matrix is not square")
            return
        self.adjacency_matrix = adjacency_matrix
        
    def dijkstra(self, src):
        lenght = self.adjacency_matrix.shape[0]
        if src > lenght:
            rospy.logerr("-- search index is out of size")
            return
        dist = np.zeros((1, lenght), np.int)
        dist.fill(sys.maxint)
        spt_set = np.zeros((1, lenght), np.bool)        
        dist[0, src] = 0
        for i in range(lenght -1):
            u = self._min_distance(dist, spt_set)
            spt_set[0, u] = True
            for j in range(lenght):
                if (spt_set[0, j] == 0) and (self.adjacency_matrix[u][j]) and (dist[0, u] != sys.maxint) and \
                   (dist[0, u] + self.adjacency_matrix[u][j] < dist[0, j]):
                    dist[0, j] = dist[0, u] + self.adjacency_matrix[u][j]
                    
        sorted_index = dist.argsort()
        s_index = sorted_index[0, 1]
        s_dist = dist[0, s_index]
        return (s_index, s_dist)

    def _min_distance(self, dist, spt_set):
        imin = sys.maxint
        min_index = None
        for i in range(self.adjacency_matrix.shape[0]):
            if np.logical_and((spt_set[0, i] == 0), (dist[0, i] <= imin)):
                imin = dist[0, i]
                min_index = i
        return min_index


class HeliportAlignmentAndPredictor:
    def __init__(self):
        self.pub_image_ = rospy.Publisher(pub_topic_, Image, queue_size=10) 
        
        self.subscribe()
        self.map_info = MapInfo()
        self.proj_matrix = None
        self.is_initalized = False
        self.kdtree = None
        self.position_list = []
        self.indices_cache = []  # to avoid search over 
        self.dijkstra = None

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

        start = time.time()
            
        current_point = np.array((point_msg.point.x, point_msg.point.y, point_msg.point.z))
        current_point = current_point.reshape(1, -1)
        distances, indices = self.kdtree.kneighbors(current_point)
        
        ##? add condition to limit neigbors
        if distances > DISTANCE_THRESH_:
            rospy.logerr("point is to far")
            return
        
        ## debug view
        im_color = cv2.cvtColor(self.map_info.image, cv2.COLOR_GRAY2BGR)
        x, y = self.map_info.indices[indices]
        cv2.circle(im_color, (x, y), 10, (0, 255, 0), -1)

        # get the direction
        self.position_list.append([current_point, point_msg.header.stamp])
        self.indices_cache.append([distances, indices])
        if len(self.position_list) < 2:
            rospy.logwarn("direction is unknown... another detection is required")
            return

        #time_diff = point_msg.header.stamp - self.position_list[0][1]
        #print "difference in time: ", time_diff

        prev_index = len(self.position_list) - 2
        index_pmp = self.indices_cache[prev_index][1]
        prev_map_pos = self.map_info.point3d[index_pmp]
        curr_map_pos = self.map_info.point3d[indices]
                
        vm_dx = curr_map_pos[0] - prev_map_pos[0]
        vm_dy = curr_map_pos[1] - prev_map_pos[1]
        vm_tetha = math.atan2(-vm_dy, vm_dx)# * (180.0/np.pi)

        iter_count = 0
        ground_z = current_point[0][2]
        previous_point = self.position_list[prev_index][0]
        while True:
            n_distance, n_index = self.kdtree.radius_neighbors(current_point, radius = VEHICLE_SPEED_, return_distance = True)

            closes_index = -1  # index of the point on the map
            max_dist = 0

            for i, indx in enumerate(n_index[0]):
                mp_x = self.map_info.point3d[indx][0]
                mp_y = self.map_info.point3d[indx][1]
                map_pt = np.array((mp_x, mp_y, ground_z)) 
                dist_mp = scipy.linalg.norm(map_pt - previous_point)
                
                if dist_mp > max_dist:
                    closes_index = indx
                    max_dist = dist_mp

            previous_point = current_point
            xx = self.map_info.point3d[closes_index][0]
            yy = self.map_info.point3d[closes_index][1]
            current_point = np.array((xx, yy, ground_z))
            
            if iter_count > 200:
                break
            iter_count += 1

            x1, y1 = self.map_info.indices[closes_index]

            #print "iter: ", iter_count, "\t", x1, ",", y1 , "\t", current_point
            
            cv2.circle(im_color, (x1, y1), 5, (0, 0, 255), -1)
            self.plot_image("plot", im_color)
            cv2.waitKey(3)
            
            rospy.sleep(1)
        
        end = time.time()
        print "PROCESSING TIME: ", (end - start)

        
        x1, y1 = self.map_info.indices[closes_index]
        cv2.circle(im_color, (x1, y1), 5, (0, 0, 255), -1)
        

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
        
        neighbors_size = 3
        self.kdtree = NearestNeighbors(n_neighbors = neighbors_size, radius = BEACON_POINT_DIST_, algorithm = "kd_tree", 
                                       leaf_size = 30, \
                                       metric='euclidean').fit(np.array(world_points))        
        

        self.plot_image("init", image)

        start = time.time()
        adjacency_list = self.sample_beacon_points(np.array(world_points))

        adjacency_matrix = np.zeros((len(world_points), len(world_points)), np.int)
        for a_list in adjacency_list:
            adjacency_matrix[a_list[0], a_list[1]] = 1
            adjacency_matrix[a_list[0], a_list[2]] = 1

        end = time.time()
        print "PROCESSING TIME: ", (end - start)
        print ""

        #! build linked list
        # adjacency_matrix = np.zeros((len(world_points), len(world_points)), np.int)
        # for windex, wpt in enumerate(world_points):
        #     w_distances, w_indices = self.kdtree.kneighbors(np.array(wpt).reshape(1, -1))
        #     adjacency_matrix[windex][windex] = 1
        #     adjacency_matrix[windex][w_indices[0][1]] = 1
        #     adjacency_matrix[windex][w_indices[0][2]] = 1

        
        self.dijkstra = DijkstraShortestPath(adjacency_matrix)
        shortest_index, shortest_dist = self.dijkstra.dijkstra(0)
        print "DISTANCE", shortest_index, "\t", shortest_dist

        img = np.zeros((image.shape[0], image.shape[0], 3),  np.uint8)
        for i, al in enumerate(adjacency_list):
            index = indices[al[0]]
            color = (0, 255, 0)
            rad = 3
            if i == 0:
                color = (0, 0, 255)
                rad = 5
            cv2.circle(img, (index[0], index[1]), rad,  color, -1)
        self.plot_image("beacon", img)
        cv2.waitKey(0)

        rospy.loginfo("-- map initialized")

        del image
        del world_points
        del indices
        del altit

    def sample_beacon_points(self, points):
        kdtree = NearestNeighbors(n_neighbors=3, radius = BEACON_POINT_DIST_, algorithm='kd_tree', metric='euclidean').fit(points) 
        #! control params
        current_index = 0
        prev_distance = 0.0
        last_flag = False

        start_point = points[current_index].reshape(1, -1)
        indices_list = []        
        flag = np.zeros((1, points.shape[current_index]), np.bool)
        flag[0][0] = True
        
        while True:
            dist, index = kdtree.radius_neighbors(start_point, radius=BEACON_POINT_DIST_, return_distance = True)
            s_ind = dist[0].argsort()[::-1]
            ind1 = current_index
            ind2 = index[0][s_ind[0]]
            ind3 = None
            if len(indices_list) > 1:
                dim = len(indices_list) - 1
                ind3 = indices_list[dim][0]
                max_d1 = 0.0
                max_d2 = BEACON_POINT_DIST_
                for i in index[0]:
                    dist1 = scipy.linalg.norm(points[ind1] - points[i])
                    dist2 = scipy.linalg.norm(points[ind3] - points[i])
                    if dist1 > max_d1 and dist2 > max_d2:
                        max_d1 = dist1
                        max_d2 = dist2
                        ind2 = i
                    if i < current_index:
                        flag[0][i] = True
            else:
                max_d = 0
                for i in index[0]:
                    d = scipy.linalg.norm(points[ind2] - points[i])
                    if d > max_d:
                        max_d = d
                        ind3 = i
                    if i < current_index:
                        flag[0][i] = True

            inl = (ind1, ind2, ind3)  #format(curent, next, prev)
            print inl
            indices_list.append(inl) 
        
            if last_flag:
                print "end reached"
                break


            if len(indices_list) > 1:
                x = indices_list[0][2]            
                d = scipy.linalg.norm(start_point - points[x]) 
                #print "\033[34m Dist: \033[0m", d, "\t", indices_list[0]
                if d < BEACON_POINT_DIST_:
                    last_flag = True

            next_idx = ind2
            print "flags: ", flag[0][ind2], "\t", flag[0][ind3]
            if flag[0][ind2] and flag[0][ind3] == False:
                next_idx = ind3
            elif flag[0][ind2] and flag[0][ind3]:
                rospy.logerr("--checking")
                # condition for broken edges
                search_radius = 5 #VEHICLE_SPEED_ * 2.0 # search for broken edges
                #select a point search_radius back 
                sel_ind = len(indices_list) - int(search_radius)
                if sel_ind < 0:
                    sel_ind = 0                
                select_index = indices_list[sel_ind][0] # to avoid extracting point on same side
                #s_distance, s_index = kdtree.radius_neighbors(start_point, radius=search_radius, return_distance = True)
                s_distance, s_index = kdtree.kneighbors(start_point, n_neighbors=100, return_distance=True)
                
                #print points[select_index], "\t", select_index, "\t", len(indices_list)
                #print start_point[0]
                angle_criteria =  math.atan2(start_point[0][1] - points[select_index][1],
                                             start_point[0][0] - points[select_index][0]) * (180.0/np.pi)
                angle1 = None
                angle2 = None
                if angle_criteria >= 0.0 and angle_criteria <= 90:
                    angle1 = 90.0
                    angle2 = 0.0
                elif angle_criteria > 90 and angle_criteria <= 180.0:
                    angle1 = -180.0
                    angle2 = 90.0
                elif angle_criteria < 0.0 and angle_criteria >= -90.0:
                    angle1 = 0.0
                    angle2 = -90.0
                elif angle_criteria <-90.0:
                    angle1 = -90.0
                    angle2 = 180.0

                dp_ac = 1000.0
                dp_ab = scipy.linalg.norm(start_point - points[select_index])
                search_index = -1
                for si in s_index[0]:
                    dist_sp1 = scipy.linalg.norm(start_point - points[si]) #dist_a*c
                    dist_sp2 = scipy.linalg.norm(points[select_index] - points[si]) #dist_b*c
                    
                    angle_sp =  math.atan2(start_point[0][1] - points[si][1],
                                           start_point[0][0] - points[si][0]) * (180.0/np.pi)

                    #print angle_criteria, ", ", angle_sp, "\t", angle1, "\t", angle2                    
                    #print "\033[34mdistance: ", dist_sp1 , "\t", dist_sp2, "\t", dp_ac, "\t", dp_ab, "\033[0m" 

                    if dist_sp1 < dp_ac and dist_sp2 > dp_ab and (angle_sp > angle1 and angle_sp < angle2):
                        dp_ac = dist_sp1
                        search_index = si
                                        
                #print "search_index: ", search_index
                #index = self.map_info.indices[search_index]
                #cv2.circle(img, (index[0], index[1]), 15,  (255, 0, 255), -1)

                if search_index > -1:
                    print "updating", search_index
                    next_idx = search_index                    
                else:
                    print "all done"
                    break
                # end condition

            start_point = points[next_idx].copy().reshape(1, -1)
            flag[0][next_idx] = True
            current_index = next_idx

            ## DEBUG plot
            # img = np.zeros((480, 640, 3),  np.uint8)
            # for l in inl:
            #     index = self.map_info.indices[l]
            #     color = (0, 255, 0)
            #     rad = 2
            #     cv2.circle(img, (index[0], index[1]), rad,  color, -1)
            # self.plot_image("beacon", img)
            # cv2.waitKey(0)
            ## DEBUG plot END
        
        del kdtree
        del flag
        del current_index

        return np.array(indices_list)
        
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

    """"
    adjacency_matrix = ((0, 4, 0, 0, 0, 0, 0, 8, 0),
                        (4, 0, 8, 0, 0, 0, 0, 11, 0),
                        (0, 8, 0, 7, 0, 4, 0, 0, 2),
                        (0, 0, 7, 0, 9, 14, 0, 0, 0),
                        (0, 0, 0, 9, 0, 10, 0, 0, 0),
                        (0, 0, 4, 0, 10, 0, 2, 0, 0),
                        (0, 0, 0, 14, 0, 2, 0, 1, 6),
                        (8, 11, 0, 0, 0, 0, 1, 0, 7),
                        (0, 0, 2, 0, 0, 0, 6, 7, 0))
    adjacency_matrix = np.array(adjacency_matrix)
    #print adjacency_matrix
    dsp = DijkstraShortestPath(adjacency_matrix)
    dsp.dijkstra(0)

    """""
