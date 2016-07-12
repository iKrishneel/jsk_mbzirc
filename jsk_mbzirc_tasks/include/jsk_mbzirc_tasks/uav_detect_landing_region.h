// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#pragma once
#ifndef _UAV_DETECT_LANDING_REGION_H_
#define _UAV_DETECT_LANDING_REGION_H_

#include <omp.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <jsk_mbzirc_msgs/Rect.h>
#include <jsk_mbzirc_msgs/ProjectionMatrix.h>

#include <jsk_mbzirc_tasks/uav_detect_landing_region_trainer.h>
#include <jsk_mbzirc_tasks/NonMaximumSuppression.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace jsk_msgs = jsk_mbzirc_msgs;
namespace jsk_tasks = jsk_mbzirc_tasks;

class UAVLandingRegion: public UAVLandingRegionTrainer {
   
    typedef geometry_msgs::Point Point3D;
    typedef geometry_msgs::PointStamped Point3DStamped;

    struct MotionInfo {
       // Point3D veh_position;
       // Point3D uav_position;
       cv::Point2f veh_position;
       cv::Point2f uav_position;

       ros::Time time;
    };
   
 private:
    typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Imu,
    nav_msgs::Odometry, jsk_msgs::ProjectionMatrix> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<sensor_msgs::Image> sub_mask_;
    message_filters::Subscriber<sensor_msgs::Imu> sub_imu_;
    message_filters::Subscriber<jsk_msgs::ProjectionMatrix> sub_proj_;
    message_filters::Subscriber<nav_msgs::Odometry> sub_odom_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    int num_threads_;
    cv::Mat templ_img_;
    int down_size_;
    int min_wsize_;
    float nms_thresh_;
   
    float track_width_;
    float landing_marker_width_;
    float ground_plane_;

    MotionInfo motion_info_[2];
    int icounter_;
   
 protected:
    ros::NodeHandle pnh_;
    ros::Publisher pub_image_;
    ros::Publisher pub_point_;
    ros::Publisher pub_cloud_;
    ros::Publisher pub_pose_;
    
    ros::ServiceClient nms_client_;
   
    void onInit();
    void subscribe();
    void unsubscribe();
    
 public:
    UAVLandingRegion();
    virtual void imageCB(const sensor_msgs::Image::ConstPtr &,
                         const sensor_msgs::Image::ConstPtr &,
                         const sensor_msgs::Imu::ConstPtr &,
                         const nav_msgs::Odometry::ConstPtr &,
                         const jsk_msgs::ProjectionMatrix::ConstPtr &);
   
    cv::Point2f traceandDetectLandingMarker(cv::Mat, const cv::Mat,
                                            const cv::Size);
    cv::Mat convertImageToMat(const sensor_msgs::Image::ConstPtr &,
                              std::string);
    cv::Size getSlidingWindowSize(const cv::Size,
                                  const jsk_msgs::ProjectionMatrix);
    float EuclideanDistance(const cv::Point3_<float> *);
    Point3DStamped pointToWorldCoords(const jsk_msgs::ProjectionMatrix,
                                      const float, const float);
    void predictVehicleRegion(cv::Point2f &, const MotionInfo*);
};


#endif  // _UAV_DETECT_LANDING_REGION_H_
