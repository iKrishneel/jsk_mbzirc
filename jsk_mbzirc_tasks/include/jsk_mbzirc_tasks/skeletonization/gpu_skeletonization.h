// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#ifndef _GPU_SKELETONIZATION_H_
#define _GPU_SKELETONIZATION_H_

#include <ros/ros.h>
#include <ros/console.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/Image.h>

#include <jsk_mbzirc_msgs/ProjectionMatrix.h>
#include <jsk_mbzirc_tasks/skeletonization/skeletonization_kernel.h>
#include <jsk_mbzirc_tasks/Skeletonization.h>

namespace jsk_tasks = jsk_mbzirc_tasks;
namespace jsk_msgs = jsk_mbzirc_msgs;

class GPUSkeletonization {

    struct Point3D {
       float x;
       float y;
       float z;
    };

 private:
    typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, jsk_msgs::ProjectionMatrix> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::Image> sub_image_;
    message_filters::Subscriber<jsk_msgs::ProjectionMatrix> sub_proj_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;

    float ground_plane_;
    float beacon_distance_;
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
  
    ros::NodeHandle pnh_;
    ros::Publisher pub_image_;
   
 public:
    GPUSkeletonization();
    void callback(const sensor_msgs::Image::ConstPtr &,
                  const jsk_msgs::ProjectionMatrix::ConstPtr &);
    Point3D pointToWorldCoords(const jsk_msgs::ProjectionMatrix,
                                  const float, const float);
    float EuclideanDistance(const cv::Point3_<float> *);
    cv::Size getSlidingWindowSize(const jsk_msgs::ProjectionMatrix);
};


#endif  // _GPU_SKELETONIZATION_H_
