// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#include <jsk_mbzirc_tasks/skeletonization/gpu_skeletonization.h>

GPUSkeletonization::GPUSkeletonization() {

    int icounter = 0;
    while (icounter++ < 50) {
       cv::Mat image = cv::imread("/home/krishneel/Desktop/mbzirc/mask.png");
       // cv::resize(image, image, cv::Size(1280, 960));
       std::cout << image.size()  << "\n";
    
       skeletonizationGPU(image);

       ROS_WARN("DONE");

       cv::imshow("input", image);
       cv::waitKey(30);
    }
}

void GPUSkeletonization::onInit() {
    this->subscribe();
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "output/image", 1);
}

void GPUSkeletonization::subscribe() {
    this->sub_image_ = this->pnh_.subscribe(
       "input_image", 1, &GPUSkeletonization::callback, this);
}

void GPUSkeletonization::unsubscribe() {
    this->sub_image_.shutdown();
}

void GPUSkeletonization::callback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat image = cv_ptr->image.clone();
}



int main(int argc, char *argv[]) {

    ros::init(argc, argv, "gpu_skeletonization");
    GPUSkeletonization gpu_s;
    ros::spin();
    return 0;
}
