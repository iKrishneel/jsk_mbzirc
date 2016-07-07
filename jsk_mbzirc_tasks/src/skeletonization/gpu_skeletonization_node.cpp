// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#include <jsk_mbzirc_tasks/skeletonization/gpu_skeletonization.h>

GPUSkeletonization::GPUSkeletonization() {
    // boost::shared_ptr<GPUSkeletonization> object(
    //    boost::make_shared<GPUSkeletonization>());
    // ros::ServiceServer service = pnh_.advertiseService(
    //    "gpu_skeletonization", &GPUSkeletonization::skeletonizationGPUSrv,
    //    object);
    this->onInit();
}

void GPUSkeletonization::onInit() {
    this->subscribe();
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/skeletonization/output/image", 1);
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
          image_msg, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat image = cv_ptr->image.clone();

    //! CHANGE TO CUDA VERSION
    int morph_size = 4;
    cv::Mat element = cv::getStructuringElement(
       cv::MORPH_ELLIPSE, cv::Size(2*morph_size + 1, 2*morph_size+1),
       cv::Point(morph_size, morph_size));
    cv::Mat dst;
    cv::dilate(image, dst, element);
    element = cv::getStructuringElement(
       cv::MORPH_ELLIPSE, cv::Size(3*morph_size + 1, 3*morph_size+1),
       cv::Point(morph_size, morph_size));
    cv::erode(dst, image, element);
    cv::GaussianBlur(image, image, cv::Size(21, 21), 1, 0);
    
    // cv::imshow("dilate", dst);
    // cv::imshow("erode", image);
    // cv::waitKey(3);
    
    skeletonizationGPU(image);
    cv_ptr->image = image.clone();
    this->pub_image_.publish(cv_ptr->toImageMsg());
}


bool GPUSkeletonization::skeletonizationGPUSrv(
    jsk_tasks::Skeletonization::Request &request,
    jsk_tasks::Skeletonization::Response &response) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          request.image, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return false;
    }
    cv::Mat image = cv_ptr->image.clone();
    skeletonizationGPU(image);
    cv_ptr->image = image.clone();
    response.image = *(cv_ptr->toImageMsg());  //! CHANGE TO ARRAY
                                               //! INSTEAD
    return true;
}


int main(int argc, char *argv[]) {

    ros::init(argc, argv, "gpu_skeletonization");
    GPUSkeletonization gpu_s;
    ros::spin();
    return 0;
}
