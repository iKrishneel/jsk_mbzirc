// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#include <jsk_mbzirc_tasks/skeletonization/gpu_skeletonization.h>

GPUSkeletonization::GPUSkeletonization() :
    ground_plane_(0.0f), beacon_distance_(1.0f) {
    this->onInit();
}

void GPUSkeletonization::onInit() {
    this->subscribe();
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/skeletonization/output/image", 1);
}

void GPUSkeletonization::subscribe() {
    this->sub_image_.subscribe(this->pnh_, "input_image", 1);
    this->sub_proj_.subscribe(this->pnh_, "input_proj_mat", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(this->sub_image_, this->sub_proj_);
    this->sync_->registerCallback(
      boost::bind(&GPUSkeletonization::callback, this, _1, _2));
}

void GPUSkeletonization::unsubscribe() {
    this->sub_image_.unsubscribe();
    this->sub_proj_.unsubscribe();
}

void GPUSkeletonization::callback(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const jsk_msgs::ProjectionMatrix::ConstPtr &proj_mat_msg) {
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
    int morph_size = 2;
    // cv::Mat element = cv::getStructuringElement(
    //    cv::MORPH_ELLIPSE, cv::Size(2*morph_size + 1, 2*morph_size+1),
    //    cv::Point(morph_size, morph_size));
    // cv::Mat dst;
    // cv::dilate(image, dst, element);
    cv::Mat element = cv::getStructuringElement(
       cv::MORPH_ELLIPSE, cv::Size(2*morph_size + 1, 2*morph_size+1),
       cv::Point(morph_size, morph_size));
    cv::erode(image, image, element);

    cv::GaussianBlur(image, image, cv::Size(21, 21), 1, 0);
    // skeletonizationGPU(image);
    
    // cv::imshow("dilate", image);
    // cv::imshow("erode", image);
    // cv::waitKey(3);
    
    skeletonizationGPU(image);

    //! window for masking beacon points

    /*
    cv::Size wsize = this->getSlidingWindowSize(*proj_mat_msg);

    cv::Mat test_img = image.clone();
    cv::cvtColor(test_img, test_img, CV_GRAY2BGR);
    for (int j = 0; j < image.rows - wsize.height; j += wsize.height) {
       for (int i = 0; i < image.cols - wsize.width; i += (wsize.width*2)) {
          cv::Rect_<int> rect = cv::Rect_<int>(i, j, 5, 5);
          cv::rectangle(image, rect, cv::Scalar(0), -1);
          image.at<uchar>(j, i) = 0;

          cv::rectangle(test_img, rect, cv::Scalar(0, 255, 0), -1);
          test_img.at<cv::Vec3b>(j, i)[1] = 255;
       }
    }
    cv::namedWindow("image_mask", cv::WINDOW_NORMAL);
    cv::imshow("image_mask", image);

    cv::namedWindow("image_mask1", cv::WINDOW_NORMAL);
    cv::imshow("image_mask1", test_img);
    cv::waitKey(3);
    
    std::cout << "\033[31m Window Size:  \033[0m" << wsize  << "\n";
    */
    cv_ptr->image = image.clone();
    this->pub_image_.publish(cv_ptr->toImageMsg());
}

// void getBeaconPoints()

cv::Size GPUSkeletonization::getSlidingWindowSize(
    const jsk_msgs::ProjectionMatrix projection_matrix) {
    float A[2][2];
    float bv[2];

    const int NUM_POINTS = 2;
    const float pixel_lenght = 10;
    float init_point = 10;
    cv::Point2f point[NUM_POINTS];
    point[0] = cv::Point2f(init_point, init_point);
    point[1] = cv::Point2f(init_point + pixel_lenght,
                           init_point + pixel_lenght);

    cv::Point3_<float> world_coords[NUM_POINTS];
    for (int k = 0; k < NUM_POINTS; k++) {
       Point3D point_3d = this->pointToWorldCoords(
          projection_matrix, static_cast<int>(point[k].y),
          static_cast<int>(point[k].x));
       world_coords[k].x = point_3d.x;
       world_coords[k].y = point_3d.y;
       world_coords[k].z = point_3d.z;
    }
    float world_distance = this->EuclideanDistance(world_coords);
    float wsize = (pixel_lenght * this->beacon_distance_) / world_distance;
    wsize = (wsize < 5) ? 5 : wsize;
    return cv::Size(static_cast<int>(wsize), static_cast<int>(wsize));
}

float GPUSkeletonization::EuclideanDistance(
    const cv::Point3_<float> *world_coords) {
    float x = world_coords[1].x - world_coords[0].x;
    float y = world_coords[1].y - world_coords[0].y;
    float z = world_coords[1].z - world_coords[0].z;
    return std::sqrt((std::pow(x, 2) + (std::pow(y, 2)) + (std::pow(z, 2))));
}

GPUSkeletonization::Point3D GPUSkeletonization::pointToWorldCoords(
    const jsk_msgs::ProjectionMatrix projection_matrix,
    const float x, const float y) {
    float A[2][2];
    float bv[2];
    int i = static_cast<int>(y);
    int j = static_cast<int>(x);
    A[0][0] = j * projection_matrix.data.at(8) -
       projection_matrix.data.at(0);
    A[0][1] = j * projection_matrix.data.at(9) -
       projection_matrix.data.at(1);
    A[1][0] = i * projection_matrix.data.at(8) -
       projection_matrix.data.at(4);
    A[1][1] = i * projection_matrix.data.at(9) -
       projection_matrix.data.at(5);
    bv[0] = projection_matrix.data.at(2)*ground_plane_ +
       projection_matrix.data.at(3) - j*projection_matrix.data.at(
             10)*ground_plane_ - j*projection_matrix.data.at(11);
    bv[1] = projection_matrix.data.at(4)*ground_plane_ +
       projection_matrix.data.at(7) - i*projection_matrix.data.at(
          10)*ground_plane_ - i*projection_matrix.data.at(11);
    float dominator = A[1][1] * A[0][0] - A[0][1] * A[1][0];

    Point3D world_coords;
    world_coords.x = (A[1][1]*bv[0]-A[0][1]*bv[1]) / dominator;
    world_coords.y = (A[0][0]*bv[1]-A[1][0]*bv[0]) / dominator;
    world_coords.z = this->ground_plane_;
    return world_coords;
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "gpu_skeletonization");
    GPUSkeletonization gpu_s;
    ros::spin();
    return 0;
}
