// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#ifndef _SKELETONIZATION_KERNEL_H_
#define _SKELETONIZATION_KERNEL_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cublas.h>
#include <cublas_v2.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#define GRID_SIZE 16

void skeletonizationGPU(cv::Mat &);


#endif   // _SKELETONIZATION_KERNEL_H_
