# ifndef Landmark_hpp
# define Landmark_hpp

#pragma once

#include "Interpreter.hpp"

#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>

class Landmark {
    public:

    Landmark(const std::string &mnn_path, int input_size = 256, int num_thread_ = 4);
    ~Landmark();

    int generate_landmark(cv::Mat &img, std::vector<float> &landmarks);

    private:

    std::shared_ptr<MNN::Interpreter> landmark_interpreter;
    MNN::Session *landmark_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;

    int num_threads;
    int input_image_size;

    const float mean_vals[3] = {0.485, 0.456, 0.406};
    const float std_vals[3] = {0.229, 0.224, 0.225};

};


# endif