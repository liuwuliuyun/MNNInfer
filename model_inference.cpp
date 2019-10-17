#include <math.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

cv::Mat pre_process(std::string image_name){
    cv::Mat mean = (cv::Mat(3,1) << 0.408, 0.447, 0.470)
    cv::Mat std = (cv::Mat(3,1) << 0.289, 0.274, 0.278)
    int pad = 31; //image padding number = 31
    cv::Mat raw_image = cv::imread(image_name.c_str());

    int height = raw_image.rows;
    int width = raw_image.cols;
    int inp_height = (height | pad) + 1;
    int inp_width = (width | pad) + 1;

    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(inp_width, inp_height));
    image.convertTo(image, CV_32FC3);
    image = (image / 255.0f - mean)/std;
    return image
}

int main(void){
    std::string file_name = ''
    cv::Mat img = pre_process(file_name)
    cv::imwrite('./test_result.jpg', img)
    return 0;
}