#include "Landmark.hpp"

Landmark::Landmark(const std::string &mnn_path, int input_size = 256, int num_thread_ = 4) {
    input_image_size = input_size;
    num_threads = num_thread_;

    landmark_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));

    MNN::ScheduleConfig config;
    config.numThread = num_threads;

    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;

    config.backendConfig = &backendConfig;

    landmark_session = landmark_interpreter->createSession(config);

    input_tensor = landmark_interpreter->getSessionInput(landmark_session, nullptr);
    auto nchw_tensor = new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE);
    input_tensor->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;
}


Landmark::~Landmark() {
    landmark_interpreter->releaseModel();
    landmark_interpreter->releaseSession(landmark_session);
}


int Landmark::generate_landmark(const uint8_t* img, std::vector<float> &landmarks){

}