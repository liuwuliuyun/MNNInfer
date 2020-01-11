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
    // auto nchw_tensor = new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE);
    // input_tensor->copyFromHostTensor(nchw_tensor);
    // delete nchw_tensor;
}


Landmark::~Landmark() {
    landmark_interpreter->releaseModel();
    landmark_interpreter->releaseSession(landmark_session);
}


int Landmark::generate_landmark(cv::Mat &raw_img, std::vector<float> &landmarks){
    cv::Mat image;
    cv::resize(raw_img, image, cv::Size(input_image_size, input_image_size));
    cv::Mat norm_image;
    cv::normalize(image, norm_image, 0, 255, NORM_MINMAX, CV_8UC3);

    landmark_interpreter->resizeTensor(input_tensor, {1, 3, input_image_size, input_image_size});
    landmark_interpreter->resizeSession(landmark_session);
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3, std_vals, 3)
    );
    pretreat->convert(norm_image.data, input_image_size, input_image_size, norm_image.step[0], input_tensor);

    auto start = std::chrono::steady_clock::now();

    landmark_interpreter->runSession(landmark_session);

    std::string output = "landmarks";
    MNN::Tensor *tensor_landmarks = landmark_interpreter->getSessionOutput(landmark_session, output.c_str());
    MNN::Tensor tensor_landmarks_host(tensor_landmarks, tensor_landmarks->getDimensionType());
    tensor_landmarks->copyToHostTensor(&tensor_landmarks_host);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "inference time:" << elapsed.count() << " s" << std::endl;

    for (int i = 0; i < 196; ++i) {
        float point = tensor_landmarks->host<float>()[i];
        landmarks.push_back(point);
    }
    return 0;
}