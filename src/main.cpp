#include "Landmark.hpp"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
    string mnn_path = argv[1];
    Landmark landmark_detector(mnn_path);

    for (int i = 2; i < argc; ++i) {
        string image_file = argv[i];
        cout << "Processing " << image_file << endl;

        cv::Mat frame = cv::imread(image_file);
        
        auto start = chrono::steady_clock::now();
        vector<float> landmark_results;
        landmark_detector.generate_landmark(frame, landmark_results);

        for (int j = 0; j < 98; ++j) {
            cv::Point pt(landmark_results[2*i], landmark_results[2*i+1]);
            cv::circle(frame, pt, 2, CV_RGB(255, 0, 0));
        }

        auto end = chrono::steady_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "all time: " << elapsed.count() << " s" << endl;
        cv::imshow("UltraFace", frame);
        cv::waitKey();
    }
    return 0;
}
