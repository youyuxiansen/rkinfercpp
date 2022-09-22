#include <VideoProcessor.h>

int get_video_frame(cv::Mat &img, cv::VideoCapture &cap) {
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }
    cap >> img;
    return 0;
}


