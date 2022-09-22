#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

enum InputType {
    IMG_INPUT = 0,
    VIDEO_INPUT
};

int get_video_frame(cv::Mat &img, cv::VideoCapture &cap);
