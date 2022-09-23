#ifndef _RKNN_UTILS_H_
#define _RKNN_UTILS_H_

#include <string.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"

int is_end_with(const std::string &filename, const std::string &end);
void plot_bbox_on_img(cv::Mat &img, detect_result_group_t &detect_result_group);

enum MEDIA_TYPE
{
	IMG = 0,
	VDO,
	DIRECTORY
};


#endif // _RKNN_UTILS_H_