#include "utils.h"

int is_end_with(const std::string &filename, const std::string &end)
{
	std::string suffix_str = filename.substr(filename.find_last_of('.') + 1);
	if (suffix_str == end)
		return 1;
	else
		return 0;
}

void plot_bbox_on_img(cv::Mat &img, detect_result_group_t &detect_result_group)
{
	for (int i = 0; i < detect_result_group.count; i++)
	{
		detect_result_t *det_result = &(detect_result_group.results[i]);
		printf("%s @ (%d %d %d %d) %f\n",
			   det_result->name,
			   det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
			   det_result->prop);
		int x1 = det_result->box.left;
		int y1 = det_result->box.top;
		int x2 = det_result->box.right;
		int y2 = det_result->box.bottom;
		rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
		putText(img, det_result->name, cv::Point(x1, y1 + 12), 1, 2, cv::Scalar(0, 255, 0, 255));
	}
}