#include "utils.h"

int ImageMedia::create_media(const std::string &file_path)
{
	_img = cv::imread(source, 1);
	if (!_img.data)
	{
		printf("cv::imread %s fail!\n", source);
		return -1;
	}
	media_width = _img.cols;
	media_height = _img.rows;
}

int ImageMedia::get_media(cv::Mat &media)
{
	media = _img;
	return 0;
}

int VideoMedia::create_media(const std::string &file_path)
{
	_cap.open(file_path);
	if (!cap.isOpened())
	{
		std::cerr << "Error opening video stream or file" << std::endl;
		return -1;
	}
	media_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	media_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
}

int VideoMedia::get_media(cv::Mat &media)
{
	cap >> media;
	return 0;
}

void VideoMedia::create_video_writer(const std::string &file_path)
{
	_video = cv::VideoWriter(file_path,
							 cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
							 cap.get(cv::CAP_PROP_FPS), cv::Size(img_width, img_height));
}

void ImagesMedia::check_if_has_next(const char *path)
{
	_dp = readdir(path.c_str());
	not_a_directory_or_file = (!strncmp(_dp->d_name, ".", 1)) || (!strncmp(_dp->d_name, "..", 2));
	while (_dp != NULL)
	{
		if (not_a_directory_or_file || !is_end_with(source, "jpg"))
			continue;
		snprintf(_p, sizeof(_p) - 1, "%s/%s", path, _dp->d_name);
		stat(_p, &st);
		if (!S_ISDIR(st.st_mode))
			_has_next = true;
	}
}

int ImagesMedia::create_media(const std::string &path)
{

	if (stat(path.c_str(), &_st) < 0 || !S_ISDIR(_st.st_mode))
	{
		printf("invalid path: %s\n", path.c_str());
		return -1;
	}
	if (!(_d = opendir(path.c_str())))
	{
		printf("opendir[%s] error: %m\n", path);
		return -3;
	}
	check_if_has_next(git);
}

int ImagesMedia::get_media(cv::Mat &media)
{
	if (!_has_next)
		return -2;
	media = cv::imread(_p);
	media_width = media.cols;
	media_height = media.rows;
	_has_next = (_dp = readdir(d)) != NULL;
	return 0;
}