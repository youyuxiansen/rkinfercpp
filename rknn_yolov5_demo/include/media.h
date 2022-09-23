#ifndef _RKNN_MEDIA_H_
#define _RKNN_MEDIA_H_

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "postprocess.h"
#include "utils.h"

class BaseMedia
{
public:
	int media_width = 0;
	int media_height = 0;
	bool still_have = false;

public:
	virtual int create_media(const std::string &path) = 0;
	virtual int get_media(cv::Mat &media) = 0;
	virtual void create_writer(const std::string &file_path) = 0;
	virtual void write_detected(cv::Mat &media, detect_result_group_t &detect_result_group) = 0;
};

class ImageMedia : public BaseMedia
{
private:
	cv::Mat _img;
	std::string _output_path;

public:
	ImageMedia(){};
	int create_media(const std::string &path);
	int get_media(cv::Mat &media);
	void create_writer(const std::string &file_path);
	void write_detected(cv::Mat &media, detect_result_group_t &detect_result_group);
};

class VideoMedia : public BaseMedia
{
public:
	int media_width = 0;
	int media_height = 0;

private:
	cv::VideoCapture _cap;
	cv::VideoWriter _video;
	bool _if_write_video = false;
	std::string _output_path;

public:
	VideoMedia(){};
	int create_media(const std::string &path);
	int get_media(cv::Mat &media);
	void create_writer(const std::string &file_path);
	void write_detected(cv::Mat &media, detect_result_group_t &detect_result_group);
};

#define MAX_PATH_LEN 256
class ImagesMedia : public BaseMedia
{
public:
	bool _has_next = false;

private:
	DIR *_d = NULL;
	struct stat _st;
	struct dirent *_dp = NULL;
	char _p[MAX_PATH_LEN] = {0};
	cv::VideoCapture _cap;
	cv::VideoWriter _video;
	const char *_path;
	FILE *_fp;
	std::string _output_path;

public:
	ImagesMedia(){};
	int create_media(const std::string &path);
	int get_media(cv::Mat &media);
	void create_writer(const std::string &file_path);
	void write_detected(cv::Mat &media, detect_result_group_t &detect_result_group);
	~ImagesMedia();

private:
	void check_if_has_next();
};

BaseMedia *
create_media(int type, const std::string &path);

#endif // _RKNN_MEDIA_H_