#include "media.h"

using namespace std;

int ImageMedia::create_media(const std::string &path)
{
	std::cout << "ImageMedia path: " << path << std::endl;
	_img = cv::imread(path, 1);
	if (!_img.data)
	{
		printf("cv::imread %s fail!\n", path);
		return -1;
	}
	media_width = _img.cols;
	media_height = _img.rows;
	create_writer(path);
}

static bool getted = false;
int ImageMedia::get_media(cv::Mat &media)
{
	if (!getted)
	{
		getted = true;
		media = _img;
		return 0;
	}
	return -1;
}

void ImageMedia::write_detected(cv::Mat &media, detect_result_group_t &detect_result_group)
{
	plot_bbox_on_img(media, detect_result_group);
	imwrite(_output_path, media);
	cout << "[ImageMedia] Saved detected img." << endl;
}

void ImageMedia::create_writer(const string &file_path)
{
	int ps = file_path.find_last_of("/");
	std::string filename_with_suffix = file_path.substr(ps + 1);
	_output_path = "model/detected_" + filename_with_suffix;
}

int VideoMedia::create_media(const string &path)
{
	if (!is_end_with(path, "mp4") && !is_end_with(path, "avi"))
	{
		cout << "Video currently only support mp4 or avi." << endl;
		return -1;
	}
	_cap.open(path);
	if (!_cap.isOpened())
	{
		cerr << "Error opening video stream or file" << endl;
		return -2;
	}
	media_width = _cap.get(cv::CAP_PROP_FRAME_WIDTH);
	media_height = _cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	create_writer(path);
}

int VideoMedia::get_media(cv::Mat &media)
{
	_cap.read(media);
	if (media.empty())
	{
		printf("Failed to read video frame.\n");
		return -1;
	}
	return 0;
}

void VideoMedia::write_detected(cv::Mat &media, detect_result_group_t &detect_result_group)
{
	plot_bbox_on_img(media, detect_result_group);
	_video.write(media);
}

void VideoMedia::create_writer(const string &file_path)
{
	int ps = file_path.find_last_of("/");
	std::string filename_with_suffix = file_path.substr(ps + 1);
	_output_path = "model/detected_" + filename_with_suffix.substr(0, filename_with_suffix.find_last_of('.')) + ".avi";
	DIR *dir;
	if ((dir = opendir(_output_path.c_str())) != NULL)
		remove(_output_path.c_str());
	_video = cv::VideoWriter(_output_path,
							 cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
							 _cap.get(cv::CAP_PROP_FPS), cv::Size(media_width, media_height));
}

void ImagesMedia::check_if_has_next()
{
	std::cout << "ImagesMedia::check_if_has_next: " << std::string(_p) << std::endl;
	while ((_dp = readdir(_d)) != NULL)
	{
		std::cout << "--------------------in while" << std::endl;
		bool not_a_directory_or_file = !strcmp(_dp->d_name, ".") || !strcmp(_dp->d_name, "..");
		// std::vector<std::string> support_formats = {"jpg", "jpeg", "png"};
		// std::find()
		std::cout << "_dp->d_name " << _dp->d_name << std::endl;
		std::cout << "not_a_directory_or_file:" << not_a_directory_or_file << std::endl;

		if (not_a_directory_or_file || !(is_end_with(_dp->d_name, "jpg") || is_end_with(_dp->d_name, "jpeg") || is_end_with(_dp->d_name, "png")))
			continue;
		snprintf(_p, sizeof(_p) - 1, "%s/%s", _path, _dp->d_name);
		std::cout << "ImagesMedia::check_if_has_next: " << std::string(_p) << std::endl;
		stat(_p, &_st);
		if (!S_ISDIR(_st.st_mode))
			_has_next = true;
			break;
	}
}

int ImagesMedia::create_media(const string &path)
{
	_path = path.c_str();
	if (stat(_path, &_st) < 0 || !S_ISDIR(_st.st_mode))
	{
		printf("invalid path: %s\n", _path);
		return -1;
	}
	if (_path != NULL && !(_d = opendir(_path)))
	{
		printf("opendir[%s] error: %m\n", _path);
		return -3;
	}
	check_if_has_next();
	create_writer(path);
}

int ImagesMedia::get_media(cv::Mat &media)
{
	if (!_has_next)
		return -2;

	std::cout << "ImagesMedia::get_media: " << std::string(_p) << std::endl;
	media = cv::imread(_p);
	_has_next = false;
	while ((_dp = readdir(_d)) != NULL)
	{
		bool not_a_directory_or_file = !strcmp(_dp->d_name, ".") || !strcmp(_dp->d_name, "..");
		if (not_a_directory_or_file || !(is_end_with(_dp->d_name, "jpg") || is_end_with(_dp->d_name, "jpeg") || is_end_with(_dp->d_name, "png")))
			continue;
		snprintf(_p, sizeof(_p) - 1, "%s/%s", _path, _dp->d_name);
		stat(_p, &_st);
		if (!S_ISDIR(_st.st_mode))
			_has_next = true;
		return 0;
	}
}

void ImagesMedia::create_writer(const string &file_path)
{
	_output_path = "detected.txt";
	_fp = fopen(_output_path.c_str(), "w+");
}

void ImagesMedia::write_detected(cv::Mat &media, detect_result_group_t &detect_result_group)
{
	std::string path(_p);
	printf("_p is %s\n", _p);
	int ps = path.find_last_of("/");
	std::string filename_with_suffix = path.substr(ps + 1);
	// TODO prevents fopen here.

	for (int i = 0; i < detect_result_group.count; i++)
	{
		detect_result_t *det_result = &(detect_result_group.results[i]);

		fscanf(_fp, "image: %s cls_name %s prob: %f bbox(xywh): %d %d %d %d\n",
			   filename_with_suffix.c_str(),
			   det_result->name,
			   det_result->prop,
			   det_result->box.left,
			   det_result->box.top,
			   det_result->box.right - det_result->box.left,
			   det_result->box.bottom - det_result->box.top);
		printf("image: %s cls_name %s prob: %f bbox(xywh): %d %d %d %d\n",
			   filename_with_suffix.c_str(),
			   det_result->name,
			   det_result->prop,
			   det_result->box.left,
			   det_result->box.top,
			   det_result->box.right - det_result->box.left,
			   det_result->box.bottom - det_result->box.top);
	}
}

ImagesMedia::~ImagesMedia()
{
	fclose(_fp);
}

BaseMedia *create_media(int type, const std::string &path)
{
	switch (type)
	{
	case IMG:
	{
		BaseMedia *media = new ImageMedia();
		media->create_media(path);
		return media;
		break;
	}
	case VDO:
	{
		BaseMedia *media = new VideoMedia();
		media->create_media(path);
		return media;
		break;
	}
	case DIRECTORY:
	{
		BaseMedia *media = new ImagesMedia();
		media->create_media(path);
		return media;
		break;
	}
	default:
	{
		printf("Usage: <rknn model> <img/vdo/dir> <jpg/jpeg/png/mp4/avi/dir> \n");
		return NULL;
	}
	}
}