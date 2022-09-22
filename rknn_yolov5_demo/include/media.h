

class BaseMedia
{
public:
	int media_width = 0;
	int media_height = 0;
	bool still_have = false;
	virtual int create_media(const std::string &path) = 0;
	virtual int get_media(cv::Mat &media);

private:
}

class ImageMedia : public BaseMedia
{
public:
	static ImageMedia &Instance()
	{
		static ImageMedia instance;
		return instance;
	}
	int create_media(const std::string &path) override;
	void get_media(cv::Mat &media) override;

private:
	cv::Mat _img;
}

class VideoMedia : public BaseMedia
{
public:
	static VideoMedia &Instance()
	{
		static ImageMedia instance;
		return instance;
	}
	int create_media(const std::string &path) override;
	void get_media(cv::Mat &media) override;
	int create_video_writer(const std::string &file_path);

private:
	cv::VideoCapture _cap;
	cv::VideoWriter _video;
	bool _if_write_video = false;
}

class ImagesMedia : public BaseMedia
{
public:
	static ImagesMedia &Instance()
	{
		static ImageMedia instance;
		return instance;
	}
	int create_media(const std::string &path) override;
	void get_media(cv::Mat &media) override;

private:
	DIR *_d = NULL;
	struct stat _st;
	struct dirent *_dp = NULL;
	char _p[MAX_PATH_LEN] = {0};
	cv::VideoCapture _cap;
	cv::VideoWriter _video;
	bool _has_next = false;

	void check_if_has_next(const char* path);
}