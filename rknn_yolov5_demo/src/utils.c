int is_end_with(const std::string &filename, const std::string &end)
{
	std::string suffix_str = filename.substr(filename.find_last_of('.') + 1);
	if (suffix_str == end)
		return 1;
	else
		return 0;
}