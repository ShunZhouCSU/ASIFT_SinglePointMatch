#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

class ASiftDetector
{
public:
	ASiftDetector();

	void detectAndComputeSingle(const cv::Mat &img, int footprint_x, int footprint_y, std::vector< cv::KeyPoint >& keypoints, cv::Mat &descriptors);
	void detectAndComputeAll(const cv::Mat& img, std::vector< cv::KeyPoint >& keypoints, cv::Mat& descriptors);
	void ASIFTMatchAndDraw(const std::string win, const cv::Mat img1, std::vector<cv::KeyPoint> kpts1, cv::Mat desc1,
		const cv::Mat img2, std::vector<cv::KeyPoint> kpts2, cv::Mat desc2);
	void ASIFTMatch(const cv::Mat img1, std::vector<cv::KeyPoint> kpts1, cv::Mat desc1, const cv::Mat img2,
		std::vector<cv::KeyPoint> kpts2, cv::Mat desc2, std::vector<cv::DMatch>& finalmatch);

private:
	void affineSkew(double tilt, double phi, cv::Mat &img, cv::Mat &mask, cv::Mat &Ai);
};