/*
 * lbp.hpp
 *
 *  Created on: Dec 15, 2015
 *      Author: thienlong
 */

#ifndef LBP_HPP_
#define LBP_HPP_
#include <opencv2/highgui/highgui.hpp>
namespace signverify {

void uniformLbpHist(const cv::Mat & img_gray, cv::Mat & hist);
void spatialUniLbpHist(const cv::Mat & img_gray, cv::Mat & hist, int grid_x,
		int grid_y);
void riuLbpHist(const cv::Mat& src, int radius, int neighbors, cv::Mat& hist);
void spatialRiuLbpHist(const cv::Mat& src, int radius, int neighbors, cv::Mat& hist, int grid_x, int grid_y);
}

#endif /* LBP_HPP_ */
