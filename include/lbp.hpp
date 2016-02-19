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
void lbpHist(const cv::Mat & img_gray, cv::Mat & hist);
void ulbpHist(const cv::Mat & img_gray, cv::Mat & hist);
void riuLbpHist(const cv::Mat& src, int radius, int neighbors, cv::Mat& hist);
void riuLbpHist1_8(const cv::Mat& img_gray, cv::Mat& hist);
}

#endif /* LBP_HPP_ */
