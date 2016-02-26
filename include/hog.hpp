/*
 * hog.hpp
 *
 *  Created on: Feb 25, 2016
 *      Author: thienlong
 */

#ifndef HOG_HPP_
#define HOG_HPP_

namespace signverify {
void rhog(const cv::Mat& src, cv::Mat& output, int bin = 9);
void hog(const cv::Mat& src, cv::Mat& output, int bin = 9);
}



#endif /* HOG_HPP_ */
