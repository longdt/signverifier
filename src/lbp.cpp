/*
 * lbp.cpp
 *
 *  Created on: Dec 15, 2015
 *      Author: thienlong
 */

#include "lbp.hpp"

using namespace cv;
namespace signverify {
uchar lbp(const Mat_<uchar> & img, int x, int y) {
	// this is pretty much the same what you already got..
	uchar v = 0;
	uchar c = img(y, x);
	v += (img(y - 1, x) > c) << 0;
	v += (img(y - 1, x + 1) > c) << 1;
	v += (img(y, x + 1) > c) << 2;
	v += (img(y + 1, x + 1) > c) << 3;
	v += (img(y + 1, x) > c) << 4;
	v += (img(y + 1, x - 1) > c) << 5;
	v += (img(y, x - 1) > c) << 6;
	v += (img(y - 1, x - 1) > c) << 7;
	return v;
}

static uchar uniform[256] = { // hardcoded 8-neighbour case
		0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58,
				58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58,
				58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 17, 58, 58, 58,
				58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58,
				58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
				58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58,
				58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 24, 58, 58, 58,
				58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 29, 30, 58, 31,
				58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58,
				58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 58, 58, 58, 58,
				58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
				58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38,
				58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40, 58, 58, 58, 58,
				58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 44,
				58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48, 58, 49,
				58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57 };

void uniformLbpHist(const Mat & img_gray, Mat & hist) {
	// 59 bins, bin 58 is the noise/non-uniform slot:
	hist = Mat::zeros(1, 59, CV_32F);

	Mat_<uchar> img(img_gray);

	for (int r = 1; r < img.rows - 1; ++r) {
		for (int c = 1; c < img.cols - 1; ++c) {
			uchar uv = lbp(img, c, r);
			hist.at<float>(0, uniform[uv])++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 1) * (img.cols - 1));
}

void spatialUniLbpHist(const cv::Mat & src, cv::Mat & hist, int grid_x, int grid_y) {
    // calculate LBP patch size
    int width = src.cols/grid_x;
    int height = src.rows/grid_y;
    hist = Mat::zeros(1, width * height * 59, CV_32FC1);

    // initial result_row
	int cellIdx = 0;
	// iterate through grid
	for(int i = 0; i < grid_y; i++) {
		for(int j = 0; j < grid_x; j++) {
			Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
			Mat cell_hist = Mat(hist, Rect(cellIdx * 59, 0, 59, 1));
			uniformLbpHist(src_cell, cell_hist);
			cellIdx++;
		}
	}
}

}

