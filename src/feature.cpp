/*
 * feature.cpp
 *
 *  Created on: Feb 19, 2016
 *      Author: thienlong
 */


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "lbp.hpp"
using namespace cv;

namespace signverify {
#define GRID_X 20
#define GRID_Y 10
#define POLAR_CORNER_ANGULAR 3
#define POLAR_CENTER_ANGULAR 12
#define POLAR_DISTANCE 15

//@tested
void removeBackground(const cv::Mat& sign, cv::Mat& dist, int postLevel) {
	dist.create(sign.size(), CV_8U);
	for (int r = 0; r < sign.rows; ++r) {
		for (int c = 0; c < sign.cols; ++c) {
			int pixel = round(round(sign.at<uchar>(r, c) * postLevel / 255.0) * 255.0 / postLevel);
			dist.at<uchar>(r, c) = (pixel == 255) ? 255 : sign.at<uchar>(r, c);
		}
	}
}

//@tested
void displaceHist(const cv::Mat& sign, cv::Mat& dist) {
	dist.create(sign.size(), CV_8U);
	double min;
	minMaxLoc(sign, &min);
	int minPixel = min;
	for (int r = 0; r < sign.rows; ++r) {
		for (int c = 0; c < sign.cols; ++c) {
			int pixel = sign.at<uchar>(r, c);
			dist.at<uchar>(r, c) = (pixel == 255) ? 255 : pixel - minPixel;
		}
	}
}

void lbpGrid(const cv::Mat& src, cv::Mat& output) {
//	imshow("src", src);
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);

	int width = static_cast<int>(ceil(src.cols / (float) GRID_X));
	int height = static_cast<int>(ceil(src.rows / (float) GRID_Y));
	int padrigh = width * GRID_X - src.cols;
	int padbottom = height * GRID_Y - src.rows;
	Mat padSrc;
	copyMakeBorder(refineSrc, padSrc, 0, padbottom, 0, padrigh, BORDER_CONSTANT, Scalar(255));
//	imshow("padSrc", padSrc);
//	waitKey(0);
	spatialUniLbpHist(padSrc, output, GRID_X, GRID_Y);
}

void riuLbpGrid(const cv::Mat& src, cv::Mat& output) {
//	imshow("src", src);
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);

//	int width = static_cast<int>(ceil(src.cols / (float) GRID_X));
//	int height = static_cast<int>(ceil(src.rows / (float) GRID_Y));
//	int padrigh = width * GRID_X - src.cols;
//	int padbottom = height * GRID_Y - src.rows;
//	Mat padSrc;
//	copyMakeBorder(refineSrc, padSrc, 0, padbottom, 0, padrigh, BORDER_CONSTANT, Scalar(255));
//	imshow("padSrc", padSrc);
//	waitKey(0);
	spatialRiuLbpHist(refineSrc, 2, 16, output, GRID_X, GRID_Y);
}

void hogGrid(const cv::Mat& src, cv::Mat& output) {
//	imshow("src", src);
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);

	int width = static_cast<int>(ceil(src.cols / (float) GRID_X));
	int height = static_cast<int>(ceil(src.rows / (float) GRID_Y));
	int padrigh = width * GRID_X - src.cols;
	int padbottom = height * GRID_Y - src.rows;
	Mat padSrc;
	copyMakeBorder(refineSrc, padSrc, 0, padbottom, 0, padrigh, BORDER_CONSTANT, Scalar(255));

	HOGDescriptor hog(padSrc.size(), Size(width, height), Size(width, height), Size(width, height), 9);
	vector<float> ders;
	vector<Point> locs;
	hog.compute(padSrc, ders, Size(0, 0), Size(0,0), locs);
	output.create(1, ders.size(), CV_32FC1);
	for (int i = 0; i < ders.size(); i++) {
		output.at<float>(0, i) = ders.at(i);
	}
//		imshow("padSrc", padSrc);
//		waitKey(0);
}

void hogPolar(const cv::Mat& src, cv::Mat& output);
}