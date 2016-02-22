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

//preprocessing

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

//end preprocessing

template <typename FeatureExtracter> void featureGrid(const cv::Mat& src, cv::Mat& output, FeatureExtracter extractor) {
	int width = static_cast<int>(ceil(src.cols / (float) GRID_X));
	int height = static_cast<int>(ceil(src.rows / (float) GRID_Y));
	int padrigh = width * GRID_X - src.cols;
	int padbottom = height * GRID_Y - src.rows;
	Mat padSrc;
	copyMakeBorder(src, padSrc, 0, padbottom, 0, padrigh, BORDER_CONSTANT, Scalar(255));
	for (int i = 0; i < GRID_Y; i++) {
		for (int j = 0; j < GRID_X; j++) {
			Mat src_cell = Mat(padSrc, Range(i * height, (i + 1) * height),
					Range(j * width, (j + 1) * width));
			Mat cell_hist;
			extractor(src_cell, cell_hist);
			output.push_back(cell_hist);
		}
	}
	output = output.reshape(0, 1);
}

template <typename FeatureExtracter> void featureOverlapGrid(const cv::Mat& src, cv::Mat& output, FeatureExtracter extractor) {
	int width = static_cast<int>(ceil(src.cols / 5.0f));
	int height = static_cast<int>(ceil(src.rows / 5.0f));
	int padrigh = width * 5 - src.cols;
	int padbottom = height * 5 - src.rows;
	Mat padSrc;
	copyMakeBorder(src, padSrc, 0, padbottom, 0, padrigh, BORDER_CONSTANT, Scalar(255));
	Rect cell(0, 0, width, height);
	for (int i = 0; i < 6; i++) {
		cell.y = i * 0.8 * height;
		for (int j = 0; j < 6; j++) {
			cell.x = j * 0.8 * width;
			Mat src_cell = padSrc(cell);
			Mat cell_hist;
			extractor(src_cell, cell_hist);
			output.push_back(cell_hist);
		}
	}
	output = output.reshape(0, 1);
}

#define featureGrid featureOverlapGrid

void lbpGrid(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	featureGrid(refineSrc, output, ulbpHist);
}

void riuLbpGrid(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	featureGrid(refineSrc, output, riuLbpHist1_8);
}

void lbpGrid28_step0(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	featureGrid(refineSrc, output, rlbpHist2_8_step0);
}

void lbpGrid28_step1(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	featureGrid(refineSrc, output, rlbpHist2_8_step1);
}

void lbpGrid38_step0(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	featureGrid(refineSrc, output, rlbpHist3_8_step0);
}

void lbpGrid38_step1(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	featureGrid(refineSrc, output, rlbpHist3_8_step1);
}

void lbpGrid38_step2(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	featureGrid(refineSrc, output, rlbpHist3_8_step2);
}


void hogGrid(const cv::Mat& src, cv::Mat& output) {
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
}

void hogPolar(const cv::Mat& src, cv::Mat& output);
}
