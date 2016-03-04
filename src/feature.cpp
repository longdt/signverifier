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
#include "hog.hpp"
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

//@tested
void removeNoise(const cv::Mat& sign, cv::Mat& dst) {
	Mat binary(sign.size(), sign.type());
	//binary
	for (int r = 0; r < sign.rows; ++r) {
		for (int c = 0; c < sign.cols; ++c) {
			binary.at<uchar>(r, c) = sign.at<uchar>(r, c) == 255 ? 0 : 255;
		}
	}
//	threshold(sign, binary, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	//remove small blobs
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	vector<int> small_blobs;
	double contour_area;
	if (!contours.empty()) {
	    for (size_t i=0; i<contours.size(); ++i) {
	        contour_area = contourArea(contours[i]) ;
	        if ( contour_area < 20)
	            small_blobs.push_back(i);
	    }
	}
	sign.copyTo(dst);
	for (size_t i=0; i < small_blobs.size(); ++i) {
	    drawContours(dst, contours, small_blobs[i], cv::Scalar(255), CV_FILLED, 8);
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

template <typename FeatureExtracter> void featureHierarchyLevelGrid(const cv::Mat& src, cv::Mat& output, FeatureExtracter extractor, int level) {
	int width = static_cast<int>(ceil(src.cols / (1 + level * 0.8f)));
	int height = static_cast<int>(ceil(src.rows / (1 + level * 0.8f)));
	int jumpWidth = static_cast<int>(ceil(0.8f * width));
	int jumpHeight = static_cast<int>(ceil(0.8f * height));
	int padrigh = width + level * jumpWidth - src.cols;
	int padbottom = height + level * jumpHeight - src.rows;
	Mat padSrc;
	copyMakeBorder(src, padSrc, 0, padbottom, 0, padrigh, BORDER_CONSTANT, Scalar(255));
	Rect cell(0, 0, width, height);
	for (int i = 0; i <= level; i++) {
		cell.y = i * jumpHeight;
		for (int j = 0; j <= level; j++) {
			cell.x = j * jumpWidth;
			Mat src_cell = padSrc(cell);
			Mat cell_hist;
			extractor(src_cell, cell_hist);
			output.push_back(cell_hist);
		}
	}
}

//l = 10 almost good for lbp, low l is for hog but not good
template <typename FeatureExtracter> void featureHierarchyGrid(const cv::Mat& src, cv::Mat& output, FeatureExtracter extractor) {
	for (int l = 0; l < 4; ++l) {
		featureHierarchyLevelGrid(src, output, extractor, l);
	}
	output = output.reshape(0, 1);
}

#define featureGrid featureHierarchyGrid

void lbpGrid(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	removeNoise(refineSrc, refineSrc);
	featureGrid(refineSrc, output, lbpHist);
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


//void hogGrid(const cv::Mat& src, cv::Mat& output) {
//	Mat refineSrc;
//	removeBackground(src, refineSrc, 4);
//	displaceHist(refineSrc, refineSrc);
//
//	int width = static_cast<int>(ceil(src.cols / (float) GRID_X));
//	int height = static_cast<int>(ceil(src.rows / (float) GRID_Y));
//	int padrigh = width * GRID_X - src.cols;
//	int padbottom = height * GRID_Y - src.rows;
//	Mat padSrc;
//	copyMakeBorder(refineSrc, padSrc, 0, padbottom, 0, padrigh, BORDER_CONSTANT, Scalar(255));
//
//	HOGDescriptor hog(padSrc.size(), Size(width, height), Size(width, height), Size(width, height), 90);
//	vector<float> ders;
//	vector<Point> locs;
//	hog.compute(padSrc, ders, Size(0, 0), Size(0,0), locs);
//	output.create(1, ders.size(), CV_32FC1);
//	for (int i = 0; i < ders.size(); i++) {
//		output.at<float>(0, i) = ders.at(i);
//	}
//}

void hogHist(const cv::Mat& img_gray, cv::Mat& output) {
	hog(img_gray, output, 9);
}


void hogGrid(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	removeNoise(refineSrc, refineSrc);
//	Mat normal(128, 256, CV_8U);
//	resize(refineSrc, normal, normal.size());
//	imshow("normal", normal);
//	waitKey(0);
	featureGrid(refineSrc, output, hogHist);
}

void phogGrid(const cv::Mat& src, cv::Mat& output) {
	Mat refineSrc;
	removeBackground(src, refineSrc, 4);
	displaceHist(refineSrc, refineSrc);
	removeNoise(refineSrc, refineSrc);
//	threshold(refineSrc, refineSrc, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	Mat normal(128, 256, CV_8U);
	resize(refineSrc, normal, normal.size());
	int bin = 36;
	output.create(1, bin * 21, CV_32F);
	int idx = 0;
	HOGDescriptor hog0(normal.size(), normal.size(), normal.size(), normal.size(), bin);
	vector<float> ders;
	hog0.compute(normal, ders);
	for (size_t i = 0; i < ders.size(); i++) {
		output.at<float>(0, idx) = ders.at(i);
		++idx;
	}
	ders.clear();
	hog0.cellSize = Size(normal.cols / 2, normal.rows / 2);
	hog0.compute(normal, ders);
	for (size_t i = 0; i < ders.size(); i++) {
		output.at<float>(0, idx) = ders.at(i);
		++idx;
	}
	ders.clear();
	hog0.blockSize = hog0.cellSize;
	hog0.blockStride = hog0.blockSize;
	hog0.cellSize = Size(normal.cols / 4, normal.rows / 4);
	hog0.compute(normal, ders);
	for (size_t i = 0; i < ders.size(); i++) {
		output.at<float>(0, idx) = ders.at(i);
		++idx;
	}
}

void hogPolar(const cv::Mat& src, cv::Mat& output);
}
