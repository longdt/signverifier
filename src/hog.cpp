/*
 * hog.cpp
 *
 *  Created on: Feb 25, 2016
 *      Author: thienlong
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cmath>
using namespace cv;
using namespace std;

namespace signverify {
void computeGradients(const cv::Mat& src, cv::Mat& output) {
	output.create(src.size(), CV_32FC2);
	float dy = 0;
	float dx = 0;
	for (int r = 1; r < src.rows - 1; ++r) {
		for (int c = 1; c < src.cols - 1; ++c) {
			Vec2f& gradient = output.at<Vec2f>(r, c);
			dy = src.at<uchar>(r + 1, c) - src.at<uchar>(r - 1, c);
			dx = src.at<uchar>(r, c + 1) - src.at<uchar>(r, c - 1);
			gradient[0] = atan2(dy, dx); //need handle dy = dx = 0
			gradient[1] = sqrt(dx * dx + dy * dy);
		}
	}
}

#define PI 3.14159265

inline int findFloorBin(vector<float>& centerBins, float angel) {
	int i = centerBins.size() - 1;
	for (; i >= 0 && centerBins[i] > angel; --i) {
	}
	return i;
}

void rhog(const cv::Mat& src, cv::Mat& output, int bin) {
	Mat grad;
	computeGradients(src, grad);
	output = Mat::zeros(1, bin, CV_32F);
	float binSize = 360 / (float) bin;
	vector<float> centerBins(bin);
	centerBins[0] = 0;
	for (int i = 1; i < bin; ++i) {
		centerBins[i] = centerBins[i - 1] + binSize;
	}
	float prev = 0;
	float next = 0;
	for (int r = 1; r < grad.rows - 1; ++r) {
		for (int c = 1; c < grad.cols - 1; ++c) {
			Vec2f& gradient = grad.at<Vec2f>(r, c);
			float angel = gradient[0] * 180 / PI + 180;
			int idx = findFloorBin(centerBins, angel);
			prev = centerBins[idx];
			int nidx = 0;
			if (idx == centerBins.size() - 1) {
				next = 360;
			} else {
				next = centerBins[idx + 1];
				nidx = idx + 1;
			}
			output.at<float>(0, idx) += gradient[1] * (next - angel)
					/ (next - prev);
			output.at<float>(0, nidx) += gradient[1] * (angel - prev)
					/ (next - prev);
		}
	}
	//normalize
	float len = 0.00001;
	for (int i = 0; i < output.cols; ++i) {
		len += output.at<float>(0, i) * output.at<float>(0, i);
	}
	len = sqrt(len);
	output = output / len;
}

void hog(const cv::Mat& src, cv::Mat& output, int bin) {
	Size window = src.size();
	HOGDescriptor hog(window, window, window, window, bin);
	vector<float> ders;
	vector<Point> locs;
	hog.compute(src, ders, Size(0, 0), Size(0,0), locs);
	output.create(1, ders.size(), CV_32FC1);
	for (int i = 0; i < ders.size(); i++) {
		output.at<float>(0, i) = ders.at(i);
	}
}

}
