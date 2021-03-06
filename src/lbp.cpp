/*
 * lbp.cpp
 *
 *  Created on: Dec 15, 2015
 *      Author: thienlong
 */

#include "lbp.hpp"

using namespace cv;
namespace signverify {
inline uchar lbp(const Mat_<uchar> & img, int x, int y) {
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

void lbpHist(const Mat & img_gray, Mat & hist) {
	hist = Mat::zeros(1, 256, CV_32F);
	Mat_<uchar> img(img_gray);
	for (int r = 1; r < img.rows - 1; ++r) {
		for (int c = 1; c < img.cols - 1; ++c) {
			uchar uv = lbp(img, c, r);
			hist.at<float>(0, uv)++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 2) * (img.cols - 2));
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

void ulbpHist(const Mat & img_gray, Mat & hist) {
	// 59 bins, bin 58 is the noise/non-uniform slot:
	hist = Mat::zeros(1, 59, CV_32F);

	Mat_<uchar> img(img_gray);
	for (int r = 1; r < img.rows - 1; ++r) {
		for (int c = 1; c < img.cols - 1; ++c) {
			uchar uv = lbp(img, c, r);
			hist.at<float>(0, uniform[uv])++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 2) * (img.cols - 2));
}

//@tested
template<typename _Tp> inline float pixelAt(const cv::Mat& src, float x,
		float y) {
	// relative indices
	int fx = static_cast<int>(floor(x));
	int fy = static_cast<int>(floor(y));
	int cx = static_cast<int>(ceil(x));
	int cy = static_cast<int>(ceil(y));
	// fractional part
	float ty = y - fy;
	float tx = x - fx;
	// set interpolation weights
	float w1 = (1 - tx) * (1 - ty);
	float w2 = tx * (1 - ty);
	float w3 = (1 - tx) * ty;
	float w4 = tx * ty;
	float t = w1 * src.at<_Tp>(fy, fx) + w2 * src.at<_Tp>(fy, cx)
			+ w3 * src.at<_Tp>(cy, fx) + w4 * src.at<_Tp>(cy, cx);
	return t;
}

inline int riuLbp(const cv::Mat& src, int r, int c, int radius, int neighbors) {
	int f = 0;
	int lastF = -1;
	int firstF = -1;
	int sumF = 0;
	int u = 0;
	for (int n = 0; n < neighbors; n++) {
		// sample points
		float x = c + static_cast<float>(radius)
				* sin(2.0 * CV_PI * n / static_cast<float>(neighbors));
		float y = r - static_cast<float>(radius)
				* cos(2.0 * CV_PI * n / static_cast<float>(neighbors));
		// calculate interpolated value
		float t = pixelAt<uchar>(src, x, y);
		// floating point precision, so check some machine-dependent epsilon
		f = (t > src.at<uchar>(r, c))
				|| (std::abs(t - src.at<uchar>(r, c))
						< std::numeric_limits<float>::epsilon());
		sumF += f;
		if (firstF >= 0) {
			u += std::abs(f - lastF);
		} else {
			firstF = f;
		}
		lastF = f;
	}
	u += std::abs(firstF - lastF);
	return (u <= 2) ? sumF : (neighbors + 1);
}

void riuLbpHist(const cv::Mat& src, int radius, int neighbors, cv::Mat& hist) {
	hist = Mat::zeros(1, neighbors + 2, CV_32F);
	for (int r = radius; r < src.rows - radius; r++) {
		for (int c = radius; c < src.cols - radius; c++) {
			int lbp = riuLbp(src, r, c, radius, neighbors);
			hist.at<float>(0, lbp)++;
		}
	}
	hist /= ((src.rows - 2 * radius) * (src.cols - 2 * radius));
}

//special case riuLbp1_8
static uchar riuTable18[256] = {
		0, 1, 1, 2, 1, 9, 2, 3, 1, 9, 9, 9, 2, 9, 3, 4,
		1, 9, 9, 9, 9, 9, 9, 9, 2, 9, 9, 9, 3, 9, 4, 5,
		1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
		2, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 9, 4, 9, 5, 6,
		1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
		9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
		2, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
		3, 9, 9, 9, 9, 9, 9, 9, 4, 9, 9, 9, 5, 9, 6, 7,
		1, 2, 9, 3, 9, 9, 9, 4, 9, 9, 9, 9, 9, 9, 9, 5,
		9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6,
		9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
		9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7,
		2, 3, 9, 4, 9, 9, 9, 5, 9, 9, 9, 9, 9, 9, 9, 6,
		9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7,
		3, 4, 9, 5, 9, 9, 9, 6, 9, 9, 9, 9, 9, 9, 9, 7,
		4, 5, 9, 6, 9, 9, 9, 7, 5, 6, 9, 7, 6, 7, 7, 8
};

void riuLbpHist1_8(const Mat & img_gray, Mat & hist) {
	hist = Mat::zeros(1, 10, CV_32F);

	Mat_<uchar> img(img_gray);
	for (int r = 1; r < img.rows - 1; ++r) {
		for (int c = 1; c < img.cols - 1; ++c) {
			uchar uv = lbp(img, c, r);
			hist.at<float>(0, riuTable18[uv])++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 2) * (img.cols - 2));
}

void rlbpHist2_8_step0(const cv::Mat& img_gray, cv::Mat& hist) {
	hist = Mat::zeros(1, 256, CV_32F);
	Mat_<uchar> img(img_gray);
	for (int r = 2; r < img.rows - 2; ++r) {
		for (int c = 2; c < img.cols - 2; ++c) {
			uchar v = 0;
			uchar p = img(r, c);
			v += (img(r - 2, c) > p) << 0;
			v += (img(r - 2, c + 2) > p) << 1;
			v += (img(r, c + 2) > p) << 2;
			v += (img(r + 2, c + 2) > p) << 3;
			v += (img(r + 2, c) > p) << 4;
			v += (img(r + 2, c - 2) > p) << 5;
			v += (img(r, c - 2) > p) << 6;
			v += (img(r - 2, c - 2) > p) << 7;
			hist.at<float>(0, v)++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 4) * (img.cols - 4));
}

void rlbpHist2_8_step1(const cv::Mat& img_gray, cv::Mat& hist) {
	hist = Mat::zeros(1, 256, CV_32F);
	Mat_<uchar> img(img_gray);
	for (int r = 2; r < img.rows - 2; ++r) {
		for (int c = 2; c < img.cols - 2; ++c) {
			uchar v = 0;
			uchar p = img(r, c);
			v += (img(r - 2, c + 1) > p) << 0;
			v += (img(r - 1, c + 2) > p) << 1;
			v += (img(r + 1, c + 2) > p) << 2;
			v += (img(r + 2, c + 1) > p) << 3;
			v += (img(r + 2, c - 1) > p) << 4;
			v += (img(r + 1, c - 2) > p) << 5;
			v += (img(r - 1, c - 2) > p) << 6;
			v += (img(r - 2, c - 1) > p) << 7;
			hist.at<float>(0, v)++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 4) * (img.cols - 4));
}

void rlbpHist3_8_step0(const cv::Mat& img_gray, cv::Mat& hist) {
	hist = Mat::zeros(1, 256, CV_32F);
	Mat_<uchar> img(img_gray);
	for (int r = 3; r < img.rows - 3; ++r) {
		for (int c = 3; c < img.cols - 3; ++c) {
			uchar v = 0;
			uchar p = img(r, c);
			v += (img(r - 3, c) > p) << 0;
			v += (img(r - 3, c + 3) > p) << 1;
			v += (img(r, c + 3) > p) << 2;
			v += (img(r + 3, c + 3) > p) << 3;
			v += (img(r + 3, c) > p) << 4;
			v += (img(r + 3, c - 3) > p) << 5;
			v += (img(r, c - 3) > p) << 6;
			v += (img(r - 3, c - 3) > p) << 7;
			hist.at<float>(0, v)++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 6) * (img.cols - 6));
}

void rlbpHist3_8_step1(const cv::Mat& img_gray, cv::Mat& hist) {
	hist = Mat::zeros(1, 256, CV_32F);
	Mat_<uchar> img(img_gray);
	for (int r = 3; r < img.rows - 3; ++r) {
		for (int c = 3; c < img.cols - 3; ++c) {
			uchar v = 0;
			uchar p = img(r, c);
			v += (img(r - 3, c + 1) > p) << 0;
			v += (img(r - 2, c + 3) > p) << 1;
			v += (img(r + 1, c + 3) > p) << 2;
			v += (img(r + 3, c + 2) > p) << 3;
			v += (img(r + 3, c - 1) > p) << 4;
			v += (img(r + 2, c - 3) > p) << 5;
			v += (img(r - 1, c - 3) > p) << 6;
			v += (img(r - 3, c - 2) > p) << 7;
			hist.at<float>(0, v)++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 6) * (img.cols - 6));
}

void rlbpHist3_8_step2(const cv::Mat& img_gray, cv::Mat& hist) {
	hist = Mat::zeros(1, 256, CV_32F);
	Mat_<uchar> img(img_gray);
	for (int r = 3; r < img.rows - 3; ++r) {
		for (int c = 3; c < img.cols - 3; ++c) {
			uchar v = 0;
			uchar p = img(r, c);
			v += (img(r - 3, c + 2) > p) << 0;
			v += (img(r - 1, c + 3) > p) << 1;
			v += (img(r + 2, c + 3) > p) << 2;
			v += (img(r + 3, c + 1) > p) << 3;
			v += (img(r + 3, c - 2) > p) << 4;
			v += (img(r + 1, c - 3) > p) << 5;
			v += (img(r - 2, c - 3) > p) << 6;
			v += (img(r - 3, c - 1) > p) << 7;
			hist.at<float>(0, v)++; // incr. the resp. histogram bin
		}
	}
	hist /= ((img.rows - 6) * (img.cols - 6));
}
}

