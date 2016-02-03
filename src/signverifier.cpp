/*
 * signverifier.cpp
 *
 *  Created on: Dec 18, 2015
 *      Author: thienlong
 */

#include "signverifier.hpp"
#include <stdexcept>

#include "lbp.hpp"
#include <cmath>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

namespace signverify {
//#define DISTANCE_TO_PROB(distance) (1.0f / (1.0f + exp(distance * 6)))
#define DISTANCE_TO_PROB(distance) (-distance)

UserVerifier::UserVerifier(FeatureExtracter extracter) : id(0), model(), extracter(extracter) {
}

void UserVerifier::train(const vector<Mat>& src, cv::Mat& labels) {
	Mat data;
	for (uint i = 0; i < src.size(); ++i) {
		Mat img = src[i];
		Mat feature;
		extracter(img, feature);
		data.push_back(feature);
		if (id == 0 && labels.at<int>(0, i) > 0) {
			id = labels.at<int>(0, i);
		}
	}
	Mat newLabels;
	transpose(labels, newLabels);
	model.train(data, newLabels);
}

float UserVerifier::verify(const cv::Mat& sign, ulong userID) const {
	if (id != userID) {
		throw logic_error("doesn't support userid: " + userID);
	}
	Mat feature;
	extracter(sign, feature);
	long label = model.predict(feature);
	float distance = model.predict(feature, true);
//	prob = abs(prob);
//	return (label > 0) ? prob : (1 - prob);
	return DISTANCE_TO_PROB(distance);
}

void UserVerifier::load(const string& filename) {

}

void UserVerifier::save(const string& filename) const {

}

GlobalVerifier::GlobalVerifier(FeatureExtracter extracter) : references(), model(), extracter(extracter) {
}

void computeStdDev(const Mat& src, Mat& sd) {
    cv::Mat meanValue;
    cv::Mat stdValue;
    sd.create(1, src.cols, CV_32FC1);
    for (int i = 0; i < src.cols; i++){
        cv::meanStdDev(src.col(i), meanValue, stdValue);
        sd.at<float>(i) = stdValue.at<double>(0) + 0.0000000001f;
    }
}

void GlobalVerifier::train(const vector<Mat>& src, cv::Mat& labels) {
	if (src.empty() || labels.empty()) {
		return;
	}
	addRefs(src, labels);
	int imgIdx = -1;
	Mat data;
	Mat1i newLabels;
	for (int u = 0; u < labels.rows; ++u) {
		ulong userID = labels.at<int>(u, 0) > 0 ? labels.at<int>(u, 0) : - labels.at<int>(u, 0);
		Mat refs = references[userID];
		Mat sigma = refs.row(refs.rows - 1);
		//diff vector of forgeries
		for (int l = 0; l < labels.cols; ++l) {
			++imgIdx;
			if (labels.at<int>(u,l) >= 0) {
				continue;
			}
			Mat feature;
			extracter(src[imgIdx], feature);
			//generate diff vector
			for (int i = 0; i < refs.rows - 1; ++i) {
				Mat diffFeature = feature - refs.row(i);
				divide(diffFeature, sigma, diffFeature);
				data.push_back(diffFeature);
				newLabels.push_back(-1);
			}
		}
		//diff vector of references
		for (int i = 0; i < refs.rows - 2; ++i) {
			Mat ref = refs.row(i);
			for (int j = i + 1; j < refs.rows - 1; ++j) {
				Mat diffFeature = ref - refs.row(j);
				divide(diffFeature, sigma, diffFeature);
				data.push_back(diffFeature);
				newLabels.push_back(1);
			}
		}
	}
	model.train(data, newLabels);
}

void GlobalVerifier::addRefs(const std::vector<cv::Mat>& src, cv::Mat& labels) {
	if (!references.empty()) {
		references.clear();
	}
	int imgIdx = -1;
	for (int u = 0; u < labels.rows; ++u) {
		Mat refs;
		ulong userID = labels.at<int>(u,0) > 0 ? labels.at<int>(u,0) : - labels.at<int>(u,0);
		for (int l = 0; l < labels.cols; ++l) {
			++imgIdx;
			if (labels.at<int>(u,l) <= 0) {
				continue;
			}
			Mat feature;
			extracter(src[imgIdx], feature);
			refs.push_back(feature);
		}
		Mat sd;
		computeStdDev(refs, sd);
		refs.push_back(sd);
		references[userID] = refs;
	}
}

float GlobalVerifier::verify(const cv::Mat& sign, ulong userID) const {
	auto idx = references.find(userID);
	if (idx == references.end()) {
		throw logic_error("doesn't support userid: " + userID);
	}
	Mat feature;
	Mat refs = idx->second;
	extracter(sign, feature);
	float prob = -1;
	float maxScore = -9;
	Mat diffFeature;
	Mat sigma = refs.row(refs.rows - 1);
	for (int i = 0; i < refs.rows - 1; ++i) {
		diffFeature = feature - refs.row(i);
		diffFeature = diffFeature / sigma;
		float distance = model.predict(diffFeature, true);
		prob = DISTANCE_TO_PROB(distance);
		maxScore = max(prob, maxScore);
	}
	return maxScore;
}

void GlobalVerifier::load(const string& filename) {

}

void GlobalVerifier::save(const string& filename) const {

}

//mixture verifier
MixtureVerifier::MixtureVerifier() : ulbps(), uhogs(), glbp(lbpGrid), ghog(hogGrid) {

}

void MixtureVerifier::train(const std::vector<cv::Mat>& src, cv::Mat& labels) {
	ulbps.clear();
	uhogs.clear();
	for (int u = 0; u < labels.rows; ++u) {
		Mat ulabel = labels.row(u);
		ulong uid = ulabel.at<int>(0, 0) > 0 ? ulabel.at<int>(0, 0) : - ulabel.at<int>(0, 0);
		vector<Mat> data(src.begin() + u * labels.cols, src.begin() + (u + 1) * labels.cols);
		//train ulbps
		shared_ptr<UserVerifier> lbpVerifier = make_shared<UserVerifier>(lbpGrid);
		lbpVerifier->train(data, ulabel);
		ulbps[uid] = lbpVerifier;
		//train uhogs
		shared_ptr<UserVerifier> hogVerifier = make_shared<UserVerifier>(hogGrid);
		hogVerifier->train(data, ulabel);
		uhogs[uid] = hogVerifier;

	}
	glbp.addRefs(src, labels);
	ghog.addRefs(src, labels);
}

void MixtureVerifier::trainGlobal(const std::vector<cv::Mat>& src, cv::Mat& labels) {
	glbp.train(src, labels);
	ghog.train(src, labels);
}

float MixtureVerifier::verify(const cv::Mat& sign, ulong userID) const {
	auto idx = ulbps.find(userID);
	if (idx == ulbps.end()) {
		throw logic_error("doesn't support userid: " + userID);
	}
	float score = glbp.verify(sign, userID) + ghog.verify(sign, userID);
	shared_ptr<UserVerifier> verifier = idx->second;
	score += verifier->verify(sign, userID);
	idx = uhogs.find(userID);
	score += idx->second->verify(sign, userID);
	return score;
}

void MixtureVerifier::load(const std::string& filename) {

}

void MixtureVerifier::save(const std::string& filename) const {

}
//end mixture verifier

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

	int width = static_cast<int>(ceil(src.cols / (float) GRID_X));
	int height = static_cast<int>(ceil(src.rows / (float) GRID_Y));
	int padrigh = width * GRID_X - src.cols;
	int padbottom = height * GRID_Y - src.rows;
	Mat padSrc;
	copyMakeBorder(refineSrc, padSrc, 0, padbottom, 0, padrigh, BORDER_CONSTANT, Scalar(255));
//	imshow("padSrc", padSrc);
//	waitKey(0);
	spatialRiuLbpHist(padSrc, 2, 16, output, GRID_X, GRID_Y);
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
