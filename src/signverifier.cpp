/*
 * signverifier.cpp
 *
 *  Created on: Dec 18, 2015
 *      Author: thienlong
 */

#include "signverifier.hpp"
#include <stdexcept>

#include "lbp.hpp"
using namespace std;
using namespace cv;

namespace signverify {

UserVerifier::UserVerifier(FeatureExtracter extracter) : id(0) {
	this->extracter = extracter;
}

void UserVerifier::train(const vector<Mat>& src, cv::Mat& labels) {
	Mat data;
	for (uint i = 0; i < src.size(); ++i) {
		Mat img = src[i];
		Mat feature;
		extracter(img, feature);
		data.push_back(feature);
	}
	model.train(data, labels);
}

float UserVerifier::verify(const cv::Mat& sign, ulong userID) const {
	if (id != userID) {
		throw logic_error("doesn't support userid: " + userID);
	}
	Mat feature;
	extracter(sign, feature);
	return model.predict(feature);
}

void UserVerifier::load(const string& filename) {

}

void UserVerifier::save(const string& filename) const {

}

void GlobalVerifier::train(const vector<Mat>& src, cv::Mat& labels) {
	Mat data;
	for (uint i = 0; i < src.size(); ++i) {
		Mat img = src[i];
		Mat feature;
		extracter(img, feature);
		data.push_back(feature);
	}
}

void GlobalVerifier::update(const std::vector<cv::Mat>& src, cv::Mat& labels) {

}

float GlobalVerifier::verify(const cv::Mat& sign, ulong userID) const {
	auto idx = userIndex.find(userID);
	if (idx == userIndex.end()) {
		throw logic_error("doesn't support userid: " + userID);
	}
	Mat feature;
	Mat refs = references[idx->second];
	extracter(sign, feature);
	float score = -1;
	float maxScore = -1;
	Mat diffFeature;
	Mat sigma = refs.row(refs.rows - 1);
	for (int i = 0; i < refs.rows - 1; ++i) {
		diffFeature = feature - refs.row(i);
		diffFeature = diffFeature / sigma;
		score = model.predict(diffFeature);
		maxScore = max(score, maxScore);
	}
	return maxScore;
}

void GlobalVerifier::load(const string& filename) {

}

void GlobalVerifier::save(const string& filename) const {

}

void lbpGrid(const cv::Mat& src, cv::Mat& output) {
	spatialUniLbpHist(src, output, GRID_X, GRID_Y);
}

void hogGrid(const cv::Mat& src, cv::Mat& output);

void hogPolar(const cv::Mat& src, cv::Mat& output);
}
