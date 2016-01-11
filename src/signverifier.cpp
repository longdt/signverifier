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
	return model.predict(feature, true);
}

void UserVerifier::load(const string& filename) {

}

void UserVerifier::save(const string& filename) const {

}

GlobalVerifier::GlobalVerifier(FeatureExtracter extracter) : userIndex(), references(), model(), extracter(extracter) {
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
		Mat refs = references[userIndex[userID]];
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
				diffFeature = diffFeature / sigma;
				data.push_back(diffFeature);
				newLabels.push_back(-1);
			}
		}
		//diff vector of references
		for (int i = 0; i < refs.rows - 2; ++i) {
			Mat ref = refs.row(i);
			for (int j = i + 1; j < refs.rows - 1; ++j) {
				Mat diffFeature = ref - refs.row(j);
				diffFeature = diffFeature / sigma;
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
		userIndex.clear();
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
		references.push_back(refs);
		userIndex[userID] = references.size() - 1;
	}
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
