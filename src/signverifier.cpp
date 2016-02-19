/*
 * signverifier.cpp
 *
 *  Created on: Dec 18, 2015
 *      Author: thienlong
 */

#include "signverifier.hpp"
#include <stdexcept>

#include <cmath>


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
	return DISTANCE_TO_PROB(distance);
}

void UserVerifier::load(const string& filename) {

}

void UserVerifier::save(const string& filename) const {

}

GlobalVerifier::GlobalVerifier(FeatureExtracter extracter) : references(), model(), extracter(extracter) {
}

//@tested
void computeStdDev(const Mat& src, Mat& sd) {
    cv::Mat meanValue;
    cv::Mat stdValue;
    sd.create(1, src.cols, CV_32FC1);
    for (int i = 0; i < src.cols; i++){
        cv::meanStdDev(src.col(i), meanValue, stdValue);
        sd.at<float>(i) = stdValue.at<double>(0);
        if (sd.at<float>(i) == 0) {
        	sd.at<float>(i) = 0.0001f;
        }
    }
}

inline void diffFeature(const Mat& feature, const Mat& ref, Mat& dst, const Mat& sigma) {
	absdiff(feature, ref, dst);
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
				Mat diffeat;
				diffFeature(feature, refs.row(i), diffeat, sigma);
				data.push_back(diffeat);
				newLabels.push_back(-1);
			}
		}
		//diff vector of references
		for (int i = 0; i < refs.rows - 2; ++i) {
			Mat ref = refs.row(i);
			for (int j = i + 1; j < refs.rows - 1; ++j) {
				Mat diffeat;
				diffFeature(ref, refs.row(j), diffeat, sigma);
				data.push_back(diffeat);
				newLabels.push_back(1);
			}
		}
	}
	model.train(data, newLabels);
}

//@tested
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
	float sumScore = 0;
	Mat diffeat;
	Mat sigma = refs.row(refs.rows - 1);
	for (int i = 0; i < refs.rows - 1; ++i) {
		diffFeature(feature, refs.row(i), diffeat, sigma);
		float distance = model.predict(diffeat, true);
		prob = DISTANCE_TO_PROB(distance);
		maxScore = max(prob, maxScore);
		sumScore += prob;
	}
	return sumScore / (refs.rows - 1);
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
}
