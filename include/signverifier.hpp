/*
 * signverifier.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: thienlong
 */

#ifndef SIGNVERIFIER_HPP_
#define SIGNVERIFIER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <unordered_map>
#include <string>
#include <vector>

namespace signverify {

class SignatureVerifier {
public:
	virtual void train(const std::vector<cv::Mat>& src, cv::Mat& labels) = 0;
	virtual float verify(const cv::Mat& sign, ulong userID = 0) const = 0;
	virtual void load(const std::string& filename) = 0;
	virtual void save(const std::string& filename) const = 0;
//	virtual ~SignatureVerifier();
};

typedef void (*FeatureExtracter)(const cv::Mat& src, cv::Mat& output);
#define GRID_X 20
#define GRID_Y 10
#define POLAR_CORNER_ANGULAR 3
#define POLAR_CENTER_ANGULAR 12
#define POLAR_DISTANCE 15

void lbpGrid(const cv::Mat& src, cv::Mat& output);
void hogGrid(const cv::Mat& src, cv::Mat& output);
void hogPolar(const cv::Mat& src, cv::Mat& output);

class UserVerifier : public SignatureVerifier {
private:
	ulong id;
	cv::SVM model;
	FeatureExtracter extracter;
public:
	UserVerifier(FeatureExtracter extracter);
	virtual void train(const std::vector<cv::Mat>& src, cv::Mat& labels);
	virtual float verify(const cv::Mat& sign, ulong userID = 0) const;
	virtual void load(const std::string& filename);
	virtual void save(const std::string& filename) const;
};

class GlobalVerifier : public SignatureVerifier {
private:
	std::unordered_map<ulong, int> userIndex;
	std::vector<cv::Mat> references;
	cv::SVM model;
	FeatureExtracter extracter;
public:
	GlobalVerifier(FeatureExtracter extracter);
	virtual void train(const std::vector<cv::Mat>& src, cv::Mat& labels);
	virtual void addRefs(const std::vector<cv::Mat>& src, cv::Mat& labels);
	virtual float verify(const cv::Mat& sign, ulong userID = 0) const;
	virtual void load(const std::string& filename);
	virtual void save(const std::string& filename) const;
};

}



#endif /* SIGNVERIFIER_HPP_ */
