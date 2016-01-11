/*
 * gsign-verifier.cpp
 *
 *  Created on: Jan 6, 2016
 *      Author: thienlong
 */


#include <opencv2/highgui/highgui.hpp>

#include "signverifier.hpp"
#include <sstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <random>       // std::default_random_engine
#include <chrono>

using boost::filesystem::path;
using boost::filesystem::directory_iterator;
using signverify::lbpGrid;
using signverify::GlobalVerifier;
using namespace cv;
using namespace std;

void loadUIDData(const path& p, long id, vector<Mat>& data, Mat& labels) {
	if (!exists(p) || !is_directory(p)) {
		return;
	}
	vector<path> files;
	for (auto&& x : directory_iterator(p))
		files.push_back(x.path());
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(files.begin(), files.end(), std::default_random_engine(seed));
    int refCnt = 0;
    int forgCnt = 0;
    int label = 0;
    int maxRefs = 10;
    int maxForg = 4;
    Mat temp(1, maxRefs + maxForg, CV_32SC1);
    int idx = 0;
    for (auto iter = files.begin(), iterend = files.end(); iter != iterend; ++iter) {
    	string file = iter->string();
    	string fileName = iter->filename().string();
    	label = fileName.length() == 10 ? id : -id;
    	if ((label == id && refCnt == maxRefs) || (label == -id && forgCnt == maxForg)) {
    		continue;
    	}
    	Mat sign = imread(file, 0);
    	if (sign.empty()) {
    		continue;
    	}

    	data.push_back(sign);
    	temp.at<int>(0, idx++) = label;
    	if (label == id) {
    		++refCnt;
    	} else {
    		++forgCnt;
    	}
    }
    labels.push_back(temp);
}

void loadGData(vector<Mat>& data, Mat& labels) {
	path p("/home/thienlong/Downloads/trainingSet/OfflineSignatures/Dutch/TrainingSet");
	if (!exists(p) || !is_directory(p)) {
		return;
	}
	vector<path> uids;
	for (auto&& x : directory_iterator(p))
		uids.push_back(x.path());
	unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(uids.begin(), uids.end(), std::default_random_engine(seed));
	for (int i = 0; i < uids.size(); ++i) {
		path idpath = uids[i];
		string uid = idpath.filename().string();
		long id = stol(uid);
		loadUIDData(idpath, id, data, labels);
	}
}

int main() {
	GlobalVerifier verifier(lbpGrid);
	vector<Mat> data;
	Mat labels;
	loadGData(data, labels);
	verifier.train(data, labels);
	//test
	int idx = 0;
	for (int r = 0; r < labels.rows; ++r) {
		for (int c = 0; c < labels.cols; ++c) {
			Mat sign = data[idx];
			imshow("sign", sign);
			int id = labels.at<int>(r, c);
			if (id < 0) {
				id = -id;
			}
			cout << verifier.verify(sign, id) << "\texpected: " << labels.at<int>(r, c) << endl;
			waitKey(0);
			++idx;
		}
	}

	cout << "test lan 2" <<endl;
	idx = 0;
	data.clear();
	labels = Mat();
	loadGData(data, labels);
	for (int r = 0; r < labels.rows; ++r) {
		for (int c = 0; c < labels.cols; ++c) {
			Mat sign = data[idx];
			imshow("sign", sign);
			int id = labels.at<int>(r, c);
			if (id < 0) {
				id = -id;
			}
			cout << verifier.verify(sign, id) << "\texpected: " << labels.at<int>(r, c) << endl;
			waitKey(0);
			++idx;
		}
	}
}
