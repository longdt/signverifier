//============================================================================
// Name        : signature-verifier.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

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
using signverify::UserVerifier;
using namespace cv;
using namespace std;

bool startsWith(const string& haystack, const string& needle) {
    return needle.length() <= haystack.length()
        && equal(needle.begin(), needle.end(), haystack.begin());
}

void loadUData(ulong id, vector<Mat>& data, Mat& labels) {
	stringstream ss;
	ss << setw(3) << setfill('0') << id;
	string prefix = ss.str() + "_";
	path p("/home/thienlong/Downloads/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine");
	if (!exists(p) || !is_directory(p)) {
		return;
	}
	vector<path> files;
	std::copy(directory_iterator(p), directory_iterator(), std::back_inserter(files));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(files.begin(), files.end(), std::default_random_engine(seed));
    int refCnt = 0;
    int forgCnt = 0;
    long label = 0;
    int maxCnt = 10;
    labels = Mat::zeros(1, maxCnt * 2, CV_32SC1);
    for (auto iter = files.begin(), iterend = files.end(); iter != iterend; ++iter) {
    	string file = iter->string();
    	string fileName = iter->filename().string();
    	label = startsWith(fileName, prefix) ? id : -id;
    	if ((label == id && refCnt == maxCnt) || (label == -id && forgCnt == maxCnt)) {
    		continue;
    	}
    	data.push_back(imread(file, 0));
    	labels.at<int>(0, data.size() - 1) = label;
    	if (label == id) {
    		++refCnt;
    	} else {
    		++forgCnt;
    	}
    }
}

int umain() {
	UserVerifier verifier(lbpGrid);
	vector<Mat> data;
	Mat labels;
	ulong id = 4;
	loadUData(id, data, labels);
	verifier.train(data, labels);
	//test
	for (uint i = 0; i < data.size(); ++i) {
		Mat sign = data[i];
		imshow("sign", sign);
		cout << verifier.verify(sign, id) << "\texpected: " << labels.at<int>(0, i) << endl;
		waitKey(0);
	}
	cout << "test lan 2" <<endl;
	data.clear();
	loadUData(id, data, labels);
	for (uint i = 0; i < data.size(); ++i) {
		Mat sign = data[i];
		imshow("sign", sign);
		cout << verifier.verify(sign, id) << "\texpected: " << labels.at<int>(0, i) << endl;
		waitKey(0);
	}
}
