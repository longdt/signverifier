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
using namespace signverify;
using namespace cv;
using namespace std;

void loadMUIDData(const path& p, long id, vector<Mat>& data, Mat& labels);

void loadUData(ulong id, vector<Mat>& data, Mat& labels) {
	stringstream ss;
	ss << setw(3) << setfill('0') << id;
	string uidFolder = ss.str();
	path p("/home/thienlong/Downloads/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Reference(646)/" + uidFolder);
	if (!exists(p) || !is_directory(p)) {
		return;
	}
	loadMUIDData(p, id, data, labels);
}

int umain() {
	UserVerifier verifier(lbpGrid);
	vector<Mat> data;
	Mat labels;
	ulong id = 18 ;
	loadUData(id, data, labels);
	verifier.train(data, labels);
	//test
	for (uint i = 0; i < data.size(); ++i) {
		Mat sign = data[i];
//		imshow("sign", sign);
		cout << verifier.verify(sign, id) << "\texpected: " << labels.at<int>(0, i) << endl;
//		waitKey(0);
	}
	cout << "test lan 2" <<endl;
	data.clear();
	labels = Mat();
	loadUData(id, data, labels);
	for (uint i = 0; i < data.size(); ++i) {
		Mat sign = data[i];
//		imshow("sign", sign);
		cout << verifier.verify(sign, id) << "\texpected: " << labels.at<int>(0, i) << endl;
//		waitKey(0);
	}

	cout << "test lan 3" <<endl;
	path p("/home/thienlong/Downloads/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Questioned(1287)/0" + std::to_string(id));
	for (auto&& file : directory_iterator(p)) {
		Mat sign = imread(file.path().string(), 0);
//		imshow("sign", sign);
		float score = verifier.verify(sign, id);
		cout << file.path().string() << ":\t" << id << "\t" << score << endl;
	}

}
