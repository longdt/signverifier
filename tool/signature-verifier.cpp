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

int g_acceptCnt = 0;
int g_faCnt = 0;
int g_rejectCnt = 0;
int g_frCnt = 0;

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

int testID(ulong id) {
	UserVerifier verifier(riuLbpGrid);
	vector<Mat> data;
	Mat labels;
	loadUData(id, data, labels);
	verifier.train(data, labels);
	int acceptCnt = 0;
	int faCnt = 0;
	int rejectCnt = 0;
	int frCnt = 0;
	path p("/home/thienlong/Downloads/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Questioned(1287)/0" + std::to_string(id));
	for (auto&& file : directory_iterator(p)) {
		Mat sign = imread(file.path().string(), 0);
//		imshow("sign", sign);
		float score = verifier.verify(sign, id);
		if (file.path().filename().string().length() > 10) {
			++rejectCnt;
			++g_rejectCnt;
			if (score > 0) {
				++frCnt;
				++g_frCnt;
			}
		} else {
			++acceptCnt;
			++g_acceptCnt;
			if (score < 0) {
				++faCnt;
				++g_faCnt;
			}
		}
		cout << file.path().string() << ":\t" << id << "\t" << score << endl;
	}
	cout << "FAR: " << faCnt / (float) acceptCnt << "\tFRR: " << frCnt / (float) rejectCnt << endl;
}

int main() {
	path p("/home/thienlong/Downloads/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Reference(646)");
	if (!exists(p) || !is_directory(p)) {
		return 0;
	}
	for (auto&& x : directory_iterator(p)) {
		string uid = x.path().filename().string();
		ulong id = stol(uid);
		testID(id);
	}
	cout << "TFAR: " << g_faCnt / (float) g_acceptCnt << "\tTFRR: " << g_frCnt / (float) g_rejectCnt << endl;
	return 0;
}
