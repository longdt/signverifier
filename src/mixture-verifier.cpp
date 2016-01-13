/*
 * mixture-verifier.cpp
 *
 *  Created on: Jan 13, 2016
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
using namespace signverify;
using namespace cv;
using namespace std;

void loadGData(vector<Mat>& data, Mat& labels);


void loadMUIDData(const path& p, long id, vector<Mat>& data, Mat& labels) {
	if (!exists(p) || !is_directory(p)) {
		return;
	}
	vector<path> files;
	for (auto&& x : directory_iterator(p))
		files.push_back(x.path());
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rd(seed);
	std::shuffle(files.begin(), files.end(), rd);
    int refCnt = 0;
    int forgCnt = 0;
    int maxRefs = 10;
    int maxForg = 10;
    Mat temp(1, maxRefs + maxForg, CV_32SC1);
    int idx = 0;
    //add refs
    for (auto iter = files.begin(), iterend = files.end(); iter != iterend; ++iter) {
    	if (refCnt == maxRefs) {
    		break;
    	}
    	Mat sign = imread(iter->string(), 0);
    	if (sign.empty()) {
    		continue;
    	}

    	data.push_back(sign);
    	temp.at<int>(0, idx++) = id;
		++refCnt;
    }
    //add random forgeries
    vector<path> forgeries;
    for (auto&& x : directory_iterator(p.parent_path())) {
    	path otherUID = x.path();
    	if (otherUID == p) {
    		continue;
    	}
    	std::copy(directory_iterator(otherUID), directory_iterator(), std::back_inserter(forgeries));
    }
    std::shuffle(forgeries.begin(), forgeries.end(), rd);
    for (auto iter = forgeries.begin(), iterend = forgeries.end(); iter != iterend; ++iter) {
    	if (forgCnt == maxForg) {
    		break;
    	}
    	Mat sign = imread(iter->string(), 0);
    	if (sign.empty()) {
    		continue;
    	}

    	data.push_back(sign);
    	temp.at<int>(0, idx++) = -id;
		++forgCnt;
    }
    labels.push_back(temp);
}

void loadMData(vector<Mat>& data, Mat& labels) {
	path p("/home/thienlong/Downloads/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Reference(646)");
	if (!exists(p) || !is_directory(p)) {
		return;
	}
	vector<path> uids;
	for (auto&& x : directory_iterator(p))
		uids.push_back(x.path());
	for (int i = 0; i < uids.size(); ++i) {
		path idpath = uids[i];
		string uid = idpath.filename().string();
		long id = stol(uid);
		loadMUIDData(idpath, id, data, labels);
	}
}

int mmain() {
	MixtureVerifier verifier;
	vector<Mat> data;
	Mat labels;
	loadGData(data, labels);
	verifier.trainGlobal(data, labels);
	data.clear();
	labels = Mat();
	loadMData(data, labels);
	verifier.train(data, labels);
	//test
	path p("/home/thienlong/Downloads/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Questioned(1287)");
	for (auto&& x : directory_iterator(p)) {
		string uid = x.path().filename().string();
		ulong id = stol(uid);
		for (auto&& file : directory_iterator(x.path())) {
			Mat sign = imread(file.path().string());
			cout << file.path().string() << ":\t" << id << "\t" << verifier.verify(sign, id) << endl;
		}
	}
}
