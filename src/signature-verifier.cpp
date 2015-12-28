//============================================================================
// Name        : signature-verifier.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <opencv2/highgui/highgui.hpp>

#include "signverifier.hpp"

using signverify::lbpGrid;
using signverify::UserVerifier;
using namespace cv;

int main() {
//	Mat src = imread("001_01.PNG", 0);
//	imshow("Image", src);
//	waitKey(0);
//	return 0;
	UserVerifier verifier(lbpGrid);
}
