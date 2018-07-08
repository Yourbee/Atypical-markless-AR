﻿#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 30;
const float GOOD_PORTION = 0.05f;

int64 work_begin = 0;
int64 work_end = 0;

static void workBegin()
{
	work_begin = getTickCount();
}

static void workEnd()
{
	work_end = getTickCount() - work_begin;
}

static double getTime()
{
	return work_end / ((double)getTickFrequency())* 1000.;
}

struct ORBDetector
{
	Ptr<Feature2D> orb;
	ORBDetector(double hessian = 800.0)
	{
		orb = ORB::create(hessian);
	}
	template<class T>
	//void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
	//{
	//	orb->detectAndCompute(in, mask, pts, descriptors, useProvided);
	//}
	void operator()(const T& in, std::vector<cv::KeyPoint>& pts, const T& mask)
	{
		orb->detect(in, pts, mask);
	}
	template<class T>
	void operator()(const T& in, std::vector<cv::KeyPoint>& pts, T& descriptors)
	{
		orb->compute(in, pts, descriptors);
	}
};

template<class KPMatcher>
struct  ORBMatcher
{
	KPMatcher matcher;
	template<class T>
	void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
	{
		matcher.match(in1, in2, matches);
	}
};

static Mat drawGoodMatches(
	const Mat& img1,
	const Mat& img2,
	const std::vector<KeyPoint>& keypoints1,
	const std::vector<KeyPoint>& keypoints2,
	std::vector<DMatch>& matches,
	std::vector<Point2f>& scene_corners_
)
{
	//-- Sort matches and preserve top 10% matches
	std::sort(matches.begin(), matches.end());//按字典顺序对矩阵行列进行排序
	std::vector< DMatch > good_matches;
	double minDist = matches.front().distance;
	double maxDist = matches.back().distance;

	const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}
	std::cout << "\nMax distance: " << maxDist << std::endl;
	std::cout << "Min distance: " << minDist << std::endl;

	std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

	// drawing the results
	Mat img_matches;

	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(img1.cols, 0);
	obj_corners[2] = Point(img1.cols, img1.rows);
	obj_corners[3] = Point(0, img1.rows);
	std::vector<Point2f> scene_corners(4);

	Mat H = findHomography(obj, scene, RANSAC);
	perspectiveTransform(obj_corners, scene_corners, H);

	scene_corners_ = scene_corners;

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches,
		scene_corners[0] + Point2f((float)img1.cols, 0), scene_corners[1] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[1] + Point2f((float)img1.cols, 0), scene_corners[2] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[2] + Point2f((float)img1.cols, 0), scene_corners[3] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[3] + Point2f((float)img1.cols, 0), scene_corners[0] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	return img_matches;
}

////////////////////////////////////////////////////
// This program demonstrates the usage of ORB_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
	const char* keys =
		"{ h help      |                  | print help message  }"
		"{ l left      | box.jpg          | specify left image  }"
		"{ r right     | box_in_scene.jpg | specify right image }"
		"{ o output    | SURF_output.jpg  | specify output save path }"
		"{ m cpu_mode  |                  | run without OpenCL }"
		"{ v video_file|       | Input from video file, if ommited, input comes from camera }"
		"{ ci          | 0     | Camera id if input doesnt come from video (-v) }";

	CommandLineParser cmd(argc, argv, keys);
	CommandLineParser parser(argc, argv, keys);
	if (cmd.has("help"))
	{
		std::cout << "Usage: surf_matcher [options]" << std::endl;
		std::cout << "Available options:" << std::endl;
		cmd.printMessage();
		return EXIT_SUCCESS;
	}
	if (cmd.has("cpu_mode"))
	{
		ocl::setUseOpenCL(false);
		std::cout << "OpenCL was disabled" << std::endl;
	}
	int camId = cmd.get<int>("ci");
	String video;
	if (cmd.has("video_file")) {
		video = cmd.get<String>("video_file");
	}


	VideoCapture inputVideo;
	int waitTime;
	if (!video.empty()) {
		inputVideo.open(video);
		waitTime = 0;
	}
	else {
		inputVideo.open(camId);
		waitTime = 10;
	}
	
	UMat img1, img2;
	std::string leftName = cmd.get<std::string>("l");
	imread(leftName, IMREAD_GRAYSCALE).copyTo(img1);
	if (img1.empty())
	{
		std::cout << "Couldn't load " << leftName << std::endl;
		cmd.printMessage();
		return EXIT_FAILURE;
	}

	//declare input/output
	std::vector<KeyPoint> keypoints1, keypoints2;
	std::vector<DMatch> matches;

	UMat _descriptors1, _descriptors2;
	Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
		descriptors2 = _descriptors2.getMat(ACCESS_RW);

	//instantiate detectors/matchers
	ORBDetector orb;

	ORBMatcher<BFMatcher> matcher;

	orb(img1.getMat(ACCESS_READ), keypoints1, Mat());//图像，特征点，描述子
	KeyPointsFilter::removeDuplicated(keypoints1);//移除重复点
	//KeyPointsFilter::retainBest(keypoints1, 500);
	orb(img1.getMat(ACCESS_READ), keypoints1, descriptors1);//图像，特征点，描述子


	while (inputVideo.grab()) {

		inputVideo.retrieve(img2);
		//std::string outpath = cmd.get<std::string>("o");



		//std::string rightName = cmd.get<std::string>("r");
		//imread(rightName, IMREAD_GRAYSCALE).copyTo(img2);
		//if (img2.empty())
		//{
		//	std::cout << "Couldn't load " << rightName << std::endl;
		//	cmd.printMessage();
		//	return EXIT_FAILURE;
		//}

		double orb_time = 0.;

		////declare input/output
		//std::vector<KeyPoint> keypoints1, keypoints2;
		//std::vector<DMatch> matches;

		//UMat _descriptors1, _descriptors2;
		//Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
		//	descriptors2 = _descriptors2.getMat(ACCESS_RW);

		////instantiate detectors/matchers
		//ORBDetector orb;

		//ORBMatcher<BFMatcher> matcher;

		//-- start of timing section

		//for (int i = 0; i <= LOOP_NUM; i++)
		//{
		//	if (i == 1) workBegin();
		////orb(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);//图像，特征点，描述子
		////orb(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
		////orb(img1.getMat(ACCESS_READ), keypoints1, Mat());//图像，特征点，描述子
		orb(img2.getMat(ACCESS_READ), keypoints2, Mat());
		////KeyPointsFilter::removeDuplicated(keypoints1);//移除重复点
		KeyPointsFilter::removeDuplicated(keypoints2);
		////KeyPointsFilter::retainBest(keypoints1, 300);
		KeyPointsFilter::retainBest(keypoints2, 500);
		////orb(img1.getMat(ACCESS_READ), keypoints1, descriptors1);//图像，特征点，描述子
		orb(img2.getMat(ACCESS_READ), keypoints2, descriptors2);


		matcher.match(descriptors1, descriptors2, matches);
		//}
		/*workEnd();*/
		std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
		std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

		orb_time = getTime();
		std::cout << "ORB run time: " << orb_time / LOOP_NUM << " ms" << std::endl << "\n";


		std::vector<Point2f> corner;
		Mat img_matches = drawGoodMatches(img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), keypoints1, keypoints2, matches, corner);

		//-- Show detected matches

		namedWindow("orb matches", 0);
		imshow("orb matches", img_matches);
		//imwrite(outpath, img_matches);
		char key = (char)waitKey(waitTime);
		if (key == 27) break;

	}
	return EXIT_SUCCESS;
}
	