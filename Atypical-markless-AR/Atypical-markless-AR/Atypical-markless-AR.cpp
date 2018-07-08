#include <iostream>
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
#include <opencv2/aruco.hpp>

using namespace cv;
//using namespace std;
using namespace cv::xfeatures2d;

const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 30;
const float GOOD_PORTION = 0.05f;
const float markerLength = 0.1;

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

static bool readCameraParameters(std::string filename, Mat &camMatrix, Mat &distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
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
	template<class T>
	void knnmatch(const T& in1, const T& in2, std::vector<std::vector<cv::DMatch>>& matches,const int k)
	{
		matcher.knnMatch(in1, in2, matches, k);
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
	std::sort(matches.begin(), matches.end());
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

static Mat drawGoodMatches(
	const Mat& img1,
	const Mat& img2,
	const std::vector<KeyPoint>& keypoints1,
	const std::vector<KeyPoint>& keypoints2,
	std::vector<std::vector<cv::DMatch>>& matchess,
	int& flag,
	std::vector<Point2f>& scene_corners_,
	const int k
)
{
	//-- Sort matches and preserve top 10% matches
	/*const float minRatio = 1.f / 1.5f;*/
	const float minRatio = 0.6;
	flag = 1;
	std::vector< DMatch > good_matchess;

	for (size_t i = 0; i < matchess.size(); i++) {
		const DMatch& bestMatch = matchess[i][0];
		const DMatch& betterMatch = matchess[i][1];

		float  distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio)
			good_matchess.push_back(bestMatch);
	}
	printf_s("点%d\n",matchess.size());
	printf_s("好点%d\n",good_matchess.size());

	// drawing the results
	Mat img_matchess;

	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matchess, img_matchess, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	if (good_matchess.size()<4) {
		//img_matchess=img2;
		flag = 0;
		return img_matchess;
	}
	for (size_t i = 0; i < good_matchess.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matchess[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matchess[i].trainIdx].pt);
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
	line(img_matchess,
		scene_corners[0] + Point2f((float)img1.cols, 0), scene_corners[1] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matchess,
		scene_corners[1] + Point2f((float)img1.cols, 0), scene_corners[2] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matchess,
		scene_corners[2] + Point2f((float)img1.cols, 0), scene_corners[3] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matchess,
		scene_corners[3] + Point2f((float)img1.cols, 0), scene_corners[0] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	return img_matchess;
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
		"{ o output    | ORB_output.jpg   | specify output save path }"
		"{ m cpu_mode  |                  | run without OpenCL }"
		"{ v video_file|                  | Input from video file, if ommited, input comes from camera }"
		"{ ci          | 0				  | Camera id if input doesnt come from video (-v) }"
		"{ c            | calib.yml        | Camera intrinsic parameters. Needed for camera pose }";

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

	bool estimatePose = parser.has("c");
	Mat camMatrix, distCoeffs;
	if (estimatePose) {
		bool readOk = readCameraParameters(parser.get<std::string>("c"), camMatrix, distCoeffs);
		if (!readOk) {
			std::cerr << "Invalid camera file" << std::endl;
			return 0;
		}
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
		std::string outpath = cmd.get<std::string>("o");
		double orb_time = 0.;

		std::vector<DMatch> matches;
		std::vector< std::vector<DMatch> > matchess;
		//-- start of timing section

		orb(img2.getMat(ACCESS_READ), keypoints2, Mat());
		////KeyPointsFilter::removeDuplicated(keypoints1);//移除重复点
		KeyPointsFilter::removeDuplicated(keypoints2);
		orb(img2.getMat(ACCESS_READ), keypoints2, descriptors2);

		matcher.match(descriptors1, descriptors2, matches);
		matcher.knnmatch(descriptors1, descriptors2, matchess, 2);
		
		//}
		/*workEnd();*/
		std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
		std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

		orb_time = getTime();
		std::cout << "ORB run time: " << orb_time / LOOP_NUM << " ms" << std::endl << "\n";

		std::vector<Point2f> corner;
		int flag = 0;
		//Mat img_matches = drawGoodMatches(img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), keypoints1, keypoints2, matches,corner);
		Mat img_matchess = drawGoodMatches(img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), keypoints1, keypoints2, matchess,flag,corner,2);
		//-- Show detected matches

		//namedWindow("orb matches", 0);
		//imshow("orb matches", img_matches);
		namedWindow("orb matches_knn", 0);
		imshow("orb matchess", img_matchess);

		////estimate pose
		//// Image dimensions
		//const float w = img1.cols;
		//const float h = img1.rows;

		//// Normalized dimensions:
		//const float maxSize = std::max(w, h);
		//const float unitW = w / maxSize;
		//const float unitH = h / maxSize;
		//std::vector< Vec3d > rvecs, tvecs;
		//std::vector<Point3f> worldpoints(4);
		//worldpoints[0] = Point3f(-unitW, -unitH, 0);
		//worldpoints[1] = Point3f(unitW, -unitH, 0);
		//worldpoints[2] = Point3f(unitW, unitH, 0);
		//worldpoints[3] = Point3f(-unitW, unitH, 0);
		//if (estimatePose&flag) {
		//	solvePnP(worldpoints, corner, camMatrix, distCoeffs, rvecs, tvecs);
		//	//aruco::estimatePoseSingleMarkers(corner, markerLength, camMatrix, distCoeffs, rvecs,
		//		//tvecs);////rvecs旋转矩阵；tvecs平移向量
		//	aruco::drawAxis(img_matchess, camMatrix, distCoeffs, rvecs, tvecs,
		//		markerLength * 0.5f);
		//}



		//imwrite(outpath, img_matches);
		char key = (char)waitKey(waitTime);
		if (key == 27) break;

	}
	return EXIT_SUCCESS;
}
	