#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <functional>
#include <string>
#include "openfile.h"
using namespace std;
using namespace cv;

//图片帧数
int framenum = 25;
int main()
{
	vector<vector<Point2f>>points[2];//左右图像平面的坐标值
	vector<vector<Point2f>>comparerectifiedpoints[2];//方法一校正后左右图像平面的坐标值
	vector<vector<Point2f>>rectifiedpoints[2];//方法二校正后左右图像平面的坐标值
	vector<Point2f>imgpoints[2];//每个图像平面的像素坐标值
	vector<Point3f>circles[2];//霍夫圆监测点
	multimap<int, Point2f>finalpoints[2];//排序结束后的像素坐标值
	multimap<int, Point2f>rectifiedfinalpoints[2];//校正结束后的像素坐标值
	int imgnum = 1;
	char filenameleft[1024];
	char filenameright[1024];
	Mat ssimgleft, ssimgright;//校正前初试图像
	Mat srcimgleft, srcimgright;
	Mat dstimgleft, dstimgright;
	Mat rectfieddst1, rectifieddst2;

	while (imgnum < framenum)
	{
		sprintf_s(filenameleft, "E:\\双目视觉\\T字形工件\\L_Z Total\\%d.jpg", imgnum);
		sprintf_s(filenameright, "E:\\双目视觉\\T字形工件\\R_Z Total\\%d.jpg", imgnum);
		ssimgleft = imread(filenameleft);
		ssimgright = imread(filenameright);
		if (ssimgleft.empty())
		{
			cout << "Open failed!" << endl;
			return -2;
		}
		if (ssimgright.empty())
		{
			cout << "Open failed!" << endl;
			return -2;
		}

		cout << "第" << imgnum << "幅图：" << endl;
		imgnum++;

		/*图像预处理：中值滤波+自适应阈值+开操作*/
		GaussianBlur(ssimgleft, srcimgleft, Size(5, 5), 0, 0);
		//medianBlur(ssimgleft, srcimgleft, 5);
		cvtColor(srcimgleft, srcimgleft, COLOR_BGR2GRAY);
		adaptiveThreshold(srcimgleft, srcimgleft, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 8);

		GaussianBlur(ssimgright, srcimgright, Size(5, 5), 0, 0);
		//medianBlur(ssimgright, srcimgright, 5);
		cvtColor(srcimgright, srcimgright, COLOR_BGR2GRAY);
		adaptiveThreshold(srcimgright, srcimgright, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 13, 8);

		Mat structmentL = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
		erode(srcimgleft, srcimgleft, structmentL, Point(-1, -1), 1);
		dilate(srcimgleft, srcimgleft, structmentL, Point(-1, -1), 4);
		namedWindow("11", 0);
		imshow("11", srcimgleft);

		Mat structmentR = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
		erode(srcimgright, srcimgright, structmentR, Point(-1, -1), 2);
		dilate(srcimgright, srcimgright, structmentR, Point(-1, -1), 4);
		namedWindow("22", 0);
		imshow("22", srcimgright);

		/*vector<Point2f>corner;
		double qualitylevel = 0.01;
		goodFeaturesToTrack(srcimgleft, corner, 20, qualitylevel, 10, noArray(), 3, false);
		TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 20, 0.1);
		cornerSubPix(srcimgleft, corner, Size(5, 5), Size(-1, -1), criteria);
		for (auto i = 0; i < corner.size(); i++)
		{
			circle(ssimgleft, corner[i], 2, Scalar(200, 127, 100), 2, 8, 0);
		}
		imshow("Output", ssimgleft);
		cvWaitKey(1000);*/

		/*霍夫圆
		第一个参数，InputArray类型的image，输入图像，即源图像，需为8位的灰度单通道图像。
		第二个参数，InputArray类型的circles，经过调用HoughCircles函数后此参数存储了检测到的圆的输出矢量，每个矢量由包含了3个元素的浮点矢量(x, y, radius)表示。
		第三个参数，int类型的method，即使用的检测方法，目前OpenCV中就霍夫梯度法一种可以使用，它的标识符为CV_HOUGH_GRADIENT，在此参数处填这个标识符即可。
		第四个参数，double类型的dp，用来检测圆心的累加器图像的分辨率于输入图像之比的倒数，且此参数允许创建一个比输入图像分辨率低的累加器。上述文字不好理解的话，来看例子吧。例如，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。
		第五个参数，double类型的minDist，为霍夫变换检测到的圆的圆心之间的最小距离，即让我们的算法能明显区分的两个不同圆之间的最小距离。这个参数如果太小的话，多个相邻的圆可能被错误地检测成了一个重合的圆。反之，这个参数设置太大的话，某些圆就不能被检测出来了。
		第六个参数，double类型的param1，有默认值100。它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。
		第七个参数，double类型的param2，也有默认值100。它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
		第八个参数，int类型的minRadius,有默认值0，表示圆半径的最小值。
		第九个参数，int类型的maxRadius,也有默认值0，表示圆半径的最大值。
		*/
		HoughCircles(srcimgleft, circles[0], CV_HOUGH_GRADIENT, 1, 20, 100, 15, 10, 15);
		HoughCircles(srcimgright, circles[1], CV_HOUGH_GRADIENT, 1, 18, 100, 15, 10, 15);
		ssimgleft.copyTo(dstimgleft);
		ssimgright.copyTo(dstimgright);
		Point2f center[2];//保存x,y值
		for (size_t i = 0; i < circles[0].size(); i++)
		{
			center[0] = Point2f(circles[0][i].x, circles[0][i].y);
			circle(dstimgleft, center[0], circles[0][i].z, Scalar(0, 255, 255), 2, LINE_AA);
			imgpoints[0].push_back(center[0]);
		}
		//namedWindow("left", 0);
		//imshow("left", dstimgleft);

		for (size_t i = 0; i < circles[1].size(); i++)
		{
			center[1] = Point2f(circles[1][i].x, circles[1][i].y);
			circle(dstimgright, center[1], circles[1][i].z, Scalar(255, 255, 0), 2, LINE_AA);
			imgpoints[1].push_back(center[1]);
		}
		//namedWindow("right", 0);
		//imshow("right", dstimgright);

		points[0].push_back(imgpoints[0]);
		points[1].push_back(imgpoints[1]);
		circles[0].clear();
		circles[1].clear();
		imgpoints[0].clear();
		imgpoints[1].clear();

		waitKey(1);
	}

	/*坐标显示*/
	//坐标排序
	Point2f temp;
	for (int i = 0; i < framenum - 1; i++)
	{
		for (int j = 0; j < (points[0][i].size() - 1); j++)
		{
			int minormax = j;
			for (int k = j + 1; k < points[0][i].size(); k++)
			{
				if (points[0][i][minormax].x > points[0][i][k].x)
				{
					minormax = k;
				}
			}
			if (j != minormax)
			{
				temp = points[0][i][minormax];
				points[0][i][minormax] = points[0][i][j];
				points[0][i][j] = temp;
			}
		}
	}
	//保存排序后的左坐标值
	int num = 1;
	FileStorage fs1("leftpoints.xml", FileStorage::WRITE);
	string str1 = "leftimage";
	for (int i = 0; i < framenum-1; i++)//图像序号循环
	{
		for (int j = 0; j < points[0][i].size(); j++)//每个图像的坐标序号循环
		{
			finalpoints[0].insert(make_pair(j + 1, points[0][i][j]));
		}
		str1 += to_string(num);
		fs1 << str1 << "[";
		for (multimap<int,Point2f>::iterator it=finalpoints[0].begin();it!=finalpoints[0].end();it++)
		{	
			fs1 << "{:" << "x" << it->second.x << "y" << it->second.y << "}";
		}
		fs1 << "]";
		finalpoints[0].clear();
		num++;
		str1 = "leftimage";
	}
	fs1.release();

	for (int i = 0; i < framenum - 1; i++)
	{
		for (int j = 0; j < (points[1][i].size() - 1); j++)
		{
			int minormax = j;
			for (int k = j + 1; k < points[1][i].size(); k++)
			{
				if (points[1][i][minormax].x > points[1][i][k].x)
				{
					minormax = k;
				}
			}
			if (j != minormax)
			{
				temp = points[1][i][minormax];
				points[1][i][minormax] = points[1][i][j];
				points[1][i][j] = temp;
			}
		}
	}

	//保存排序后的右坐标值
	num = 1;
	FileStorage fs2("rightpoints.xml", FileStorage::WRITE);
	string str2 = "rightimage";
	for (int i = 0; i < framenum - 1; i++)
	{
		for (int j = 0; j < points[1][i].size(); j++)
		{
			finalpoints[1].insert(make_pair(j + 1, points[1][i][j]));
		}
		
		str2 += to_string(num);
		fs2 << str2 << "[";
		for (multimap<int, Point2f>::iterator it = finalpoints[1].begin(); it != finalpoints[1].end(); it++)
		{
			fs2 << "{:" << "x" << it->second.x << "y" << it->second.y << "}";
		}
		fs2 << "]";
		finalpoints[1].clear();
		num++;
		str2 = "rightimage";
	}
	fs2.release();

	//读取文件记录，并显示排序好的坐标信息
	//Openfile openfile(framenum);
	//num = 1;
	//for (vector<vector<Point2f>>::iterator it = points[0].begin(); it != points[0].end(); it++)
	//{
	//	cout << "第" << num << "幅左图坐标值：" << endl;
	//	for (vector<Point2f>::iterator vit = (*it).begin(); vit != (*it).end(); vit++)
	//	{
	//		cout << "x= " << vit->x << " y=" << vit->y << endl;
	//	}
	//	num++;
	//}
	//num = 1;
	//for (vector<vector<Point2f>>::iterator it = points[1].begin(); it != points[1].end(); it++)
	//{
	//	cout << "第" << num << "幅右图坐标值：" << endl;
	//	for (vector<Point2f>::iterator vit = (*it).begin(); vit != (*it).end(); vit++)
	//	{
	//		cout << "x= " << vit->x << " y=" << vit->y << endl;
	//	}
	//	num++;
	//}

	////计算基本矩阵F
	//vector<Point2f>allpoints[2];
	//for (int i = 0; i < framenum; i++)
	//{
	//	//把左右相机所有图像上的点赋给allpoints
	//	copy(points[0][i].begin(), points[0][i].end(), back_inserter(allpoints[0]));
	//	copy(points[1][i].begin(), points[1][i].end(), back_inserter(allpoints[1]));
	//}
	//Mat F = findFundamentalMat(allpoints[0], allpoints[1], FM_8POINT);//单位为pixel;8点法

	Mat M1, M2, D1, D2, R, T;//单目标定得到的M1，M2内参矩阵, 畸变稀疏D1,D2
	M1 = (Mat_<double>(3, 3) << 2368.67668, - 0.32929, 1226.96770, 0.00000, 2367.29583, 996.79342, 0.00000, 0.00000, 1.00000);
	M2 = (Mat_<double>(3, 3) << 2348.70924, - 1.15290, 1205.79305, 0.00000, 2349.93521, 1003.43383, 0.00000, 0.00000, 1.00000);
	D1 = (Mat_<double>(1, 5) << -0.09899, 0.12739, - 0.00067, 0.00027, - 0.05622);
	D2 = (Mat_<double>(1, 5) << -0.09994, 0.13880, 0.00032, 0.00073, - 0.09704);
	R = (Mat_<double>(3, 3) << 0.99996, - 0.00700, - 0.00489, 0.00693, 0.99986, - 0.01500, 0.00499, 0.01496, 0.99988);
	T = (Mat_<double>(3, 1) << -91.27156, - 1.85131, - 2.00440);
	/*
	M:  fx a cx
		0 fy cy
		0  0  1
	D: k1 k2 p1 p2 k3
	*/
	
	//openfile.openfile();
	Mat map11, map12, map21, map22;
	Mat ssdstleft, ssdstright;//校正后的左右图像
	vector<Mat>finalQ;
	finalQ.clear();
	imgnum = 1;
	for (int i = 0; i < framenum - 1; i++) 
	{
		sprintf_s(filenameleft, "E:\\双目视觉\\T字形工件\\L_Z Total\\%d.jpg", imgnum);
		sprintf_s(filenameright, "E:\\双目视觉\\T字形工件\\R_Z Total\\%d.jpg", imgnum);
		ssimgleft = imread(filenameleft);
		ssimgright = imread(filenameright);
		if (ssimgleft.empty())
		{
			cout << "Open failed!" << endl;
			return -2;
		}
		if (ssimgright.empty())
		{
			cout << "Open failed!" << endl;
			return -2;
		}
		imgnum++;
		//畸变校正
		//稀疏点集畸变校正
		//undistortpoints(inputarray src, outputarray dst, inputarray cameramatrix, inputarray distcoeffs, inputarrayr=noarray(), inputarray p=noarray()//new cameramatrix)
		undistortPoints(points[0][i], points[0][i], M1, D1, Mat(), M1);
		undistortPoints(points[1][i], points[1][i], M2, D2, Mat(), M2);

		//立体校正
		Mat R1, R2, P1, P2, Q;//r1,r2为rrect;p1,p2为行对准后的单应性矩阵;q为从二维至三维的变换矩阵
		/*stereorectify(inputarray cameramatrix1, inputarray distcoeffs1,
		inputarray cameramatrix2, inputarray distcoeffs2, size imagesize,
			inputarray r, inputarray t, outputarray r1, outputarray r2, outputarray p1,
			outputarray p2, outputarray q, int flags = calib_zero_disparity, double alpha = -1,
			size newimagesize = size(), rect* validpixroi1 = 0, rect* validpixroi2 = 0)*/
		stereoRectify(M1, D1, M2, D2, ssimgleft.size(), R, T, R1, R2, P1, P2, Q, 0);
		finalQ.push_back(Q);
		/*
		从源图像(u,v)->校正后图像(u',v')
		step1:(x1,y1,z1)=M1-1*z1*(u,v,1)相机坐标系（源图像）
		step2:(x1',y1',z1')=Rl*(x1,y1,z1)(相机坐标系)(目标图像)
		step3:z1'*(u',v',1)=Mrect*(x1',y1',z1')(图像坐标系)(目标图像)*/

		//Mat pixeldst1, pixeldst2;//(u',v',1)
		//vector<Point2f>temprectifiedpoints[2];
		//for (int j = 0; j < points[0][i].size(); j++)
		//{
		//	Mat pixelsrc1 = (Mat_<double>(3, 1) << points[0][i][j].x, points[0][i][j].y, 1);//points[0][i][j]为畸变矫正后的像素点
		//	pixeldst1 = R1 * pixelsrc1;//step3
		//	pixeldst1.at<double>(0, 0) = pixeldst1.at<double>(0, 0) / pixeldst1.at<double>(2, 0);
		//	pixeldst1.at<double>(1, 0) = pixeldst1.at<double>(1, 0) / pixeldst1.at<double>(2, 0);
		//	temprectifiedpoints[0].push_back(Point2f(pixeldst1.at<double>(0, 0), pixeldst1.at<double>(1, 0)));
		//}
		//for (int j = 0; j < points[1][i].size(); j++)
		//{
		//	Mat pixelsrc2 = (Mat_<double>(3, 1) << points[1][i][j].x, points[1][i][j].y, 1);//points[0][i][j]为畸变矫正后的像素点
		//	pixeldst2 = R2 * pixelsrc2;//step3
		//	pixeldst2.at<double>(0, 0) = pixeldst2.at<double>(0, 0) / pixeldst2.at<double>(2, 0);
		//	pixeldst2.at<double>(1, 0) = pixeldst2.at<double>(1, 0) / pixeldst2.at<double>(2, 0);
		//	temprectifiedpoints[1].push_back(Point2f(pixeldst2.at<double>(0, 0), pixeldst2.at<double>(1, 0)));
		//}

		/*计算三维坐标*/
		//校正映射
		/*initUndistortRectifyMap(InputArray cameraMatrix, InputArray distCoeffs,
		InputArray R, InputArray newCameraMatrix,
		Size size, int m1type, OutputArray map1, OutputArray map2)*/
		initUndistortRectifyMap(M1, D1, R1, P1, ssimgleft.size(), CV_32FC1, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, ssimgright.size(), CV_32FC1, map21, map22);

		/*void remap(InputArray src, OutputArray dst, InputArray map1, InputArray map2,
			int interpolation, intborderMode = BORDER_CONSTANT,
			const Scalar& borderValue = Scalar())
			第一个参数：输入图像，即原图像，需要单通道8位或者浮点类型的图像
			第二个参数：输出图像，即目标图像，需和原图形一样的尺寸和类型
			第三个参数：它有两种可能表示的对象：（1）表示点（x, y）的第一个映射；（2）表示CV_16SC2，CV_32FC1等
			第四个参数：它有两种可能表示的对象：（1）若map1表示点（x, y）时，这个参数不代表任何值；（2）表示  CV_16UC1，CV_32FC1类型的Y值
			第五个参数：插值方式，有四中插值方式：（1）INTER_NEAREST——最近邻插值
			（2）INTER_LINEAR——双线性插值（默认）
			（3）INTER_CUBIC——双三样条插值（默认）
			（4）INTER_LANCZOS4——lanczos插值（默认）
			第六个参数：边界模式，默认BORDER_CONSTANT
			第七个参数：边界颜色，默认Scalar()黑色*/
		remap(ssimgleft, ssdstleft, map11, map12, cv::INTER_LINEAR);
		remap(ssimgright, ssdstright, map21, map22, cv::INTER_LINEAR);
		
		/*重新检测校正后的像素特征点*/
		Mat proleft, proright;//校正后图像预处理矩阵
		GaussianBlur(ssdstleft, proleft, Size(5, 5), 0, 0);
		cvtColor(proleft, proleft, COLOR_BGR2GRAY);
		adaptiveThreshold(proleft, proleft, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 7);

		GaussianBlur(ssdstright, proright, Size(5, 5), 0, 0);
		cvtColor(proright, proright, COLOR_BGR2GRAY);
		adaptiveThreshold(proright, proright, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 7);

		Mat structmentL = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
		erode(proleft, proleft, structmentL, Point(-1, -1), 1);
		dilate(proleft, proleft, structmentL, Point(-1, -1), 1);

		Mat structmentR = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
		erode(proright, proright, structmentR, Point(-1, -1), 1);
		dilate(proright, proright, structmentR, Point(-1, -1), 3);
		namedWindow("rectifiedright", 0);
		imshow("rectifiedright", proright);

		/*霍夫圆*/
		HoughCircles(proleft, circles[0], CV_HOUGH_GRADIENT, 1, 18, 100, 15, 10, 15);
		HoughCircles(proright, circles[1], CV_HOUGH_GRADIENT, 1, 18, 100, 15, 10, 15);
		ssdstleft.copyTo(rectfieddst1);
		ssdstright.copyTo(rectifieddst2);
		Point2f center[2];
		for (size_t i = 0; i < circles[0].size(); i++)
		{
			center[0] = Point2f(circles[0][i].x, circles[0][i].y);
			circle(rectfieddst1, center[0], circles[0][i].z, Scalar(0, 255, 255), 2, LINE_AA);
			imgpoints[0].push_back(center[0]);
		}
		namedWindow("left", 0);
		imshow("left", rectfieddst1);

		for (size_t i = 0; i < circles[1].size(); i++)
		{
			center[1] = Point2f(circles[1][i].x, circles[1][i].y);
			circle(rectifieddst2, center[1], circles[1][i].z, Scalar(255, 255, 0), 2, LINE_AA);
			imgpoints[1].push_back(center[1]);
		}
		//cvtColor(dstimgright, dstimgright, COLOR_BGR2GRAY);
		namedWindow("right", 0);
		imshow("right", rectifieddst2);
		//comparerectifiedpoints[0].push_back(temprectifiedpoints[0]);//方法一校正图像图像坐标存取
		//comparerectifiedpoints[1].push_back(temprectifiedpoints[1]);
		rectifiedpoints[0].push_back(imgpoints[0]);//方法二校正图像图像坐标存取
		rectifiedpoints[1].push_back(imgpoints[1]);

		//temprectifiedpoints[0].clear();
		//temprectifiedpoints[1].clear();
		circles[0].clear();
		circles[1].clear();
		imgpoints[0].clear();
		imgpoints[1].clear();

		waitKey(1);
		/*namedWindow("dst1", 0);
		imshow("dst1", rectfieddst1);
		namedWindow("dst2", 0);
		imshow("dst2", rectifieddst2);
		waitKey(10);*/
	}
	/*坐标显示*/
	//坐标排序
	/*for (int i = 0; i < framenum - 1; i++)
	{
		for (int j = 0; j < (comparerectifiedpoints[0][i].size() - 1); j++)
		{
			int minormax = j;
			for (int k = j + 1; k < comparerectifiedpoints[0][i].size(); k++)
			{
				if (comparerectifiedpoints[0][i][minormax].y > comparerectifiedpoints[0][i][k].y)
				{
					minormax = k;
				}
			}
			if (j != minormax)
			{
				temp = comparerectifiedpoints[0][i][minormax];
				comparerectifiedpoints[0][i][minormax] = comparerectifiedpoints[0][i][j];
				comparerectifiedpoints[0][i][j] = temp;
			}
		}
	}*/
	for (int i = 0; i < framenum - 1; i++)
	{
		for (int j = 0; j < (rectifiedpoints[0][i].size() - 1); j++)
		{
			int minormax = j;
			for (int k = j + 1; k < rectifiedpoints[0][i].size(); k++)
			{
				if (rectifiedpoints[0][i][minormax].x > rectifiedpoints[0][i][k].x)//找最小值
				{
					minormax = k;
				}
			}
			if (j != minormax)
			{
				temp = rectifiedpoints[0][i][minormax];
				rectifiedpoints[0][i][minormax] = rectifiedpoints[0][i][j];
				rectifiedpoints[0][i][j] = temp;
			}
		}
	}

	//保存排序后的左坐标值
	num = 1;
	FileStorage fs3("rectifiedleftpoints.xml", FileStorage::WRITE);
	string str3 = "rectifiedleftimage";
	for (int i = 0; i < framenum - 1; i++)
	{
		for (int j = 0; j < rectifiedpoints[0][i].size(); j++)
		{
			rectifiedfinalpoints[0].insert(make_pair(j + 1, rectifiedpoints[0][i][j]));
		}
		str3 += to_string(num);
		fs3 << str3 << "[";
		for (multimap<int, Point2f>::iterator it = rectifiedfinalpoints[0].begin(); it != rectifiedfinalpoints[0].end(); it++)
		{
			fs3 << "{:" << "x" << it->second.x << "y" << it->second.y << "}";
		}
		fs3 << "]";
		rectifiedfinalpoints[0].clear();
		num++;
		str3 = "rectifiedleftimage";
	}
	fs3.release();

	for (int i = 0; i < framenum - 1; i++)
	{
		for (int j = 0; j < (rectifiedpoints[1][i].size() - 1); j++)
		{
			int minormax = j;
			for (int k = j + 1; k < rectifiedpoints[1][i].size(); k++)
			{
				if (rectifiedpoints[1][i][minormax].x > rectifiedpoints[1][i][k].x)
				{
					minormax = k;
				}
			}
			if (j != minormax)
			{
				temp = rectifiedpoints[1][i][minormax];
				rectifiedpoints[1][i][minormax] = rectifiedpoints[1][i][j];
				rectifiedpoints[1][i][j] = temp;
			}
		}
	}

	//保存排序后的右坐标值
	num = 1;
	FileStorage fs4("rectifiedrightpoints.xml", FileStorage::WRITE);
	string str4 = "rectifiedrightimage";
	for (int i = 0; i < framenum - 1; i++)
	{
		for (int j = 0; j < rectifiedpoints[1][i].size(); j++)
		{
			rectifiedfinalpoints[1].insert(make_pair(j + 1, rectifiedpoints[1][i][j]));
		}
		str4 += to_string(num);
		fs4 << str4 << "[";
		for (multimap<int, Point2f>::iterator it = rectifiedfinalpoints[1].begin(); it != rectifiedfinalpoints[1].end(); it++)
		{
			fs4 << "{:" << "x" << it->second.x << "y" << it->second.y << "}";
		}
		fs4 << "]";
		rectifiedfinalpoints[1].clear();
		num++;
		str4 = "rectifiedrightimage";
	}
	fs4.release();
	Openfile openfile;
	openfile.openrectified(framenum);

	/*提取视差*/
	float d;//视差
	map<int,Point3d>temp3Dpoints;
	map<int, map<int, Point3d>>final3Dpoints;//第一个int——图像序号，map<int,float>中的第一个int——坐标序号,第二个float为相应点间的视差
	Mat uv;

	int tempnum = 0;
	for (int i = 0; i < framenum - 1; i++)
	{
		cout << "第" << i+1 << "幅校正左图片视差开始提取" << endl;
		//判断左右图片特定点数量是否相等
		if (openfile.rectifiedallpoints[0][i].size() == openfile.rectifiedallpoints[1][i].size())
		{
			for (int j = 0; j < openfile.rectifiedallpoints[0][i].size(); j++)
			{
				d = openfile.rectifiedallpoints[0][i][j].x - openfile.rectifiedallpoints[1][i][j].x;//d=xL-xR
				uv = (Mat_<double>(4, 1) << openfile.rectifiedallpoints[0][i][j].x, openfile.rectifiedallpoints[0][i][j].y, d, 1);
				//Q*[x,y,d,1]=[X,Y,Z,W]
				Mat Q = finalQ[i];
				Mat objectpoints = Q * uv;
				//cout << objectpoints << endl;
				//[X/W,Y/W,Z/W]
				objectpoints.at<double>(0, 0) = objectpoints.at<double>(0, 0) / objectpoints.at<double>(3, 0);
				objectpoints.at<double>(1, 0) = objectpoints.at<double>(1, 0) / objectpoints.at<double>(3, 0);
				objectpoints.at<double>(2, 0) = objectpoints.at<double>(2, 0) / objectpoints.at<double>(3, 0);
				Point3d tempobjectpoints = Point3d(objectpoints.at<double>(0, 0), objectpoints.at<double>(1, 0), objectpoints.at<double>(2, 0));
				temp3Dpoints.insert(make_pair(j, tempobjectpoints));
			}
			final3Dpoints.insert(make_pair(tempnum, temp3Dpoints));
			if (tempnum > 0)
			{
				for (int j = 0; j < openfile.rectifiedallpoints[0][i].size(); j++)
				{
					cout << "x方向：" << setprecision(4) << final3Dpoints[tempnum][j].x - final3Dpoints[tempnum - 1][j].x;
					cout << "\ty方向：" << setprecision(4) << final3Dpoints[tempnum][j].y - final3Dpoints[tempnum - 1][j].y;
					cout << "\tz方向：" << setprecision(4) << final3Dpoints[tempnum][j].z - final3Dpoints[tempnum - 1][j].z << endl;
				}
			}
			tempnum++;
			//更新数据
			temp3Dpoints.clear();
		}
		else
		{
			cout << "第" << i + 1 << "左右图像特征点不匹配！" << endl;
			//break;
		}
	}

	//三维坐标显示
	//三维坐标保存
	num = 1;
	FileStorage fs5("objectpoints.xml", FileStorage::WRITE);
	string str5 = "objectpoints";
	for (int i = 0; i < framenum - 1; i++)
	{
		str5 += to_string(num);
		fs5 << str5 << "[";
		for (int j = 0; j < final3Dpoints[i].size(); j++)
		{
			fs5 << "{:" << "index" << j + 1 << "x" << final3Dpoints[i][j].x << "y" << final3Dpoints[i][j].y << "z" << final3Dpoints[i][j].z << "}";
		}
		fs5 << "]";
		num++;
		str5 = "objectpoints";
	}
	fs5.release();
	///*三维坐标点的计算*/
	//vector<Point3d> imgpoint;
	//vector<Point3d> objectpoints;
	//Mat Q;
	//for (int i = 0; i < framenum - 1; i++)
	//{
	//	if (openfile.rectifiedallpoints[0][i].size() == openfile.rectifiedallpoints[1][i].size())
	//	{
	//		for (int j = 0; j < openfile.rectifiedallpoints[0][i].size(); j++)
	//		{
	//			imgpoint.push_back(Point3d(openfile.rectifiedallpoints[0][i][j].x, openfile.rectifiedallpoints[0][i][j].y, finalD[i][j]));
	//		}
	//		Q = finalQ[i];
	//		perspectiveTransform(imgpoint, objectpoints, Q);	
	//	}
	//	else
	//	{
	//		cout << "左右图像对应坐标点个数不匹配！" << endl;
	//		break;
	//	}
	//}
	//openfile.open3D(framenum);
	
	return 0;
}
