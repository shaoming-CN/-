#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <functional>
#include <string>
#include "openfile.h"
using namespace std;
using namespace cv;

//ͼƬ֡��
int framenum = 25;
int main()
{
	vector<vector<Point2f>>points[2];//����ͼ��ƽ�������ֵ
	vector<vector<Point2f>>comparerectifiedpoints[2];//����һУ��������ͼ��ƽ�������ֵ
	vector<vector<Point2f>>rectifiedpoints[2];//������У��������ͼ��ƽ�������ֵ
	vector<Point2f>imgpoints[2];//ÿ��ͼ��ƽ�����������ֵ
	vector<Point3f>circles[2];//����Բ����
	multimap<int, Point2f>finalpoints[2];//������������������ֵ
	multimap<int, Point2f>rectifiedfinalpoints[2];//У�����������������ֵ
	int imgnum = 1;
	char filenameleft[1024];
	char filenameright[1024];
	Mat ssimgleft, ssimgright;//У��ǰ����ͼ��
	Mat srcimgleft, srcimgright;
	Mat dstimgleft, dstimgright;
	Mat rectfieddst1, rectifieddst2;

	while (imgnum < framenum)
	{
		sprintf_s(filenameleft, "E:\\˫Ŀ�Ӿ�\\T���ι���\\L_Z Total\\%d.jpg", imgnum);
		sprintf_s(filenameright, "E:\\˫Ŀ�Ӿ�\\T���ι���\\R_Z Total\\%d.jpg", imgnum);
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

		cout << "��" << imgnum << "��ͼ��" << endl;
		imgnum++;

		/*ͼ��Ԥ������ֵ�˲�+����Ӧ��ֵ+������*/
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

		/*����Բ
		��һ��������InputArray���͵�image������ͼ�񣬼�Դͼ����Ϊ8λ�ĻҶȵ�ͨ��ͼ��
		�ڶ���������InputArray���͵�circles����������HoughCircles������˲����洢�˼�⵽��Բ�����ʸ����ÿ��ʸ���ɰ�����3��Ԫ�صĸ���ʸ��(x, y, radius)��ʾ��
		������������int���͵�method����ʹ�õļ�ⷽ����ĿǰOpenCV�оͻ����ݶȷ�һ�ֿ���ʹ�ã����ı�ʶ��ΪCV_HOUGH_GRADIENT���ڴ˲������������ʶ�����ɡ�
		���ĸ�������double���͵�dp���������Բ�ĵ��ۼ���ͼ��ķֱ���������ͼ��֮�ȵĵ������Ҵ˲���������һ��������ͼ��ֱ��ʵ͵��ۼ������������ֲ������Ļ����������Ӱɡ����磬���dp= 1ʱ���ۼ���������ͼ�������ͬ�ķֱ��ʡ����dp=2���ۼ�����������ͼ��һ����ô��Ŀ�Ⱥ͸߶ȡ�
		�����������double���͵�minDist��Ϊ����任��⵽��Բ��Բ��֮�����С���룬�������ǵ��㷨���������ֵ�������ͬԲ֮�����С���롣����������̫С�Ļ���������ڵ�Բ���ܱ�����ؼ�����һ���غϵ�Բ����֮�������������̫��Ļ���ĳЩԲ�Ͳ��ܱ��������ˡ�
		������������double���͵�param1����Ĭ��ֵ100�����ǵ���������method���õļ�ⷽ���Ķ�Ӧ�Ĳ������Ե�ǰΨһ�ķ��������ݶȷ�CV_HOUGH_GRADIENT������ʾ���ݸ�canny��Ե������ӵĸ���ֵ��������ֵΪ����ֵ��һ�롣
		���߸�������double���͵�param2��Ҳ��Ĭ��ֵ100�����ǵ���������method���õļ�ⷽ���Ķ�Ӧ�Ĳ������Ե�ǰΨһ�ķ��������ݶȷ�CV_HOUGH_GRADIENT������ʾ�ڼ��׶�Բ�ĵ��ۼ�����ֵ����ԽС�Ļ����Ϳ��Լ�⵽������������ڵ�Բ������Խ��Ļ�����ͨ������Բ�͸��ӽӽ�������Բ���ˡ�
		�ڰ˸�������int���͵�minRadius,��Ĭ��ֵ0����ʾԲ�뾶����Сֵ��
		�ھŸ�������int���͵�maxRadius,Ҳ��Ĭ��ֵ0����ʾԲ�뾶�����ֵ��
		*/
		HoughCircles(srcimgleft, circles[0], CV_HOUGH_GRADIENT, 1, 20, 100, 15, 10, 15);
		HoughCircles(srcimgright, circles[1], CV_HOUGH_GRADIENT, 1, 18, 100, 15, 10, 15);
		ssimgleft.copyTo(dstimgleft);
		ssimgright.copyTo(dstimgright);
		Point2f center[2];//����x,yֵ
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

	/*������ʾ*/
	//��������
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
	//����������������ֵ
	int num = 1;
	FileStorage fs1("leftpoints.xml", FileStorage::WRITE);
	string str1 = "leftimage";
	for (int i = 0; i < framenum-1; i++)//ͼ�����ѭ��
	{
		for (int j = 0; j < points[0][i].size(); j++)//ÿ��ͼ����������ѭ��
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

	//����������������ֵ
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

	//��ȡ�ļ���¼������ʾ����õ�������Ϣ
	//Openfile openfile(framenum);
	//num = 1;
	//for (vector<vector<Point2f>>::iterator it = points[0].begin(); it != points[0].end(); it++)
	//{
	//	cout << "��" << num << "����ͼ����ֵ��" << endl;
	//	for (vector<Point2f>::iterator vit = (*it).begin(); vit != (*it).end(); vit++)
	//	{
	//		cout << "x= " << vit->x << " y=" << vit->y << endl;
	//	}
	//	num++;
	//}
	//num = 1;
	//for (vector<vector<Point2f>>::iterator it = points[1].begin(); it != points[1].end(); it++)
	//{
	//	cout << "��" << num << "����ͼ����ֵ��" << endl;
	//	for (vector<Point2f>::iterator vit = (*it).begin(); vit != (*it).end(); vit++)
	//	{
	//		cout << "x= " << vit->x << " y=" << vit->y << endl;
	//	}
	//	num++;
	//}

	////�����������F
	//vector<Point2f>allpoints[2];
	//for (int i = 0; i < framenum; i++)
	//{
	//	//�������������ͼ���ϵĵ㸳��allpoints
	//	copy(points[0][i].begin(), points[0][i].end(), back_inserter(allpoints[0]));
	//	copy(points[1][i].begin(), points[1][i].end(), back_inserter(allpoints[1]));
	//}
	//Mat F = findFundamentalMat(allpoints[0], allpoints[1], FM_8POINT);//��λΪpixel;8�㷨

	Mat M1, M2, D1, D2, R, T;//��Ŀ�궨�õ���M1��M2�ڲξ���, ����ϡ��D1,D2
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
	Mat ssdstleft, ssdstright;//У���������ͼ��
	vector<Mat>finalQ;
	finalQ.clear();
	imgnum = 1;
	for (int i = 0; i < framenum - 1; i++) 
	{
		sprintf_s(filenameleft, "E:\\˫Ŀ�Ӿ�\\T���ι���\\L_Z Total\\%d.jpg", imgnum);
		sprintf_s(filenameright, "E:\\˫Ŀ�Ӿ�\\T���ι���\\R_Z Total\\%d.jpg", imgnum);
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
		//����У��
		//ϡ��㼯����У��
		//undistortpoints(inputarray src, outputarray dst, inputarray cameramatrix, inputarray distcoeffs, inputarrayr=noarray(), inputarray p=noarray()//new cameramatrix)
		undistortPoints(points[0][i], points[0][i], M1, D1, Mat(), M1);
		undistortPoints(points[1][i], points[1][i], M2, D2, Mat(), M2);

		//����У��
		Mat R1, R2, P1, P2, Q;//r1,r2Ϊrrect;p1,p2Ϊ�ж�׼��ĵ�Ӧ�Ծ���;qΪ�Ӷ�ά����ά�ı任����
		/*stereorectify(inputarray cameramatrix1, inputarray distcoeffs1,
		inputarray cameramatrix2, inputarray distcoeffs2, size imagesize,
			inputarray r, inputarray t, outputarray r1, outputarray r2, outputarray p1,
			outputarray p2, outputarray q, int flags = calib_zero_disparity, double alpha = -1,
			size newimagesize = size(), rect* validpixroi1 = 0, rect* validpixroi2 = 0)*/
		stereoRectify(M1, D1, M2, D2, ssimgleft.size(), R, T, R1, R2, P1, P2, Q, 0);
		finalQ.push_back(Q);
		/*
		��Դͼ��(u,v)->У����ͼ��(u',v')
		step1:(x1,y1,z1)=M1-1*z1*(u,v,1)�������ϵ��Դͼ��
		step2:(x1',y1',z1')=Rl*(x1,y1,z1)(�������ϵ)(Ŀ��ͼ��)
		step3:z1'*(u',v',1)=Mrect*(x1',y1',z1')(ͼ������ϵ)(Ŀ��ͼ��)*/

		//Mat pixeldst1, pixeldst2;//(u',v',1)
		//vector<Point2f>temprectifiedpoints[2];
		//for (int j = 0; j < points[0][i].size(); j++)
		//{
		//	Mat pixelsrc1 = (Mat_<double>(3, 1) << points[0][i][j].x, points[0][i][j].y, 1);//points[0][i][j]Ϊ�������������ص�
		//	pixeldst1 = R1 * pixelsrc1;//step3
		//	pixeldst1.at<double>(0, 0) = pixeldst1.at<double>(0, 0) / pixeldst1.at<double>(2, 0);
		//	pixeldst1.at<double>(1, 0) = pixeldst1.at<double>(1, 0) / pixeldst1.at<double>(2, 0);
		//	temprectifiedpoints[0].push_back(Point2f(pixeldst1.at<double>(0, 0), pixeldst1.at<double>(1, 0)));
		//}
		//for (int j = 0; j < points[1][i].size(); j++)
		//{
		//	Mat pixelsrc2 = (Mat_<double>(3, 1) << points[1][i][j].x, points[1][i][j].y, 1);//points[0][i][j]Ϊ�������������ص�
		//	pixeldst2 = R2 * pixelsrc2;//step3
		//	pixeldst2.at<double>(0, 0) = pixeldst2.at<double>(0, 0) / pixeldst2.at<double>(2, 0);
		//	pixeldst2.at<double>(1, 0) = pixeldst2.at<double>(1, 0) / pixeldst2.at<double>(2, 0);
		//	temprectifiedpoints[1].push_back(Point2f(pixeldst2.at<double>(0, 0), pixeldst2.at<double>(1, 0)));
		//}

		/*������ά����*/
		//У��ӳ��
		/*initUndistortRectifyMap(InputArray cameraMatrix, InputArray distCoeffs,
		InputArray R, InputArray newCameraMatrix,
		Size size, int m1type, OutputArray map1, OutputArray map2)*/
		initUndistortRectifyMap(M1, D1, R1, P1, ssimgleft.size(), CV_32FC1, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, ssimgright.size(), CV_32FC1, map21, map22);

		/*void remap(InputArray src, OutputArray dst, InputArray map1, InputArray map2,
			int interpolation, intborderMode = BORDER_CONSTANT,
			const Scalar& borderValue = Scalar())
			��һ������������ͼ�񣬼�ԭͼ����Ҫ��ͨ��8λ���߸������͵�ͼ��
			�ڶ������������ͼ�񣬼�Ŀ��ͼ�����ԭͼ��һ���ĳߴ������
			�������������������ֿ��ܱ�ʾ�Ķ��󣺣�1����ʾ�㣨x, y���ĵ�һ��ӳ�䣻��2����ʾCV_16SC2��CV_32FC1��
			���ĸ��������������ֿ��ܱ�ʾ�Ķ��󣺣�1����map1��ʾ�㣨x, y��ʱ����������������κ�ֵ����2����ʾ  CV_16UC1��CV_32FC1���͵�Yֵ
			�������������ֵ��ʽ�������в�ֵ��ʽ����1��INTER_NEAREST��������ڲ�ֵ
			��2��INTER_LINEAR����˫���Բ�ֵ��Ĭ�ϣ�
			��3��INTER_CUBIC����˫��������ֵ��Ĭ�ϣ�
			��4��INTER_LANCZOS4����lanczos��ֵ��Ĭ�ϣ�
			�������������߽�ģʽ��Ĭ��BORDER_CONSTANT
			���߸��������߽���ɫ��Ĭ��Scalar()��ɫ*/
		remap(ssimgleft, ssdstleft, map11, map12, cv::INTER_LINEAR);
		remap(ssimgright, ssdstright, map21, map22, cv::INTER_LINEAR);
		
		/*���¼��У���������������*/
		Mat proleft, proright;//У����ͼ��Ԥ�������
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

		/*����Բ*/
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
		//comparerectifiedpoints[0].push_back(temprectifiedpoints[0]);//����һУ��ͼ��ͼ�������ȡ
		//comparerectifiedpoints[1].push_back(temprectifiedpoints[1]);
		rectifiedpoints[0].push_back(imgpoints[0]);//������У��ͼ��ͼ�������ȡ
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
	/*������ʾ*/
	//��������
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
				if (rectifiedpoints[0][i][minormax].x > rectifiedpoints[0][i][k].x)//����Сֵ
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

	//����������������ֵ
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

	//����������������ֵ
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

	/*��ȡ�Ӳ�*/
	float d;//�Ӳ�
	map<int,Point3d>temp3Dpoints;
	map<int, map<int, Point3d>>final3Dpoints;//��һ��int����ͼ����ţ�map<int,float>�еĵ�һ��int�����������,�ڶ���floatΪ��Ӧ�����Ӳ�
	Mat uv;

	int tempnum = 0;
	for (int i = 0; i < framenum - 1; i++)
	{
		cout << "��" << i+1 << "��У����ͼƬ�Ӳʼ��ȡ" << endl;
		//�ж�����ͼƬ�ض��������Ƿ����
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
					cout << "x����" << setprecision(4) << final3Dpoints[tempnum][j].x - final3Dpoints[tempnum - 1][j].x;
					cout << "\ty����" << setprecision(4) << final3Dpoints[tempnum][j].y - final3Dpoints[tempnum - 1][j].y;
					cout << "\tz����" << setprecision(4) << final3Dpoints[tempnum][j].z - final3Dpoints[tempnum - 1][j].z << endl;
				}
			}
			tempnum++;
			//��������
			temp3Dpoints.clear();
		}
		else
		{
			cout << "��" << i + 1 << "����ͼ�������㲻ƥ�䣡" << endl;
			//break;
		}
	}

	//��ά������ʾ
	//��ά���걣��
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
	///*��ά�����ļ���*/
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
	//		cout << "����ͼ���Ӧ����������ƥ�䣡" << endl;
	//		break;
	//	}
	//}
	//openfile.open3D(framenum);
	
	return 0;
}
