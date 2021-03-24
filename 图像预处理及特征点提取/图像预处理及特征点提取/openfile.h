#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <map>
#include <algorithm>
#include <functional>
using namespace std;
using namespace cv;

class Openfile
{
public:
	Openfile();

	Openfile(int framenum);//读取校正前坐标点信息

	void openrectified(int framenum);//读取校正后坐标点信息

	void open3D(int framenum);//读取三维坐标点信息
	//void openfile();

	int imgnum;
	int pointnum;
	map<int, map<int,Point2f>>allpoints[2];//校正前
	map<int, map<int, Point2f>>rectifiedallpoints[2];//校正后：第一个int——图像序号，map<int,Point2f>中的int——坐标排序，Point2f相应各点
	map<int, map<int, Point3d>>finalobjectpoints;//读取的三维坐标
	Mat M1, M2, R, D1, D2, T;
};