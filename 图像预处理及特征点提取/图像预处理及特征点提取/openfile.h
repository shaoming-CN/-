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

	Openfile(int framenum);//��ȡУ��ǰ�������Ϣ

	void openrectified(int framenum);//��ȡУ�����������Ϣ

	void open3D(int framenum);//��ȡ��ά�������Ϣ
	//void openfile();

	int imgnum;
	int pointnum;
	map<int, map<int,Point2f>>allpoints[2];//У��ǰ
	map<int, map<int, Point2f>>rectifiedallpoints[2];//У���󣺵�һ��int����ͼ����ţ�map<int,Point2f>�е�int������������Point2f��Ӧ����
	map<int, map<int, Point3d>>finalobjectpoints;//��ȡ����ά����
	Mat M1, M2, R, D1, D2, T;
};