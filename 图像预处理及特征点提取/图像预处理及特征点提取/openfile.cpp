#include "openfile.h"

Openfile::Openfile()
{

}
Openfile::Openfile(int framenum)
{
	this->imgnum = 0;
	this->pointnum = 0;
	this->allpoints[0].clear();
	this->allpoints[1].clear();
	this->rectifiedallpoints[0].clear();
	this->rectifiedallpoints[1].clear();
	this->finalobjectpoints.clear();

	map<int, Point2f>temppoints;
	FileStorage fs1("leftpoints.xml", FileStorage::READ);
	string str1 = "leftimage";
	int index = 0;//坐标序号
	for (int i = 0; i < framenum-1; i++) 
	{
		str1 += to_string(i+1);
		FileNode leftimage = fs1[str1];
		cout << "leftimage#" << i+1 << ":" << endl;
		for (FileNodeIterator it = leftimage.begin(); it != leftimage.end(); it++, index++)
		{	
			cout << "坐标序号：" << index+1 << ",x=" << (float)(*it)["x"] << ",y=" << (float)(*it)["y"] << endl;
			temppoints.insert(make_pair(index, Point2f((float)(*it)["x"], (float)(*it)["y"])));
		}
		this->allpoints[0].insert(make_pair(i, temppoints));
		//更新数据
		temppoints.clear();
		str1 = "leftimage";
		index = 1;
	}
	fs1.release();

	index = 0;
	FileStorage fs2("rightpoints.xml", FileStorage::READ);
	string str2 = "rightimage";
	for (int i = 0; i < framenum-1; i++)
	{
		str2 += to_string(i+1);
		FileNode rightimage = fs2[str2];
		cout << "rightimage#" << i+1 << ":" << endl;
		for (FileNodeIterator it = rightimage.begin(); it != rightimage.end(); it++, index++)
		{
			cout << "坐标序号：" << index+1 << ",x=" << (float)(*it)["x"] << ",y=" << (float)(*it)["y"] << endl;
			temppoints.insert(make_pair(index, Point2f((float)(*it)["x"], (float)(*it)["y"])));
		}
		//更新数据
		this->allpoints[1].insert(make_pair(i, temppoints));
		temppoints.clear();
		str2 = "rightimage";
		index = 0;
	}
	fs2.release();
}
void Openfile::openrectified(int framenum)
{
	this->rectifiedallpoints[0].clear();
	this->rectifiedallpoints[1].clear();
	/*校正后坐标像素点提取*/
	map<int, Point2f>temppoints;
	int index = 0;
	FileStorage fs3("rectifiedleftpoints.xml", FileStorage::READ);
	string str3 = "rectifiedleftimage";
	for (int i = 0; i < framenum - 1; i++)
	{
		str3 += to_string(i + 1);
		FileNode rectifiedleftimage = fs3[str3];
		cout << "rectifiedleftimage#" << i + 1 << ":" << endl;
		for (FileNodeIterator it = rectifiedleftimage.begin(); it != rectifiedleftimage.end(); it++, index++)
		{
			cout << "坐标序号：" << index + 1 << ",x=" << (float)(*it)["x"] << ",y=" << (float)(*it)["y"] << endl;
			temppoints.insert(make_pair(index, Point2f((float)(*it)["x"], (float)(*it)["y"])));
		}
		//更新数据
		this->rectifiedallpoints[0].insert(make_pair(i, temppoints));
		str3 = "rectifiedleftimage";
		temppoints.clear();
		index = 0;
	}
	fs3.release();

	index = 0;
	FileStorage fs4("rectifiedrightpoints.xml", FileStorage::READ);
	string str4 = "rectifiedrightimage";
	for (int i = 0; i < framenum - 1; i++)
	{
		str4 += to_string(i + 1);
		FileNode rectifiedrightimage = fs4[str4];
		cout << "rectifiedrightimage#" << i + 1 << ":" << endl;
		for (FileNodeIterator it = rectifiedrightimage.begin(); it != rectifiedrightimage.end(); it++, index++)
		{
			cout << "坐标序号：" << index + 1 << ",x=" << (float)(*it)["x"] << ",y=" << (float)(*it)["y"] << endl;
			temppoints.insert(make_pair(index, Point2f((float)(*it)["x"], (float)(*it)["y"])));
		}
		//更新数据
		this->rectifiedallpoints[1].insert(make_pair(i, temppoints));
		temppoints.clear();
		str4 = "rectifiedrightimage";
		index = 0;
	}
	fs4.release();
}
void Openfile::open3D(int framenum)
{
	this->finalobjectpoints.clear();
	//三维坐标
	int index = 0;
	map<int, Point3d>tempobjectpoints;
	FileStorage fs5("objectpoints.xml", FileStorage::READ);
	string str5 = "objectpoints";
	for (int i = 0; i < framenum-1; i++)
	{
		str5 += to_string(i+1);
		FileNode objectpoints = fs5[str5];
		cout << "image#" << i+1 << ":" << endl;
		for (FileNodeIterator it = objectpoints.begin(); it != objectpoints.end(); it++, index++)
		{
			cout << "坐标序号：" << index+1 << ",x=" << (float)(*it)["x"] << ",y=" << (float)(*it)["y"] << ",z=" << (float)(*it)["z"] << endl;
			tempobjectpoints.insert(make_pair(index, Point3d((float)(*it)["x"], (float)(*it)["y"], (float)(*it)["z"])));
		/*	if (i > 0)
			{
				cout << "x方向:" << finalobjectpoints[i][index].x - finalobjectpoints[i - 1][index].x << endl;
				cout << "y方向:" << finalobjectpoints[i][index].y - finalobjectpoints[i - 1][index].y << endl;
				cout << "z方向:" << finalobjectpoints[i][index].z - finalobjectpoints[i - 1][index].z << endl;
			}*/
		}
		//更新数据
		this->finalobjectpoints.insert(make_pair(i, tempobjectpoints));
		tempobjectpoints.clear();
		str5 = "objectpoints";
		index = 0;
	}
	fs5.release();
}