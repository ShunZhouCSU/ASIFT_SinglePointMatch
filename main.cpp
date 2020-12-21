#include "ASiftDetector.h"
#include <iostream>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "tinyxml2.h"

using namespace tinyxml2;

bool ReadParaXml(std::string m_strXmlPath, std::vector<double>& vecNode)
{
	//读取xml文件中的参数值
	XMLDocument Document;
	if (Document.LoadFile(m_strXmlPath.c_str()) != XML_SUCCESS)
	{
		std::cout << "Not a XML file!" << std::endl;
		return false;
	}

	XMLElement* RootElement = Document.RootElement();		//根目录
	const char* RootElementFilename = RootElement->Attribute("name");//读取根目录的名称
	if (strcmp(RootElementFilename, "ObtainFootPrintImagePosInfoInStereoImage") == 0)//判断根目录名称是否正确
	{
		std::cout << "**********XML document is ready!**********" << std::endl;
	}
	else
	{
		std::cout << "XML document name error! Please select another one." << std::endl;
		return false;
	}

	XMLElement* NextElement = RootElement->FirstChildElement("InputLaserPointsSourceList")->FirstChildElement("LaserPointInfo");	//根目录下的第二个节点层
	while (NextElement)		//判断有没有读完
	{
		XMLElement* finalElement = NextElement->FirstChildElement("FootPrintImage");
		double origin_col = atof(finalElement->FirstChildElement("LaserPtCenterInImage")->FirstChildElement("x")->GetText());
		double origin_row = atof(finalElement->FirstChildElement("LaserPtCenterInImage")->FirstChildElement("y")->GetText());
		//加入到向量中
		vecNode.push_back(origin_col);
		vecNode.push_back(origin_row);
		NextElement = NextElement->NextSiblingElement();
	}
	std::cout << "The XML file is parsed ! Waiting for processing..." << std::endl;
	std::cout << "******************************************" << std::endl;
	return true;
}

int main()
{
	//std::string str_img1 = std::string(SOURCE_DIR) + "/data/graffA.jpg";
	//std::string str_img2 = std::string(SOURCE_DIR) + "/data/graffB.jpg";

	//读取所有影像及xml文件中足印点坐标 2020.11.29
	cv::String dir_path_foot = "D:/SinglePointMatching8.3/FootprintAndSimulateIMG/9874(8bit)/Laser";
	cv::String dir_path_Simulate = "D:/SinglePointMatching8.3/FootprintAndSimulateIMG/9874(8bit)/output";
	std::vector<cv::String> img_footprint_files, img_Simulate_files;
	cv::glob(dir_path_foot, img_footprint_files);
	cv::glob(dir_path_Simulate, img_Simulate_files);
	if (img_footprint_files.size()==0 || img_Simulate_files.size()==0)
	{
		std::cerr << "No image files!\n";
		return -1;
	}
	std::vector<double> coordinate;
	ReadParaXml("D:\\SinglePointMatching8.3\\FootprintAndSimulateIMG\\9874\\InputParam_9874#E101.9_N28.5_LaserPtPosInImg.xml", coordinate);

	//cv::Mat img1 = cv::imread("D:/SinglePointMatching8.3/FootprintAndSimulateIMG/4899(8bit)/Laser/895229917_FootprintImage.tif");
	//cv::Mat img2 = cv::imread("D:/SinglePointMatching8.3/FootprintAndSimulateIMG/4899(8bit)/output/011-4899#E120.4_N31.3#5-BWD-Simulate.tif");

	int dx = 0;
	for (int i = 0; i < img_footprint_files.size(); i++)
	{
		std::cout << "THE " << i + 1 << " IMG\n";
		int a = 2 * i;
		int b = 2 * i + 1;
		std::vector<cv::KeyPoint> kpts1, kpts2;
		cv::Mat des1, des2;
		std::vector<cv::DMatch> matches1;
		ASiftDetector asift;
		cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();

		//将图像从路径中转化为mat格式并读取足印点坐标 2020.11.29
		cv::Mat footimg = imread(img_footprint_files[i]);
		cv::Mat BWDimg = imread(img_Simulate_files[a]);
		//cv::Mat FWDimg = imread(img_Simulate_files[b]);
		double BaseCol = coordinate[a];
		double BaseRow = coordinate[b];
		int FootCol = (int)(BaseCol + 0.5);
		int FootRow = (int)(BaseRow + 0.5);

		asift.detectAndComputeSingle(footimg, FootCol, FootRow, kpts1, des1);//足印影像描述子

		for (int i = 0; i != BWDimg.cols; i++)
			for (int j = 0; j != BWDimg.rows; j++)
			{
				cv::Point2f pt(i, j);
				cv::KeyPoint kpt(pt, 5);
				kpts2.push_back(kpt);
			}
		f2d->compute(BWDimg, kpts2, des2);
		std::cout << "仿真影像已完成!\n";
		//asift.detectAndComputeAll(BWDimg, kpts2, des2);//后视仿真描述子
		//asift.detectAndComputeAll(FWDimg, kpts3, des3);//前视仿真描述子

		std::cout << "\n";
		std::cout << "Size of des1(footimg): [" << des1.cols << "*" << des1.rows << "]\n";
		std::cout << "Size of des2(simulate_BWD): [" << des2.cols << "*" << des2.rows << "]\n";
		//std::cout << "Size of des2(simulate_FWD): [" << des3.cols << "*" << des3.rows << "]\n\n";
		std::cout << "matching...\n\n";

		asift.ASIFTMatch(footimg, kpts1, des1, BWDimg, kpts2, des2, matches1);
		//asift.ASIFTMatch(footimg, kpts1, des1, FWDimg, kpts3, des3, matches2);

		//保存匹配图像 2020.11.29
		cv::Mat outimg_BWD_matches;
		drawMatches(footimg, kpts1, BWDimg, kpts2, matches1, outimg_BWD_matches, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//drawMatches(footimg, kpts1, FWDimg, kpts3, matches2, outimg_FWD_matches, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		char path1[128] = { 0 };
		//char path2[128] = { 0 };
		sprintf_s(path1, "D:\\SinglePointMatching8.3\\FootprintAndSimulateIMG\\9874(8bit)\\match_image(1simulate)\\%d.tif", ++dx);
		imwrite(path1, outimg_BWD_matches);
		/*sprintf_s(path2, "D:\\SinglePointMatching8.3\\FootprintAndSimulateIMG\\4899(8bit)\\match_image(t=0,1,2 phi=144)\\%d.tif", ++dx);
		imwrite(path2, outimg_FWD_matches);*/

		//asift.ASIFTMatchAndDraw("match img(ASIFT)", img1, kpts1, des1, img2, kpts2, des2);
	}
	
	return 0;

}

//int main()
//{
//	//std::string str_img1 = std::string(SOURCE_DIR) + "/data/graffA.jpg";
//	//std::string str_img2 = std::string(SOURCE_DIR) + "/data/graffB.jpg";
//
//	cv::Mat img1 = cv::imread("D:/SinglePointMatching8.3/FootprintAndSimulateIMG/4899(8bit)/Laser/895229910_FootprintImage.tif");
//	cv::Mat img2 = cv::imread("D:/SinglePointMatching8.3/FootprintAndSimulateIMG/4899(8bit)/output/005-4899#E120.4_N31.3#2-BWD-Simulate.tif");
//
//	std::vector<cv::KeyPoint> kpts1, kpts2;
//	cv::Mat des1, des2;
//	ASiftDetector asift;
//	cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
//
//	asift.detectAndComputeSingle(img1, 231, 321, kpts1, des1);//足印影像描述子
//	//asift.detectAndComputeAll(img2, kpts2, des2);//后视仿真描述子
//	for (int i = 0; i != img2.cols; i++)
//		for (int j = 0; j != img2.rows; j++)
//		{
//			cv::Point2f pt(i, j);
//			cv::KeyPoint kpt(pt, 5);
//			kpts2.push_back(kpt);
//		}
//	f2d->compute(img2, kpts2, des2);
//	std::cout << "仿真影像已完成!\n";
//
//	std::cout << "\n";
//	std::cout << "Size of des1(footimg): [" << des1.cols << "*" << des1.rows << "]\n";
//	std::cout << "Size of des2(simulate_BWD): [" << des2.cols << "*" << des2.rows << "]\n";
//	std::cout << "matching...\n\n";
//
//	asift.ASIFTMatchAndDraw("ASIFT", img1, kpts1, des1, img2, kpts2, des2);
//	
//	return 0;
//
//}