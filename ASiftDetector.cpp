#define _USE_MATH_DEFINES
#include "ASiftDetector.h"

using namespace cv;

ASiftDetector::ASiftDetector()
{

}

void ASiftDetector::detectAndComputeSingle(const Mat& img, int footprint_x, int footprint_y, std::vector< KeyPoint >& keypoints, Mat& descriptors)
{
    keypoints.clear();
    descriptors = Mat(0, 128, CV_32F);
    int modelnum = 0;
    for (int tl = 1; tl < 6; tl++)
    {
        double t = pow(2, 0.5 * tl);
        for (int phi = 0; phi < 180; phi += 72.0 / t)
        {
            std::vector<KeyPoint> kps;
            Mat desc;

            Mat timg, mask, Ai;
            img.copyTo(timg);

            affineSkew(t, phi, timg, mask, Ai);
            Mat A;
            invertAffineTransform(Ai, A);
#if 0
            Mat img_disp;
            bitwise_and(mask, timg, img_disp);
            namedWindow("Skew", WINDOW_AUTOSIZE);// Create a window for display.
            imshow("Skew", img_disp);
            waitKey(0);
#endif

            //SiftFeatureDetector detector;
            Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
            //Ptr<Feature2D> f3d = xfeatures2d::DAISY::create();

            //detector.detect(timg, kps, mask);
            //f2d->detect(timg, kps, mask);
            Point3f pt3_true(footprint_x, footprint_y, 1.);
            Mat kpt_true_coordinate = A * Mat(pt3_true);
            Point2f pt2_ture(kpt_true_coordinate.at<float>(0, 0), kpt_true_coordinate.at<float>(1, 0));
            KeyPoint kpts(pt2_ture, 5);
            kps.push_back(kpts);

            //SiftDescriptorExtractor extractor;
            //extractor.compute(timg, kps, desc);
            f2d->compute(timg, kps, desc);

            for (unsigned int i = 0; i < kps.size(); i++)
            {
                Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
                Mat kpt_t = Ai * Mat(kpt);
                kps[i].pt.x = kpt_t.at<float>(0, 0);
                kps[i].pt.y = kpt_t.at<float>(1, 0);
                keypoints.push_back(kps[i]);
            }
            descriptors.push_back(desc);

            ++modelnum;
            std::cout << "足印影像模型：" << modelnum << " 已完成..." << "\r";
        }
    }
//    for (int tl = 3; tl < 6; tl++)
//    {
//        double t = pow(2, 0.5 * tl);
//        for (int phi = 0; phi < 180; phi += 288.0 / t)
//        {
//            std::vector<KeyPoint> kps;
//            Mat desc;
//
//            Mat timg, mask, Ai;
//            img.copyTo(timg);
//
//            affineSkew(t, phi, timg, mask, Ai);
//            Mat A;
//            invertAffineTransform(Ai, A);
//#if 0
//            Mat img_disp;
//            bitwise_and(mask, timg, img_disp);
//            namedWindow("Skew", WINDOW_AUTOSIZE);// Create a window for display.
//            imshow("Skew", img_disp);
//            waitKey(0);
//#endif
//
//            //SiftFeatureDetector detector;
//            Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//            //Ptr<Feature2D> f3d = xfeatures2d::DAISY::create();
//
//            //detector.detect(timg, kps, mask);
//            //f2d->detect(timg, kps, mask);
//            Point3f pt3_true(footprint_x, footprint_y, 1.);
//            Mat kpt_true_coordinate = A * Mat(pt3_true);
//            Point2f pt2_ture(kpt_true_coordinate.at<float>(0, 0), kpt_true_coordinate.at<float>(1, 0));
//            KeyPoint kpts(pt2_ture, 5);
//            kps.push_back(kpts);
//
//            //SiftDescriptorExtractor extractor;
//            //extractor.compute(timg, kps, desc);
//            f2d->compute(timg, kps, desc);
//
//            for (unsigned int i = 0; i < kps.size(); i++)
//            {
//                Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
//                Mat kpt_t = Ai * Mat(kpt);
//                kps[i].pt.x = kpt_t.at<float>(0, 0);
//                kps[i].pt.y = kpt_t.at<float>(1, 0);
//                keypoints.push_back(kps[i]);
//            }
//            descriptors.push_back(desc);
//
//            ++modelnum;
//            std::cout << "足印影像模型：" << modelnum << " 已完成..." << "\r";
//        }
//    }
    std::cout << std::endl;
}

void ASiftDetector::detectAndComputeAll(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors)
{
    keypoints.clear();
    descriptors = Mat(0, 128, CV_32F);
    int modelnum = 0;
    for (int tl = 0; tl < 3; tl++)
    {
        double t = pow(2, tl);
        for (int phi = 0; phi < 180; phi += 144.0 / t)
        {
            std::vector<KeyPoint> kps;
            Mat desc;

            Mat timg, mask, Ai;
            img.copyTo(timg);

            affineSkew(t, phi, timg, mask, Ai);
            Mat A;
            invertAffineTransform(Ai, A);
#if 0
            Mat img_disp;
            bitwise_and(mask, timg, img_disp);
            namedWindow("Skew", WINDOW_AUTOSIZE);// Create a window for display.
            imshow("Skew", img_disp);
            waitKey(0);
#endif
            //SiftFeatureDetector detector;
            Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
            //Ptr<Feature2D> f3d = xfeatures2d::DAISY::create();

            //detector.detect(timg, kps, mask);
            //f2d->detect(timg, kps, mask);
            for (int i = 0; i < img.rows; i++)
            {
                for (int j = 0; j < img.cols; j++)
                {
                    Point3f pt3_true(j, i, 1.);
                    Mat kpt_true_coordinate = A * Mat(pt3_true);
                    Point2f pt2_ture(kpt_true_coordinate.at<float>(0, 0), kpt_true_coordinate.at<float>(1, 0));
                    KeyPoint kpts(pt2_ture, 5);
                    kps.push_back(kpts);
                }
            }

            //SiftDescriptorExtractor extractor;
            //extractor.compute(timg, kps, desc);
            f2d->compute(timg, kps, desc);

            for (unsigned int i = 0; i < kps.size(); i++)
            {
                Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
                Mat kpt_t = Ai * Mat(kpt);
                kps[i].pt.x = kpt_t.at<float>(0, 0);
                kps[i].pt.y = kpt_t.at<float>(1, 0);
                keypoints.push_back(kps[i]);
            }
            descriptors.push_back(desc);

            ++modelnum;
            std::cout << "仿真影像模型：" << modelnum << " 已完成..." << "\r";
        }
    }
//    for (int tl = 3; tl < 6; tl++)
//    {
//        double t = pow(2, 0.5 * tl);
//        for (int phi = 0; phi < 180; phi += 288.0 / t)
//        {
//            std::vector<KeyPoint> kps;
//            Mat desc;
//
//            Mat timg, mask, Ai;
//            img.copyTo(timg);
//
//            affineSkew(t, phi, timg, mask, Ai);
//            Mat A;
//            invertAffineTransform(Ai, A);
//#if 0
//            Mat img_disp;
//            bitwise_and(mask, timg, img_disp);
//            namedWindow("Skew", WINDOW_AUTOSIZE);// Create a window for display.
//            imshow("Skew", img_disp);
//            waitKey(0);
//#endif
//
//            //SiftFeatureDetector detector;
//            Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//            //Ptr<Feature2D> f3d = xfeatures2d::DAISY::create();
//
//            //detector.detect(timg, kps, mask);
//            //f2d->detect(timg, kps, mask);
//            for (int i = 0; i < img.rows; i++)
//            {
//                for (int j = 0; j < img.cols; j++)
//                {
//                    Point3f pt3_true(j, i, 1.);
//                    Mat kpt_true_coordinate = A * Mat(pt3_true);
//                    Point2f pt2_ture(kpt_true_coordinate.at<float>(0, 0), kpt_true_coordinate.at<float>(1, 0));
//                    KeyPoint kpts(pt2_ture, 5);
//                    kps.push_back(kpts);
//                }
//            }
//
//            //SiftDescriptorExtractor extractor;
//            //extractor.compute(timg, kps, desc);
//            f2d->compute(timg, kps, desc);
//
//            for (unsigned int i = 0; i < kps.size(); i++)
//            {
//                Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
//                Mat kpt_t = Ai * Mat(kpt);
//                kps[i].pt.x = kpt_t.at<float>(0, 0);
//                kps[i].pt.y = kpt_t.at<float>(1, 0);
//                keypoints.push_back(kps[i]);
//            }
//            descriptors.push_back(desc);
//
//            ++modelnum;
//            std::cout << "仿真影像模型：" << modelnum << " 已完成..." << "\r";
//        }
//    }
    std::cout << std::endl;
}

void ASiftDetector::affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai)
{
    int h = img.rows;
    int w = img.cols;

    mask = Mat(h, w, CV_8UC1, Scalar(255));

    Mat A = Mat::eye(2, 3, CV_32F);

    if (phi != 0.0)
    {
        phi *= M_PI / 180.;
        double s = sin(phi);
        double c = cos(phi);

        A = (Mat_<float>(2, 2) << c, -s, s, c);

        Mat corners = (Mat_<float>(4, 2) << 0, 0, w, 0, w, h, 0, h);
        Mat tcorners = corners * A.t();
        Mat tcorners_x, tcorners_y;
        tcorners.col(0).copyTo(tcorners_x);
        tcorners.col(1).copyTo(tcorners_y);
        std::vector<Mat> channels;
        channels.push_back(tcorners_x);
        channels.push_back(tcorners_y);
        merge(channels, tcorners);

        Rect rect = boundingRect(tcorners);
        A = (Mat_<float>(2, 3) << c, -s, -rect.x, s, c, -rect.y);

        warpAffine(img, img, A, Size(rect.width, rect.height), INTER_LINEAR, BORDER_REPLICATE);
    }
    if (tilt != 1.0)
    {
        double s = 0.8 * sqrt(tilt * tilt - 1);
        GaussianBlur(img, img, Size(0, 0), s, 0.01);
        resize(img, img, Size(0, 0), 1.0 / tilt, 1.0, INTER_NEAREST);
        A.row(0) = A.row(0) / tilt;
    }
    if (tilt != 1.0 || phi != 0.0)
    {
        h = img.rows;
        w = img.cols;
        warpAffine(mask, mask, A, Size(w, h), INTER_NEAREST);
    }
    invertAffineTransform(A, Ai);
}

void ASiftDetector::ASIFTMatchAndDraw(const std::string win, const cv::Mat img1, std::vector<cv::KeyPoint> kpts1, cv::Mat desc1,
    const cv::Mat img2, std::vector<cv::KeyPoint> kpts2, cv::Mat desc2)
{
    cv::FlannBasedMatcher fbmatcher;
    std::vector<cv::DMatch> good_matches;
    //std::vector<std::vector< cv::DMatch>> matches;
    fbmatcher.match(desc1, desc2, good_matches);
    //fbmatcher.knnMatch(desc1, desc2, matches, 2);

    //std::vector<cv::DMatch> good_matches;
    std::vector<cv::DMatch> final_match;
    /*for (int i = 0; i < matches.size(); ++i) {
          good_matches.push_back(matches[i][0]);
    }*/

    if (good_matches.size() == 0)
    {
        std::cout << "No match found!\n";
        exit(1);
    }
    else if (good_matches.size() == 1)
    {
        final_match.push_back(good_matches[0]);
    }
    else
    {
        for (int i = 0; i < good_matches.size() - 1; i++)
            if (good_matches[i].distance < good_matches[i + 1].distance)
                good_matches[i + 1] = good_matches[i];
        final_match.push_back(good_matches[(good_matches.size() - 1)]);
    }

    cv::Mat img_match;
    cv::drawMatches(img1, kpts1, img2, kpts2, final_match, img_match, cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow(win, img_match);
    cv::waitKey(0);
}

void ASiftDetector::ASIFTMatch(const cv::Mat img1, std::vector<cv::KeyPoint> kpts1, cv::Mat desc1, const cv::Mat img2,
    std::vector<cv::KeyPoint> kpts2, cv::Mat desc2, std::vector<cv::DMatch>& finalmatch)
{
    cv::FlannBasedMatcher fbmatcher;
    std::vector<cv::DMatch> good_matches;
    //std::vector<std::vector< cv::DMatch> > matches;
    fbmatcher.match(desc1, desc2, good_matches);

    //std::vector<cv::DMatch> good_matches;
   /* for (int i = 0; i < matches.size(); ++i) {
        good_matches.push_back(matches[i][0]);
    }*/

    if (good_matches.size()==0)
    {
        std::cerr << "No match found!\n";
        exit(1);
    }
    else if (good_matches.size()==1)
    {
        finalmatch.push_back(good_matches[0]);
    }
    else
    {
        for (int i = 0; i < good_matches.size() - 1; i++)
            if (good_matches[i].distance < good_matches[i + 1].distance)
                good_matches[i + 1] = good_matches[i];
        finalmatch.push_back(good_matches[(good_matches.size() - 1)]);
    }

}

//void ASiftDetector::detectAndCompute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors)
//{
//    keypoints.clear();
//    descriptors = Mat(0, 128, CV_32F);
//    for (int tl = 1; tl < 6; tl++)
//    {
//        double t = pow(2, 0.5 * tl);
//        for (int phi = 0; phi < 180; phi += 72.0 / t)
//        {
//            std::vector<KeyPoint> kps;
//            Mat desc;
//
//            Mat timg, mask, Ai;
//            img.copyTo(timg);
//
//            affineSkew(t, phi, timg, mask, Ai);
//
//#if 0
//            Mat img_disp;
//            bitwise_and(mask, timg, img_disp);
//            namedWindow("Skew", WINDOW_AUTOSIZE);// Create a window for display.
//            imshow("Skew", img_disp);
//            waitKey(0);
//#endif
//
//            //SiftFeatureDetector detector;
//            Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//
//            //detector.detect(timg, kps, mask);
//            f2d->detect(timg, kps, mask);
//
//            //SiftDescriptorExtractor extractor;
//            //extractor.compute(timg, kps, desc);
//            f2d->compute(timg, kps, desc);
//
//
//            for (unsigned int i = 0; i < kps.size(); i++)
//            {
//                Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
//                Mat kpt_t = Ai * Mat(kpt);
//                kps[i].pt.x = kpt_t.at<float>(0, 0);
//                kps[i].pt.y = kpt_t.at<float>(1, 0);
//            }
//            keypoints.insert(keypoints.end(), kps.begin(), kps.end());
//            descriptors.push_back(desc);
//        }
//    }
//}