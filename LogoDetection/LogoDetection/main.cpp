#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace std;

Mat merge_images(Mat objectImg , Mat sceneImg);

int main( int argc, char** argv )
{
    
    String logo = "burgerking";
    int sceneNumber = 1;
    String file_path = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/";
    String logo_path = "Logos/" + logo + "_logo.jpg";
    String object_path = logo + "_scenes/" + logo + "_scene" + to_string(sceneNumber) + ".jpg";
    
    
    cout << file_path + logo_path << endl;
    cout << file_path + object_path << endl;
    
    Mat objectImg = imread(file_path + logo_path);
    Mat sceneImg = imread(file_path + object_path);
    
    if( !objectImg.data || !sceneImg.data){
        cout << "Error with image files" << endl;
        return -1;
    }
    
    
    //-- Step 1: Detect the keypoints using SIFT Detector
    //int minHessian = 400;
    SiftFeatureDetector detector;
    std::vector<KeyPoint> object_keypoints, scene_keypoints;
    detector.detect(objectImg,object_keypoints);
    detector.detect(sceneImg,scene_keypoints);
    
    
    //-- Step 2: Calculate descriptors (feature vectors)
    SiftDescriptorExtractor extractor;
    Mat object_descriptors,scene_descriptors;
    extractor.compute(objectImg,object_keypoints,object_descriptors);
    extractor.compute(sceneImg,scene_keypoints,scene_descriptors);
    
    
    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(object_descriptors,scene_descriptors,matches);
    
    double max_dist = 0; double min_dist = 100;
    
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < object_descriptors.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    
    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    
    std::vector< DMatch > good_matches;
    float threshold = (min_dist * 0.2 + max_dist * 0.8);
    for( int i = 0; i < object_descriptors.rows; i++ ){
        if(matches[i].distance <= threshold){
            good_matches.push_back(matches[i]);
        }
    }
    
    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( objectImg, object_keypoints, sceneImg, scene_keypoints,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    //-- Show detected matches
    imshow( "Good Matches", img_matches );
    
    
    cout << "Number of matches: " << good_matches.size() << endl;
    
    
    
    //vector arrays to hold matched points
    vector<Point2f> object_matchpoints;
    vector<Point2f> scene_matchpoints;
    
    for(int i=0; i<good_matches.size(); i++){
        int object_index = good_matches[i].queryIdx;
        int scene_index = good_matches[i].trainIdx;
        
        object_matchpoints.push_back(object_keypoints[object_index].pt);
        scene_matchpoints.push_back(scene_keypoints[scene_index].pt);
        
    }
    
    
    //*Homography Frame
    //Obtain H
    Mat H = findHomography(object_matchpoints, scene_matchpoints, CV_RANSAC);
    cout << "H: \n" << H << endl;
    //Define Corners
    vector<Point2f> objectCorners(4);
    vector<Point2f> sceneCorners(4);
    objectCorners[0] = cvPoint(0,0); //top-left
    objectCorners[1] = cvPoint(objectImg.cols, 0); //top-right
    objectCorners[2] = cvPoint(objectImg.cols, objectImg.rows); //bottom-right
    objectCorners[3] = cvPoint(0, objectImg.rows); //bottom-left
    
    //Obtain the corners in the scene image for the object by the transform of H
    perspectiveTransform(objectCorners, sceneCorners, H);
    //Find the object (Draw the frame in the scene image around the detected object);
    Mat mergedImage = merge_images(objectImg, sceneImg);
    Mat mergedImageLINES = mergedImage.clone();
    sceneCorners[0] = Point2f(sceneCorners[0].x + objectImg.cols, sceneCorners[0].y);
    sceneCorners[1] = Point2f(sceneCorners[1].x + objectImg.cols, sceneCorners[1].y);
    sceneCorners[2] = Point2f(sceneCorners[2].x + objectImg.cols, sceneCorners[2].y);
    sceneCorners[3] = Point2f(sceneCorners[3].x + objectImg.cols, sceneCorners[3].y);
    line(mergedImageLINES, sceneCorners[0], sceneCorners[1], Scalar(255,0,255),4);
    line(mergedImageLINES, sceneCorners[1], sceneCorners[2], Scalar(255,0,255),4);
    line(mergedImageLINES, sceneCorners[2], sceneCorners[3], Scalar(255,0,255),4);
    line(mergedImageLINES, sceneCorners[3], sceneCorners[0], Scalar(255,0,255),4);
    //Display
    namedWindow("Frame"); imshow("Frame", mergedImageLINES);
    
    
    
    //Write Results to file
    imwrite("/Users/ahmetcanozbek/Desktop/result.jpg", mergedImageLINES);
    
    
    
    
    
    waitKey(0);
    
    return 0;
}


Mat merge_images(Mat objectImg , Mat sceneImg){
    
    //Merging the object image and the scene image so that we can see it in one display
    Mat mergedImage;
    if(objectImg.rows > sceneImg.rows){
        //If object image is longer than the scene image
        Mat paddedSceneImg;
        vconcat(sceneImg, Mat::zeros((objectImg.rows-sceneImg.rows), sceneImg.cols, sceneImg.type()), paddedSceneImg);
        hconcat(objectImg, paddedSceneImg, mergedImage);
    }else{
        //If the scene image is longer than the object image
        Mat paddedObjectImg;
        vconcat(objectImg, Mat::zeros((sceneImg.rows-objectImg.rows), objectImg.cols, objectImg.type()), paddedObjectImg);
        hconcat(paddedObjectImg, sceneImg, mergedImage);
    }
    
    return mergedImage;
}


//#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/features2d.hpp>
//#include "opencv2/calib3d/calib3d.hpp"
//#include <string>
//#include <opencv2/imgproc/imgproc.hpp>
//
//using namespace std;
//using namespace cv;
//
//int getMatchIndex(Mat featureVector,
//                  vector<KeyPoint> keypointsScene, Mat descScene);
//
//int main(int argc, const char* argv[])
//{
//    //Input Image
//    String fileLocation = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/LogoDetection/BurgerKingScenes/";
//    String fileExtension = ".jpg";
//    String fileName1 = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/Logos/ford_logo.png";
//    String fileName2 = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/ford_scenes/ford_scene2.jpg";
//    Mat objectImg = imread(fileName1);
//    Mat sceneImg = imread(fileName2);
//    
//    namedWindow("Object"); imshow("Object", objectImg);
//    namedWindow("Scene"); imshow("Scene", sceneImg);
//    
//    //waitKey(0);
//    
//    //*SIFT
//    //constructing with default parameters
//    SIFT mySift;
//    //Defining keypoints
//    vector<KeyPoint> keypointsObject;
//    vector<KeyPoint> keypointsScene;
//    //getting keypoints and their locations
//    Mat descObject;
//    Mat descScene;
//    //Initializing
//    mySift.operator()(objectImg, noArray(),keypointsObject,descObject);
//    mySift.operator()(sceneImg, noArray(),keypointsScene,descScene);
//    
//    
//    //Merging the object image and the scene image so that we can see it in one display
//    Mat mergedImage;
//    if(objectImg.rows > sceneImg.rows){
//        //If object image is longer than the scene image
//        Mat paddedSceneImg;
//        vconcat(sceneImg, Mat::zeros((objectImg.rows-sceneImg.rows), sceneImg.cols, sceneImg.type()), paddedSceneImg);
//        hconcat(objectImg, paddedSceneImg, mergedImage);
//    }else{
//        //If the scene image is longer than the object image
//        Mat paddedObjectImg;
//        vconcat(objectImg, Mat::zeros((sceneImg.rows-objectImg.rows), objectImg.cols, objectImg.type()), paddedObjectImg);
//        hconcat(paddedObjectImg, sceneImg, mergedImage);
//    }
//    
//    
//    Mat mergedImageSIFT = mergedImage.clone();
//    //vector arrays to hold matched points
//    vector<Point2f> objMatchPts;
//    vector<Point2f> sceneMatchPts;
//    //get the match index on the scene
//    int matchPoints = 0;
//    for(int i=0; i<keypointsObject.size(); i++){
//        int fromIndex = i;
//        int toIndex = getMatchIndex(descObject.row(i), keypointsScene, descScene);
//        if(toIndex != -1){ //If good match
//            //Declare the two points of the line
//            Point2f fromPoint = keypointsObject[fromIndex].pt;    //From object image
//            Point2f toPoint = keypointsScene[toIndex].pt;         //To scene image
//            
//            //Record Match Points
//            objMatchPts.push_back(fromPoint);
//            sceneMatchPts.push_back(toPoint);
//            
//            //Modifying to the appropriate location in the merged image
//            toPoint = Point2f(toPoint.x + objectImg.cols, toPoint.y);
//            
//            //Draw the line
//            line(mergedImageSIFT, fromPoint, toPoint, Scalar(0, 0, 255));
//            //increment
//            matchPoints = matchPoints + 1;
//        }
//    }
//    //Display
//    putText(mergedImageSIFT, "Match Points: " + to_string(matchPoints),cvPoint(20,20),
//            FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,255));
//    namedWindow("SIFT"); imshow("SIFT", mergedImageSIFT);
//    imwrite(fileLocation + fileName1 + fileName2 + fileExtension, mergedImageSIFT);
//    
//    
//    //*Homography Frame
//    //Obtain H
//    Mat H = findHomography(objMatchPts, sceneMatchPts, CV_RANSAC);
//    cout << "H: \n" << H << endl;
//    //Define Corners
//    vector<Point2f> objectCorners(4);
//    vector<Point2f> sceneCorners(4);
//    objectCorners[0] = cvPoint(0,0); //top-left
//    objectCorners[1] = cvPoint(objectImg.cols, 0); //top-right
//    objectCorners[2] = cvPoint(objectImg.cols, objectImg.rows); //bottom-right
//    objectCorners[3] = cvPoint(0, objectImg.rows); //bottom-left
//    
//    //Obtain the corners in the scene image for the object by the transform of H
//    perspectiveTransform(objectCorners, sceneCorners, H);
//    //Find the object (Draw the frame in the scene image around the detected object);
//    Mat mergedImageLINES = mergedImage.clone();
//    sceneCorners[0] = Point2f(sceneCorners[0].x + objectImg.cols, sceneCorners[0].y);
//    sceneCorners[1] = Point2f(sceneCorners[1].x + objectImg.cols, sceneCorners[1].y);
//    sceneCorners[2] = Point2f(sceneCorners[2].x + objectImg.cols, sceneCorners[2].y);
//    sceneCorners[3] = Point2f(sceneCorners[3].x + objectImg.cols, sceneCorners[3].y);
//    line(mergedImageLINES, sceneCorners[0], sceneCorners[1], Scalar(255,0,255),4);
//    line(mergedImageLINES, sceneCorners[1], sceneCorners[2], Scalar(255,0,255),4);
//    line(mergedImageLINES, sceneCorners[2], sceneCorners[3], Scalar(255,0,255),4);
//    line(mergedImageLINES, sceneCorners[3], sceneCorners[0], Scalar(255,0,255),4);
//    //Display
//    namedWindow("Frame"); imshow("Frame", mergedImageLINES);
//    String resultType = "Frame";
//    imwrite(fileLocation + fileName1 + fileName2 + resultType + fileExtension, mergedImageLINES);
//    
//    
//      
//    waitKey(0);
//    return 0;
//}
//
//int getMatchIndex(Mat featureVector,
//                  vector<KeyPoint> keypointsScene, Mat descScene){
//    double minEucDist = 1e6; //some large number
//    int matchIndex = -1;
//    for(int i=0; i<keypointsScene.size(); i++){
//        double eucDist = norm(featureVector, descScene.row(i));
//        if(eucDist < minEucDist){
//            minEucDist = eucDist;
//            matchIndex = i;
//        }
//    }
//    if(minEucDist > 5000){
//        return -1;
//    }
//    return matchIndex;
//}