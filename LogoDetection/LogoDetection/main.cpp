#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <string>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int getMatchIndex(Mat featureVector,
                  vector<KeyPoint> keypointsScene, Mat descScene);

int main(int argc, const char* argv[])
{
    //Input Image
    String fileLocation = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/LogoDetection/BurgerKingScenes/";
    String fileExtension = ".jpg";
    String fileName1 = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/Logos/burgerking_logo2.jpg";
    String fileName2 = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/BurgerKingScenes/burgerking_scene6.jpg";
    Mat objectImg = imread(fileName1);
    Mat sceneImg = imread(fileName2);
    
    
    
    //*SIFT
    //constructing with default parameters
    SIFT mySift;
    //Defining keypoints
    vector<KeyPoint> keypointsObject;
    vector<KeyPoint> keypointsScene;
    //getting keypoints and their locations
    Mat descObject;
    Mat descScene;
    //Initializing
    mySift.operator()(objectImg, noArray(),keypointsObject,descObject);
    mySift.operator()(sceneImg, noArray(),keypointsScene,descScene);
    
    
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
    
    
    Mat mergedImageSIFT = mergedImage.clone();
    //vector arrays to hold matched points
    vector<Point2f> objMatchPts;
    vector<Point2f> sceneMatchPts;
    //get the match index on the scene
    int matchPoints = 0;
    for(int i=0; i<keypointsObject.size(); i++){
        int fromIndex = i;
        int toIndex = getMatchIndex(descObject.row(i), keypointsScene, descScene);
        if(toIndex != -1){ //If good match
            //Declare the two points of the line
            Point2f fromPoint = keypointsObject[fromIndex].pt;    //From object image
            Point2f toPoint = keypointsScene[toIndex].pt;         //To scene image
            
            //Record Match Points
            objMatchPts.push_back(fromPoint);
            sceneMatchPts.push_back(toPoint);
            
            //Modifying to the appropriate location in the merged image
            toPoint = Point2f(toPoint.x + objectImg.cols, toPoint.y);
            
            //Draw the line
            line(mergedImageSIFT, fromPoint, toPoint, Scalar(0, 0, 255));
            //increment
            matchPoints = matchPoints + 1;
        }
    }
    //Display
    putText(mergedImageSIFT, "Match Points: " + to_string(matchPoints),cvPoint(20,20),
            FONT_HERSHEY_PLAIN, 1.5, cvScalar(0,0,255));
    namedWindow("SIFT"); imshow("SIFT", mergedImageSIFT);
    imwrite(fileLocation + fileName1 + fileName2 + fileExtension, mergedImageSIFT);
    
    
    //*Homography Frame
    //Obtain H
    Mat H = findHomography(objMatchPts, sceneMatchPts, CV_RANSAC);
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
    String resultType = "Frame";
    imwrite(fileLocation + fileName1 + fileName2 + resultType + fileExtension, mergedImageLINES);
    
    
      
    waitKey(0);
    return 0;
}

int getMatchIndex(Mat featureVector,
                  vector<KeyPoint> keypointsScene, Mat descScene){
    double minEucDist = 1e6; //some large number
    int matchIndex = -1;
    for(int i=0; i<keypointsScene.size(); i++){
        double eucDist = norm(featureVector, descScene.row(i));
        if(eucDist < minEucDist){
            minEucDist = eucDist;
            matchIndex = i;
        }
    }
    if(minEucDist > 110){
        return -1;
    }
    return matchIndex;
}