#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <string>

using namespace std;
using namespace cv;

int getMatchIndex(Mat featureVector,
                  vector<KeyPoint> keypointsScene, Mat descScene);

int main(int argc, const char* argv[])
{
    //Input Image
    String fileLocation = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/LogoDetection/BurgerKingScenes/";
    String fileExtension = ".jpg";
    String fileName1 = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/LogoDetection/burgerking_logo.jpg";
    String fileName2 = "/Users/ahmetcanozbek/Desktop/CodePortfolio/LogoDetection/LogoDetection/LogoDetection/burgerking_scene3.jpg";
    Mat objectImg = imread(fileName1);
    Mat sceneImg = imread(fileName2);
    
    namedWindow("objectIm"); imshow("objectIm", objectImg);
    namedWindow("sceneIm"); imshow("sceneIm", sceneImg);
    
    //waitKey(0);
    
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
    if(objectImg.rows > sceneImg.rows){
        Mat paddedSceneImg;
        vconcat(sceneImg, Mat::zeros((objectImg.rows-sceneImg.rows), sceneImg.cols, sceneImg.type()), paddedSceneImg);
        
        Mat mergedImage;
        hconcat(objectImg, paddedSceneImg, mergedImage);
    }else{
        
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
            toPoint = Point2f(toPoint.x + 480, toPoint.y);
            
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
    sceneCorners[0] = Point2f(sceneCorners[0].x + 480, sceneCorners[0].y);
    sceneCorners[1] = Point2f(sceneCorners[1].x + 480, sceneCorners[1].y);
    sceneCorners[2] = Point2f(sceneCorners[2].x + 480, sceneCorners[2].y);
    sceneCorners[3] = Point2f(sceneCorners[3].x + 480, sceneCorners[3].y);
    line(mergedImageLINES, sceneCorners[0], sceneCorners[1], Scalar(255,0,255),4);
    line(mergedImageLINES, sceneCorners[1], sceneCorners[2], Scalar(255,0,255),4);
    line(mergedImageLINES, sceneCorners[2], sceneCorners[3], Scalar(255,0,255),4);
    line(mergedImageLINES, sceneCorners[3], sceneCorners[0], Scalar(255,0,255),4);
    //Display
    namedWindow("Frame"); imshow("Frame", mergedImageLINES);
    String resultType = "Frame";
    imwrite(fileLocation + fileName1 + fileName2 + resultType + fileExtension, mergedImageLINES);
    
    
    //*Homography Matching Points
    vector<Point2f> siftKeypointsObject;
    for(int i=0; i<keypointsObject.size();i++){
        siftKeypointsObject.push_back(keypointsObject[i].pt);
    }
    vector<Point2f> siftKeypointsSceneNormal;
    for(int i=0; i<keypointsScene.size();i++){
        siftKeypointsSceneNormal.push_back(keypointsScene[i].pt);
    }
    
    
    Mat outDraw = mergedImage.clone();
    for(int i=0; i<siftKeypointsObject.size();i++){
        circle(outDraw,siftKeypointsObject[i], 3, Scalar(0,0,255));
    }
    vector<Point2f> siftKeypointsSceneByH;
    perspectiveTransform(siftKeypointsObject, siftKeypointsSceneByH, H);
    //Red by H (the points on the object image transformed by H matrix onto the test(scene) image)
    for(int i=0; i<siftKeypointsSceneByH.size(); i++){
        siftKeypointsSceneByH[i] = Point2f(siftKeypointsSceneByH[i].x + 480, siftKeypointsSceneByH[i].y);
        circle(outDraw,siftKeypointsSceneByH[i], 2, Scalar(0,0,255),1);
    }
    
    //Green (SIFT feature points) (the feature points calculated by SIFT algorithm on the test(scene) image)
    for(int i=0; i<siftKeypointsSceneNormal.size(); i++){
        siftKeypointsSceneNormal[i] = Point2f(siftKeypointsSceneNormal[i].x + 480, siftKeypointsSceneNormal[i].y);
        circle(outDraw,siftKeypointsSceneNormal[i], 2, Scalar(0,255,0),1);
    }
    
    namedWindow("Homography"); imshow("Homography", outDraw);
    resultType = "Homography";
    imwrite(fileLocation + fileName1 + fileName2 + resultType + fileExtension, outDraw);

    
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