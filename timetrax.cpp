/*

███████╗███████╗███╗   ██╗██████╗      ██████╗ ██████╗ ██████╗ ██████╗
██╔════╝██╔════╝████╗  ██║██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔══██╗
███████╗█████╗  ██╔██╗ ██║██║  ██║    ██║     ██║   ██║██████╔╝██████╔╝
╚════██║██╔══╝  ██║╚██╗██║██║  ██║    ██║     ██║   ██║██╔══██╗██╔═══╝
███████║███████╗██║ ╚████║██████╔╝    ╚██████╗╚██████╔╝██║  ██║██║
╚══════╝╚══════╝╚═╝  ╚═══╝╚═════╝      ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝

██████╗ ███████╗ █████╗ ██████╗     ████████╗██╗  ██╗██╗███████╗    ███████╗██╗  ██╗██╗████████╗
██╔══██╗██╔════╝██╔══██╗██╔══██╗    ╚══██╔══╝██║  ██║██║██╔════╝    ██╔════╝██║  ██║██║╚══██╔══╝
██████╔╝█████╗  ███████║██║  ██║       ██║   ███████║██║███████╗    ███████╗███████║██║   ██║
██╔══██╗██╔══╝  ██╔══██║██║  ██║       ██║   ██╔══██║██║╚════██║    ╚════██║██╔══██║██║   ██║
██║  ██║███████╗██║  ██║██████╔╝       ██║   ██║  ██║██║███████║    ███████║██║  ██║██║   ██║
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝╚══════╝    ╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝

1. installera opencv3, brew install opencv3 --with-ffmpeg (brew info opencv3 för att lista alla flaggor)
2. kompilera med: g++ $(pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.2.0/lib/pkgconfig/opencv.pc) timetrax.cpp -o timetrax
3. ???
4. ./timetrax
5  profit!
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    // Camera frame
    cv::Mat frame;

    // >>>> Kalman Filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;

    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
    cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A, set dT at each processing step
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));

    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // <<<< Kalman Filter

    // Camera Index
    int idx = 0;

    // Camera Capture
    cv::VideoCapture cap;

    // >>>>> Setup shit
    if (!cap.open("klonk2.mp4"))
    {
        cout << "Webcam not connected.\n" << "Please verify\n";
        return EXIT_FAILURE;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 854);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    cout << "\nHit 'q' to exit...\n";

    char ch = 0;

    double ticks = 0;
    bool found = false;

    int notFoundCount = 0;

    // >>>>> Main loop
    while (ch != 'q' && ch != 'Q')
    {
        double precTick = ticks;
        ticks = (double) cv::getTickCount();

        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

        // Get a frame
        cap >> frame;

        cv::resize(frame, frame, cv::Size(854, 480));

        cv::Mat res;
        frame.copyTo( res );

        if (found)
        {
            // >>>> Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            // <<<< Matrix A

            //cout << "dT:" << endl << dT << endl;
            state = kf.predict();
            //cout << "State post:" << endl << state << endl;

            cv::Rect predRect;
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;

            cv::Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            cv::circle(res, center, 2, CV_RGB(255,0,0), -1);

            cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            //cv::putText(res, sstr.str(),
            //            cv::Point(center.x + 15, center.y + 30),
            //            cv::FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255,0,0), 2);
        }

        // >>>>> Blur it
        cv::Mat blur;
        cv::GaussianBlur(frame, blur, cv::Size(5, 5), 3.0, 3.0);
        // <<<<< Blur it

        // >>>>> convert to HSV
        cv::Mat frmHsv;
        cv::cvtColor(blur, frmHsv, CV_BGR2HSV);
        // <<<<< convert to HSV

        // >>>>> Color Thresholding (find sendball)
        cv::Mat rangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::inRange(frmHsv, cv::Scalar(0, 130, 130),
                    cv::Scalar(15, 255, 255), rangeRes);
        // <<<<< Color Thresholding (find sendball)

        // >>>>> Improve result
        cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
        // <<<<< Improve result

        //cv::imshow("Threshold", rangeRes);

        // >>>>> Find contours
        vector<vector<cv::Point> > contours;
        cv::findContours(rangeRes, contours, CV_RETR_EXTERNAL,
                         CV_CHAIN_APPROX_NONE);
        // <<<<< Find contours

        // >>>>> Filtering
        vector<vector<cv::Point> > balls;
        vector<cv::Rect> ballsBox;
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Rect bBox;
            bBox = cv::boundingRect(contours[i]);

            float ratio = (float) bBox.width / (float) bBox.height;
            if (ratio > 1.0f)
                ratio = 1.0f / ratio;

            // Searching for a bBox "rounded" square
            if (ratio > 0.75 && bBox.area() >= 300)
            {
                balls.push_back(contours[i]);
                ballsBox.push_back(bBox);
            }
        }
        // <<<<< Filtering

        //cout << "Sendballs:" << ballsBox.size() << endl;
        cv::Rect goal(763,195, 45,100); // x, y, width, height
        cv::Rect goal2(85,195, 45,100);
        cv::rectangle(res, goal, CV_RGB(0,0,255), 2);
        cv::rectangle(res, goal2, CV_RGB(0,0,255), 2);

        // >>>>> Detection result
        for (size_t i = 0; i < balls.size(); i++)
        {
            cv::drawContours(res, balls, i, CV_RGB(20,150,20), 1);
            cv::rectangle(res, ballsBox[i], CV_RGB(0,255,0), 2);

            cv::Point center;
            center.x = ballsBox[i].x + ballsBox[i].width / 2;
            center.y = ballsBox[i].y + ballsBox[i].height / 2;
            cv::circle(res, center, 2, CV_RGB(20,150,20), -1);

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            //cv::putText(res, sstr.str(),
            //         cv::Point(center.x + 15, center.y - 15),
            //         cv::FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(0,255,0), 2);

        }
        // <<<<< Detection result

        // >>>>> Kalman Update
        if (balls.size() == 0)
        {
            notFoundCount++;
            //cout << "notFoundCount:" << notFoundCount << endl;
            if( notFoundCount >= 100 )
            {
                found = false;
            }
            /*else
                kf.statePost = state;*/
        }
        else
        {
            notFoundCount = 0;

            meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
            meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
            meas.at<float>(2) = (float)ballsBox[0].width;
            meas.at<float>(3) = (float)ballsBox[0].height;

            if (!found) // First detection!
            {
                // >>>> Init
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1; // px
                kf.errorCovPre.at<float>(35) = 1; // px

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Init

                kf.statePost = state;

                found = true;
            }
            else
                kf.correct(meas); // kalman correction

            //cout << "measure matrix:" << endl << meas << endl;
        }
        // <<<<< kalman update

        // Check if ball is sent to goal
        if (goal.contains(cv::Point(state.at<float>(0), state.at<float>(1)))) {
               cout << "HANNES GJORDE MÅÅÅL!" << endl;
            }
         if (goal2.contains(cv::Point(state.at<float>(0), state.at<float>(1)))) {
               cout << "REZEK GJORDE MÅÅÅL!" << endl;
            }

        cv::imshow("Tracking", res);

        ch = cv::waitKey(1);
    }

    return EXIT_SUCCESS;
}