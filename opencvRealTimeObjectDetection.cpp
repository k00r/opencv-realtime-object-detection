#include <iostream>
#include <stdio.h>
#include <time.h>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "cudaobjdetect.hpp"

using namespace std;
using namespace cv;

int main()
{
	VideoCapture myCapture("frankfurt.mp4");
	myCapture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	myCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	Mat img, gray;
	Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create("lbpcascade_humanFace_04-04-2018.xml");
	cascade_gpu->setMinNeighbors(50);
	cascade_gpu->setScaleFactor(1.1);
	cascade_gpu->setMinObjectSize(cascade_gpu->getClassifierSize());
	namedWindow("TestWindow", CV_WINDOW_AUTOSIZE);
	vector<Rect> objects;
	
	for (;;)
	{	
		clock_t start = clock();
		myCapture.read(img);
		cvtColor(img, gray, CV_BGR2GRAY);
		cascade_gpu->setMaxObjectSize(Size(gray.rows, gray.rows));
		cuda::GpuMat image_gpu(gray);
		cuda::GpuMat objbuf;
		cascade_gpu->detectMultiScale(image_gpu, objbuf);
		cascade_gpu->convert(objbuf, objects);
		if (objects.size() > 0)
		{
			for (int i = 0; i < objects.size(); i++)
			{
				rectangle(img, objects[i], Scalar(235, 235, 235), 2, 8, 0);
				putText(img, to_string(objects[i].width) + "x" + to_string(objects[i].height), Point(objects[i].x, objects[i].y - 7), FONT_HERSHEY_SIMPLEX, 0.43, Scalar(235, 235, 235), 1, 8, false);
			}
		}
		clock_t stop = clock();
		clock_t clockTicksTaken = stop - start;
		float elapsed = (float)(clockTicksTaken);
		putText(img, "FPS: " + to_string(1000.f / elapsed), Point(20, 30), FONT_HERSHEY_PLAIN, 1.25, Scalar(0, 255, 0), 2, 8, false);
		putText(img, "scaleFactor: " + to_string(cascade_gpu->getScaleFactor()), Point(20, 50), FONT_HERSHEY_PLAIN, 1.25, Scalar(0, 255, 0), 2, 8, false);
		putText(img, "minNeighbours: " + to_string(cascade_gpu->getMinNeighbors()), Point(20, 70), FONT_HERSHEY_PLAIN, 1.25, Scalar(0, 255, 0), 2, 8, false);
		putText(img, "minObjectSize: " + to_string(cascade_gpu->getMinObjectSize().width) + "x" + to_string(cascade_gpu->getMinObjectSize().height), Point(20, 90), FONT_HERSHEY_PLAIN, 1.25, Scalar(0, 255, 0), 2, 8, false);
		putText(img, "maxObjectSize: " + to_string(cascade_gpu->getMaxObjectSize().width) + "x" + to_string(cascade_gpu->getMaxObjectSize().height), Point(20, 110), FONT_HERSHEY_PLAIN, 1.25, Scalar(0, 255, 0), 2, 8, false);
		imshow("TestWindow", img);
		waitKey(1);
	}
	return 0;
}
