//
// Created by hyx on 2020/10/26.
//
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <fstream>
#include "yolo_v2_class.hpp"
#include <mynteyed/camera.h>
#include <mynteyed/utils.h>
#include <time.h>
#include <sstream>

using namespace std;
using namespace cv;
MYNTEYE_USE_NAMESPACE
#define GPU

//画出检测框和相关信息
void DrawBoxes(Mat &frame, vector<string> classes, int classId, float conf, int left, int top, int right, int bottom) {
    //画检测框
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    //该检测框对应的类别和置信度
    string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ":" + label;
    }
    //将标签显示在检测框顶部
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)),
              Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}


//画出检测结果
void Drawer(Mat &frame, vector<bbox_t> outs, vector<string> classes) {
    //获取所有最佳检测框信息
    for (int i = 0; i < outs.size(); i++) {
        DrawBoxes(frame, classes, outs[i].obj_id, outs[i].prob, outs[i].x, outs[i].y,
                  outs[i].x + outs[i].w, outs[i].y + outs[i].h);
    }
}


int main(void) {
    string classesFile = "./coco.names";
    string modelConfig = "./cfg/yolov4.cfg";
    string modelWeights = "./weights/yolov4.weights";

    //加载类别名

    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    //加载网络模型，0是指定第一块GPU
    Detector detector(modelConfig, modelWeights, 0);
    cv::namedWindow("a");

    Camera cam;
    DeviceInfo dev_info;
    if (!util::select(cam, &dev_info)) {
        return 1;
    }
    util::print_stream_infos(cam, dev_info.index);

    std::cout << "Open device: " << dev_info.index << ", "
              << dev_info.name << std::endl << std::endl;

    OpenParams params(dev_info.index);
    params.stream_mode = StreamMode::STREAM_1280x720;
    params.ir_intensity = 4;
    params.framerate = 30;

    cam.Open(params);

    std::cout << std::endl;
    if (!cam.IsOpened()) {
        std::cerr << "Error: Open camera failed" << std::endl;
        return 1;
    }
    std::cout << "Open device success" << std::endl << std::endl;

    std::cout << "Press ESC/Q on Windows to terminate" << std::endl;

    clock_t start,ends;
    cv::namedWindow("video");

    for (;;) {
        auto left_color = cam.GetStreamData(ImageType::IMAGE_LEFT_COLOR);
        if (left_color.img) {
            start = clock();
            // cv::Mat left = left_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
            // cv::imshow("left", left);
            auto left_img = left_color.img->To(ImageFormat::COLOR_BGR);
            cv::Mat frame(left_img->height(), left_img->width(), CV_8UC3,
                          left_img->data());
            // cv::imshow("video", frame);


//        Mat frame = imread("/home/hyx/git/darknet-AB/data/dog.jpg");
            // Mat图像转为yolo输入格式
            //    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

            shared_ptr<image_t> detImg = detector.mat_to_image_resize(frame);
            //前向预测
            vector<bbox_t> outs = detector.detect_resized(*detImg, frame.cols, frame.rows, 0.25);
            //画图
            Drawer(frame, outs, classes);
            ends = clock();
            double fps = 1.0/(ends-start);
            printf("%f\n", (ends-start));
            string text = "fps:"+to_string(int(fps));
            putText(frame, text, Point(10,10), FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, 4, false);
            imshow("video", frame);
        }
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q' || key == 'Q') break;
    }

    cam.Close();

    cv::destroyAllWindows();
    return 0;
}