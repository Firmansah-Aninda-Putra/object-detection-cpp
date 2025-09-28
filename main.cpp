#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

std::queue<cv::Mat> frameQueue;
std::mutex queueMutex;
std::condition_variable queueCV;
bool stopFlag = false;

void captureThread(const std::string& rtspUrl) {
    cv::VideoCapture cap(rtspUrl);
    if (!cap.isOpened()) {
        std::cerr << "Error: Tidak dapat membuka stream RTSP: " << rtspUrl << std::endl;
        stopFlag = true;
        return;
    } else {
        std::cout << "Berhasil membuka stream RTSP: " << rtspUrl << std::endl;
    }
    cv::Mat frame;
    while (!stopFlag) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Frame kosong... Mencoba lagi" << std::endl;
            continue;
        }

        // Kurangi resolusi frame untuk mengurangi lag
        cv::Mat resizedFrame;
        // Misalnya, resize ke 640x480. Sesuaikan ukuran jika perlu.
        cv::resize(frame, resizedFrame, cv::Size(640, 480));

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(resizedFrame.clone());
        }
        queueCV.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    cap.release();
    std::cout << "Capture thread selesai." << std::endl;
}

void detectionThread(const std::string& modelConfig, const std::string& modelWeights) {
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Anda bisa menyesuaikan ukuran input blob juga, namun model YOLO biasanya memerlukan ukuran tertentu (misal, 416x416).
    const int inpWidth = 416;
    const int inpHeight = 416;
    const float confThreshold = 0.5;

    std::cout << "Mulai detection thread..." << std::endl;

    while (!stopFlag) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [] { return !frameQueue.empty() || stopFlag; });
            if (stopFlag) break;
            frame = frameQueue.front();
            frameQueue.pop();
        }
        if (frame.empty()) continue;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outs;
        std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
        net.forward(outs, outNames);

        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = reinterpret_cast<float*>(outs[i].data);
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                float confidence = data[4];
                if (confidence > confThreshold) {
                    int centerX = static_cast<int>(data[0] * frame.cols);
                    int centerY = static_cast<int>(data[1] * frame.rows);
                    int width   = static_cast<int>(data[2] * frame.cols);
                    int height  = static_cast<int>(data[3] * frame.rows);
                    int left    = centerX - width / 2;
                    int top     = centerY - height / 2;
                    cv::rectangle(frame, cv::Point(left, top),
                                  cv::Point(left + width, top + height),
                                  cv::Scalar(0, 255, 0), 2);
                    std::string label = cv::format("%.2f", confidence);
                    cv::putText(frame, label, cv::Point(left, top - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        cv::imshow("Object Detection", frame);
        if (cv::waitKey(1) == 27) { // Tekan ESC untuk keluar
            std::cout << "ESC terdeteksi, menghentikan program..." << std::endl;
            stopFlag = true;
        }
    }
    std::cout << "Detection thread selesai." << std::endl;
}

int main() {
    std::cout << "Memulai program...\n";
    // Pastikan path model sesuai dengan lokasi file di C:\Users\bumii\Desktop\yolodnn
    std::string modelConfig = "C:/Users/bumii/Desktop/yolodnn/yolov4.cfg";
    std::string modelWeights = "C:/Users/bumii/Desktop/yolodnn/yolov4.weights";
    // RTSP URL yang digunakan
    std::string rtspUrl = "rtsp://admin:dishub2024@10.10.77.130:554/cam/realmonitor?chh";

    std::thread capThread(captureThread, rtspUrl);
    std::thread infThread(detectionThread, modelConfig, modelWeights);

    capThread.join();
    infThread.join();

    std::cout << "Program selesai. Tekan ENTER untuk keluar...\n";
    std::cin.get();
    return 0;
}
