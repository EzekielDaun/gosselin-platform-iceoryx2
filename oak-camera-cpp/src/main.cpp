#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>
#include <iceoryx2/v0.7.0/iox2/iceoryx2.hpp>

using namespace iox2;

struct ImageData
{
    uint8_t data[480 * 640 * 3];
    // uint8_t data[1];
};
inline auto operator<<(std::ostream &stream, const ImageData &value) -> std::ostream &
{
    stream << "Distance { distance_in_meters: " << value.data << " }";
    return stream;
}

int main()
{
    auto node = NodeBuilder()
                    .create<ServiceType::Ipc>()
                    .expect("");
    auto service = node.service_builder(ServiceName::create("/oak-camera/RGBD").expect(""))
                       .publish_subscribe<ImageData>()
                       .open_or_create()
                       .expect("");
    auto publisher = service.publisher_builder().create().expect("");

    // Create device
    std::shared_ptr<dai::Device> device = std::make_shared<dai::Device>();

    // Create pipeline
    dai::Pipeline pipeline(device);

    // Create nodes
    auto cam = pipeline.create<dai::node::Camera>()->build();
    auto videoQueue =
        cam->requestOutput(std::make_pair(640, 480))->createOutputQueue();

    // Start pipeline
    pipeline.start();

    while (true)
    {
        auto videoIn = videoQueue->get<dai::ImgFrame>();
        if (videoIn == nullptr)
            continue;

        // auto sample = publisher.loan_uninit().expect("");
        // auto &payload = sample.payload_mut();
        // std::memcpy(payload.data, mat.data, mat.total() * mat.elemSize());
        // auto initialized_sample = assume_init(std::move(sample));
        // send(std::move(initialized_sample)).expect("");

        auto frame = videoIn->getCvFrame();
        auto sample = publisher.loan_uninit().expect("");
        auto &payload = sample.payload_mut();
        std::memcpy(payload.data, frame.data, frame.total() * frame.elemSize());
        auto initialized_sample = assume_init(std::move(sample));
        send(std::move(initialized_sample)).expect("");

        // std::cout << videoIn->getCvFrame().total() << std::endl;
        // std::cout << videoIn->getCvFrame().elemSize() << std::endl;
        // std::cout << videoIn->getCvFrame().data << std::endl;

        // cv::Mat output;
        // auto frame = videoIn->getFrame();
        // cv::cvtColor(frame, output, cv::ColorConversionCodes::COLOR_YUV2BGR_IYUV);
        // cv::imshow("video", output);

        // auto foo = videoIn->getType();
        // std::cout << int(foo) << std::endl;

        // // cv::imshow("video", videoIn->getCvFrame());
        // if (cv::waitKey(1) == 'q')
        // {
        //     break;
        // }
    }

    return 0;
}
