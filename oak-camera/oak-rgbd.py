import time

import cv2
import depthai as dai
import iceoryx2 as iox2
import numpy as np
from oak_camera_const import (
    CTYPE_DEPTH_UINT16_FLAT_ARRAY,
    CTYPE_RGB_UINT8_FLAT_ARRAY,
    RESOLUTION,
    OAKCameraData,
)

if __name__ == "__main__":
    # Create pipeline
    with dai.Pipeline() as p:
        left_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        color_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        stereo = p.create(dai.node.StereoDepth).build(
            left=left_cam.requestOutput(RESOLUTION),
            right=right_cam.requestOutput(RESOLUTION),
        )
        rgbd = p.create(dai.node.RGBD).build()

        # stereo.setRectifyEdgeFillColor(0)
        # stereo.enableDistortionCorrection(True)
        # stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        # stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 10000
        # rgbd.setDepthUnits(dai.StereoDepthConfig.AlgorithmControl.DepthUnit.METER)

        color_out = color_cam.requestOutput(
            RESOLUTION, dai.ImgFrame.Type.RGB888i, dai.ImgResizeMode.CROP, fps=30
        )
        color_out.link(stereo.inputAlignTo)
        color_out.link(rgbd.inColor)

        stereo.depth.link(rgbd.inDepth)

        rgbd_queue = rgbd.rgbd.createOutputQueue()

        p.start()

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore
        oak_cam_publisher = (
            node.service_builder(iox2.ServiceName.new("oak-camera/rgbd"))  # type: ignore
            .publish_subscribe(OAKCameraData)
            .open_or_create()
            .publisher_builder()
            .create()
        )

        while p.isRunning():
            rgbd_data_list: list[dai.RGBDData] = rgbd_queue.getAll()  # type: ignore
            for rgbd_data in rgbd_data_list:
                print(time.perf_counter())
                rgb_image_frame = rgbd_data.getRGBFrame()
                # print(rgb_image_frame.getType())
                rgb_888i_mat = rgb_image_frame.getFrame()
                # print(rgb_888i_mat.shape)

                depth_image_frame = rgbd_data.getDepthFrame()
                # print(depth_image_frame.getType())
                depth_raw16_mat = depth_image_frame.getFrame()
                # print(depth_raw16_mat.shape)

                foo = rgbd_data.getData()
                print(foo)
                print(foo.nbytes)

                # oak_cam_publisher.loan_uninit().write_payload(
                #     OAKCameraData(
                #         CTYPE_RGB_UINT8_FLAT_ARRAY(
                #             *np.array(rgb_888i_mat, dtype=np.uint8).flatten()
                #         ),
                #         CTYPE_DEPTH_UINT16_FLAT_ARRAY(
                #             *np.array(depth_raw16_mat, dtype=np.uint16).flatten()
                #         ),
                #     )
                # ).send()

                # iox2_data = OAKCameraData(
                #     CTYPE_RGB_UINT8_FLAT_ARRAY(*rgb_888i_mat.flatten().data),
                #     CTYPE_DEPTH_UINT16_FLAT_ARRAY(*depth_raw16_mat.flatten().data),
                # )
                # oak_cam_publisher.loan_uninit().write_payload(iox2_data).send()

                # cv2.imshow("OAK Camera RGB", rgb_888i_mat)
                # if cv2.waitKey(1) == ord("q"):
                #     p.stop()
                #     break
