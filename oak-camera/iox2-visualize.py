import time

import cv2
import iceoryx2 as iox2
import numpy as np
from oak_camera_const import (
    RESOLUTION,
    OAKCameraData,
)

if __name__ == "__main__":
    iox2_node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore

    oak_cam_subscriber = (
        iox2_node.service_builder(iox2.ServiceName.new("oak-camera/rgbd"))  # type: ignore
        .publish_subscribe(OAKCameraData)
        .open_or_create()
        .subscriber_builder()
        .create()
    )

    while True:
        maybe_data = oak_cam_subscriber.receive()
        if maybe_data is None:
            time.sleep(0.001)
            continue

        data = maybe_data.payload().contents

        rgb_data = np.array(
            data.rgb_data_uint8_flat_array,
            # shape=(RESOLUTION[1], RESOLUTION[0], 3),
        ).reshape((RESOLUTION[1], RESOLUTION[0], 3))
        # TODO: convert RGB to BGR for OpenCV display
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)


        # depth_data = np.ctypeslib.as_array(
        #     data.depth_data_uint16_flat_array,
        #     shape=(RESOLUTION[1], RESOLUTION[0]),
        # )

        cv2.imshow("OAK Camera RGB", rgb_data)
        if cv2.waitKey(1) == ord("q"):
            break
