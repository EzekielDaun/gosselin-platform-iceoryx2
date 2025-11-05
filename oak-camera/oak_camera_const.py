import ctypes

RESOLUTION = (640, 480)

CTYPE_RGB_UINT8_FLAT_ARRAY = ctypes.c_uint8 * (RESOLUTION[0] * RESOLUTION[1] * 3)
CTYPE_DEPTH_UINT16_FLAT_ARRAY = ctypes.c_uint16 * (RESOLUTION[0] * RESOLUTION[1])


class OAKCameraData(ctypes.Structure):
    _fields_ = [
        (
            "rgb_data_uint8_flat_array",
            CTYPE_RGB_UINT8_FLAT_ARRAY,
        ),
        (
            "depth_data_uint16_flat_array",
            CTYPE_DEPTH_UINT16_FLAT_ARRAY,
        ),
    ]

    @staticmethod
    def type_name() -> str:
        return "OAKCameraData"
