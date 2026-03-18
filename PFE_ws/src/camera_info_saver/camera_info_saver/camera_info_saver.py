#!/usr/bin/env python3
import os
import sys
import yaml
import rclpy
import argparse
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.srv import SetCameraInfo


def camera_info_to_dict(msg: CameraInfo) -> dict:
    """Convert a CameraInfo message to a yaml-serialisable dictionary."""
    return {
        "image_width":  msg.width,
        "image_height": msg.height,
        "camera_name":  "",
        "camera_matrix": {
            "rows": 3, "cols": 3,
            "data": list(msg.k),
        },
        "distortion_model": msg.distortion_model,
        "distortion_coefficients": {
            "rows": 1, "cols": len(msg.d),
            "data": list(msg.d),
        },
        "rectification_matrix": {
            "rows": 3, "cols": 3,
            "data": list(msg.r),
        },
        "projection_matrix": {
            "rows": 3, "cols": 4,
            "data": list(msg.p),
        },
    }


class CameraInfoSaverNode(Node):
    def __init__(self, camera_name: str, save_path: str, namespace: str):
        super().__init__("camera_info_saver")

        self.camera_name = camera_name
        self.save_path = save_path
        self.namespace = namespace.rstrip("/")

        self.publisher_ = self.create_publisher(
            CameraInfo,
            f"{self.namespace}/camera_info",
            10,
        )

        self.srv = self.create_service(
            SetCameraInfo,
            f"{self.namespace}/set_camera_info",
            self.handle_set_camera_info,
        )

        self.get_logger().info(
            f"[camera_info_saver] Serving set_camera_info on "
            f"'{self.namespace}/set_camera_info'"
        )
        self.get_logger().info(
            f"[camera_info_saver] Calibration will be saved to: {self.save_path}"
        )

    def handle_set_camera_info(
        self,
        request: SetCameraInfo.Request,
        response: SetCameraInfo.Response,
    ) -> SetCameraInfo.Response:

        info = request.camera_info
        self.get_logger().info(
            f"[camera_info_saver] Received calibration "
            f"({info.width}x{info.height}). Saving …"
        )

        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            data = camera_info_to_dict(info)
            data["camera_name"] = self.camera_name

            with open(self.save_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

            self.get_logger().info(
                f"[camera_info_saver] Saved → {self.save_path}"
            )
        except Exception as e:
            msg = f"Failed to save calibration: {e}"
            self.get_logger().error(f"[camera_info_saver] {msg}")
            response.success = False
            response.status_message = msg
            return response

        self.publisher_.publish(info)
        self.get_logger().info(
            f"[camera_info_saver] Published updated camera_info on "
            f"'{self.namespace}/camera_info'"
        )

        response.success = True
        response.status_message = f"Calibration saved to {self.save_path}"
        return response

def main():
    parser = argparse.ArgumentParser(
        description="ROS 2 set_camera_info service + calibration YAML saver"
    )
    parser.add_argument(
        "--camera-name",
        default="oakd_rgb",
        help="Camera name written into the YAML (default: oakd_rgb)",
    )
    parser.add_argument(
        "--save-path",
        default=os.path.expanduser("~/.ros/camera_info/oakd_rgb.yaml"),
        help="Full path where the calibration YAML will be written",
    )
    parser.add_argument(
        "--namespace",
        default="/oakd/rgb/preview",
        help="ROS namespace of the camera (default: /oakd/rgb/preview)",
    )

    argv = rclpy.utilities.remove_ros_args(sys.argv[1:])
    args = parser.parse_args(argv)

    rclpy.init()
    node = CameraInfoSaverNode(
        camera_name=args.camera_name,
        save_path=args.save_path,
        namespace=args.namespace,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()