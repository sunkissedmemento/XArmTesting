import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe.start(cfg)

try:
    while True:
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_cm = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.5),
            cv2.COLORMAP_JET
        )

        h, w, _ = color_image.shape
        cx = w // 2
        cy = h // 2

        # Get distance at center point in meters
        distance = depth_frame.get_distance(cx, cy)

        # Draw center indicators on RGB image
        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(color_image, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(color_image, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

        # Draw center indicators on depth image
        cv2.circle(depth_cm, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(depth_cm, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(depth_cm, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)

        # Text indicators
        rgb_text_1 = f"Center: ({cx}, {cy})"
        rgb_text_2 = f"Distance: {distance:.3f} m"
        rgb_text_3 = "Press Q to Quit"

        depth_text_1 = f"Depth at Center: {distance:.3f} m"
        depth_text_2 = "Depth View"

        cv2.putText(color_image, rgb_text_1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(color_image, rgb_text_2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(color_image, rgb_text_3, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(depth_cm, depth_text_1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_cm, depth_text_2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("rgb", color_image)
        cv2.imshow("depth", depth_cm)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()