import pyrealsense2 as rs

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
pipeline.start(cfg)  # if this works, your camera is ready
