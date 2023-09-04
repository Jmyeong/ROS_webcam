#!/usr/bin/env python3


import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# 초기화
rospy.init_node('cam')
image_pub = rospy.Publisher('msg', Image, queue_size=10)
bridge = CvBridge()
camera = cv2.VideoCapture(0)  # 웹캠 사용을 위한 VideoCapture 객체, 0은 기본 카메라를 의미

while not rospy.is_shutdown():
    ret, frame = camera.read()  # 프레임 읽기

    if not ret:
        rospy.logwarn("Failed to grab frame")
        continue

    try:
        image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")  # OpenCV 이미지를 ROS 이미지 메시지로 변환
        image_pub.publish(image_msg)  # 이미지 메시지를 발행
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s", e)

    rospy.sleep(0.1)  # 매 0.1초마다 프레임 전송

camera.release()  # 카메라 릴리스
cv2.destroyAllWindows()  # OpenCV 창 닫기

