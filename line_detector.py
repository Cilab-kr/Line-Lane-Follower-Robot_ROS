#!/usr/bin/env python
# BEGIN ALL
import rospy
import cv2
import cv_bridge
import numpy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cilab_line_follower.msg import pos

global perr, ptime, serr, dt, move
perr = 0
ptime = 0
serr = 0
dt = 0
move = False


class Follower:
	def __init__(self):
		self.bridge = cv_bridge.CvBridge()
		self.image_sub = rospy.Subscriber('/usb_cam/image_raw',
										  Image, self.image_callback)
		self.line_pub = rospy.Publisher('/line_position', pos, queue_size=1)
		self.image_pub = rospy.Publisher('/lane_image', Image, queue_size=1)
		self.position = pos()

		self.line_pub.publish(self.position)

	def image_callback(self, msg):
		global perr, ptime, serr, dt
		image0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

		# transformation
		img = cv2.resize(image0, None, fx=0.6, fy=0.6,
						 interpolation=cv2.INTER_CUBIC)
		#print img.shape
		rows, cols, ch = img.shape
		pts1 = numpy.float32([[150, 150], [115, 200], [288, 150], [335, 200]])
		pts2 = numpy.float32([[50, 50], [50, 450], [450, 50], [450, 450]])

		M = cv2.getPerspectiveTransform(pts1, pts2)
		img_size = (img.shape[1], img.shape[0])
		image = cv2.warpPerspective(img, M, (500, 500))  # img_size

		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		lower_yellow = numpy.array([26,  43,  46])
		upper_yellow = numpy.array([34, 255, 255])

		lower_white = numpy.array([0, 0, 210])
		upper_white = numpy.array([200, 255, 255])
		# sensitivity = 31
		# lower_white = numpy.array([0,0,255-sensitivity])
		# upper_white = numpy.array([255,sensitivity,255])

		lower_black = numpy.array([0, 0, 10])
		upper_black = numpy.array([180, 255, 100])

		# threshold to get only white
		maskw = cv2.inRange(hsv, lower_white, upper_white)
		maskb = cv2.inRange(hsv, lower_black, upper_black)
		# maskb = cv2.inRange(gray, 0, 120)  # masking with gray threshold
		masky = cv2.inRange(hsv, lower_yellow, upper_yellow)

		# remove pixels not in this range
		mask_yw = cv2.bitwise_or(maskw, masky)
		mask_yb = cv2.bitwise_or(maskb, masky)

		rgb_yb = cv2.bitwise_and(image, image, mask=maskw).astype(numpy.uint8)
		rgb_yb = cv2.cvtColor(rgb_yb, cv2.COLOR_RGB2GRAY)

		# filter mask
		kernel = numpy.ones((7, 7), numpy.uint8)
		opening = cv2.morphologyEx(rgb_yb, cv2.MORPH_OPEN, kernel)
		rgb_yb2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

		# ROI
		out_img = rgb_yb2.copy()
		h, w = out_img.shape
		search_top = int(1*h/4+20)
		search_bot = int(4*h/4+20)
		search_mid = int(w/2)
		out_img[0:search_top, 0:w] = 0
		out_img[search_bot:h, 0:w] = 0
		M = cv2.moments(out_img)
		c_time = rospy.Time.now()
		if M['m00'] > 0:
			cxm = int(M['m10']/M['m00'])
			cym = int(M['m01']/M['m00'])

			cx = cxm - 200
			self.position = cxm

			cv2.circle(out_img, (cxm, cym), 20, (255, 0, 0), -1)
			cv2.circle(out_img, (cx, cym), 20, (255, 0, 0), 2)


		else:
			self.position = 1*w/2

		self.line_pub.publish(self.position)
		output_img = self.bridge.cv2_to_imgmsg(out_img)
		self.image_pub.publish(output_img)

rospy.init_node('line_lane_detector', anonymous=True)
follower = Follower()
rospy.spin()
