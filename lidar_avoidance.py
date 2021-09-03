#!/usr/bin/env python
# BEGIN ALL
import rospy
import cv2
import cv_bridge
import numpy
import math
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

global perr, ptime, serr, dt, move, ray_angle
perr = 0
ptime = 0
serr = 0
dt = 0
move = False
angle_step_deg = 20


class Follower:
	def __init__(self):
		self.bridge = cv_bridge.CvBridge()
		self.image_sub = rospy.Subscriber('/usb_cam/image_raw',	Image, self.image_callback)
		self.lidar_sub = rospy.Subscriber('/scan_raw', LaserScan, self.lidar_callback)
		self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
		self.image_pub = rospy.Publisher('/lane_image', Image, queue_size=1)
		self.twist = Twist()
		self.ray_angle = [x for x in range(angle_step_deg, 180, angle_step_deg)]
		self.dists = None

		self.cmd_vel_pub.publish(self.twist)

	def lidar_callback(self, msg):
		# get lidar distance at ray_angle in degree
		# dynamic offset
		# angles = [(x - 90) % 360 for x in self.ray_angle]
		# self.dists = [msg.ranges[x*2] for x in angles]
		# self.dists = list(map(lambda x: 0.1 if x == float('inf') else x, self.dists))
		# self.dists = list(map(lambda x: 0.5 if x >= 0.5 else x, self.dists))

		# static offset
		angles = [x for x in range(-10, -90, -5)]
		self.dists = [msg.ranges[x*2] for x in angles]

	def get_obstacle_threshold(self):
		if self.dists == None:
			return 0

		# dynamic offset
		# lateral_dists = [dist * numpy.cos(numpy.deg2rad(theta)) for dist, theta in zip(self.dists, self.ray_angle)]

		# static offset
		lateral_count = 0
		for d in self.dists:
			if d < 0.5:
				lateral_count += 1
		if lateral_count >= 1:
			print("lateral_cnt :{}".format(lateral_count))
			return 120
		else:
			return 0

		# dynamic offset
		# return sum(lateral_dists)

	def image_callback(self, msg):
		global perr, ptime, serr, dt
		image0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

		# transformation
		img = cv2.resize(image0, None, fx=0.6, fy=0.6,
						 interpolation=cv2.INTER_CUBIC)
		#print img.shape
		rows, cols, ch = img.shape
		pts1 = numpy.float32([[30, 80], [20, 130], [160, 80], [170, 130]])
		pts2 = numpy.float32([[0, 0], [0, 300], [300, 0], [300, 300]])

		M = cv2.getPerspectiveTransform(pts1, pts2)
		img_size = (img.shape[1], img.shape[0])
		image = cv2.warpPerspective(img, M, (300, 300))  # img_size

		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		lower_yellow = numpy.array([26,  43,  46])
		upper_yellow = numpy.array([34, 255, 255])
		lower_yellow = numpy.array([22, 0, 200])
		upper_yellow = numpy.array([38, 255, 255])

		# lower_white = numpy.array([0,0,221])
		# upper_white = numpy.array([180,30,225])
		# lower_white = numpy.array([0,0,210])
		# upper_white = numpy.array([200,255,255])
		sensitivity = 30
		lower_white = numpy.array([0, 0, 255-sensitivity])
		upper_white = numpy.array([255, sensitivity, 255])

		# lower_black = numpy.array([0,0,4])
		# upper_black = numpy.array([180,255,140])
		lower_black = numpy.array([0, 0, 10])
		upper_black = numpy.array([180, 255, 100])

		# threshold to get only white
		maskw = cv2.inRange(hsv, lower_white, upper_white)
		maskb = cv2.inRange(hsv, lower_black, upper_black)
		# maskb = cv2.inRange(gray, 0, 120)

		masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
		# remove pixels not in this range
		maskyw = cv2.bitwise_or(maskw, masky)
		maskyb = cv2.bitwise_or(maskb, masky)

		#rgb_yb = cv2.bitwise_and(image, image, mask = mask_yb).astype(numpy.uinet8)
		# rgb_yb = cv2.bitwise_and(image, image, mask = maskb).astype(numpy.uint8)
		rgb_yb = cv2.bitwise_and(image, image, mask=maskw).astype(numpy.uint8)
		rgb_yb = cv2.cvtColor(rgb_yb, cv2.COLOR_RGB2GRAY)

		# filter mask
		kernel = numpy.ones((7, 7), numpy.uint8)
		opening = cv2.morphologyEx(rgb_yb, cv2.MORPH_OPEN, kernel)
		rgb_yb2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		_, rgb_yb2 = cv2.threshold(rgb_yb2, 210, 255, cv2.THRESH_BINARY)

		# out_img = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

		# ROI
		out_img = rgb_yb2.copy()
		h, w = out_img.shape
		search_top = int(1*h/4+20)
		search_bot = int(3*h/4+20)
		search_mid = int(w/2)
		out_img[0:search_top, 0:w] = 0
		out_img[search_bot:h, 0:w] = 0
		#_, contour, _ = cv2.findContours(out_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		M = cv2.moments(out_img)
		c_time = rospy.Time.now()
		if M['m00'] > 0:
			cxm = int(M['m10']/M['m00'])
			cym = int(M['m01']/M['m00'])

			# cx = cxm - 110 #120#143 #CW
			#cx = cxm - 150
			offset = self.get_obstacle_threshold()
			#print("offset: ", offset)
			cx = cxm - offset

			cv2.circle(out_img, (cxm, cym), 20, (255, 0, 0), -1)
			cv2.circle(out_img, (cx, cym), 20, (255, 0, 0), 2)

			# BEGIN CONTROL
			err = cx - 4*w/8
		#   K_p = 0.63
			K_p = 0.5

		#   self.twist.linear.x = K_p
			dt = rospy.get_time() - ptime
		#   self.twist.angular.z = (-float(err) / 100)*2.5 + ((err - perr)/(rospy.get_time() - ptime))*1/50/100 #+ (serr*dt)*1/20/100 #1 is best, starting 3 unstable
		#   ang_z = err*0.0028
			# + (serr*dt)*1/20/100 #1 is best, starting 3 unstable
			ang_z = (float(err) / 100)*(0.25) + \
				((err - perr)/(rospy.get_time() - ptime))*1/20/100
			ang_z = min(0.8, max(-0.8, ang_z))
		# 0.143
			lin_x = ang_z
			if lin_x < 0:
				lin_x = -(lin_x)
			lin_x = K_p * (1-lin_x)

			#self.twist.linear.x = lin_x
			self.twist.linear.x = 0.2
			self.twist.angular.z = -ang_z

		#   print(cx, err*0.02, ang_z)
			# print("cx:{}, err:{:.4f}, ang_z:{:4f}, lin_x:{:4f}".format(
			#	cx, err*0.0015, ang_z, lin_x))
			serr = err + serr
			perr = err
			ptime = rospy.get_time()

		else:
			self.twist.linear.x = 0.1
			self.twist.angular.z = -0.3
			err = 0

		self.cmd_vel_pub.publish(self.twist)
		output_img = self.bridge.cv2_to_imgmsg(out_img)
		self.image_pub.publish(output_img)

		# END CONTROL

		#cv2.imshow("win3", rgb_yw2)
		# cv2.waitKey(3)


rospy.init_node('follower')
follower = Follower()
rospy.spin()
# END ALL
