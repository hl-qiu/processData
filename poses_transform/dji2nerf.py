import argparse
import xml.etree.ElementTree as ET
import math
import cv2
import numpy as np
import os
import json

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def central_point(out):
	# # find a central point they are all looking at
	# print("computing center of attention...")
	# totw = 0.0
	# totp = np.array([0.0, 0.0, 0.0])
	# for f in out["frames"]:
	# 	mf = np.array(f["transform_matrix"])[0:3,:]
	# 	for g in out["frames"]:
	# 		mg = g["transform_matrix"][0:3,:]
	# 		p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
	# 		if w > 0.0001:
	# 			totp += p*w
	# 			totw += w
	# totp /= totw
	# print(totp) # the cameras are looking at totp
	for f in out["frames"]:
		# f["transform_matrix"][0:3,3] -= totp
		f["transform_matrix"] = f["transform_matrix"].tolist()
	return out

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	if image is None:
		print("Image not found:", imagePath)
		return 0
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	return fm

#END
###############################################################################

#Copyright (C) 2022, Enrico Philip Ahlers. All rights reserved.

def parse_args():
	parser = argparse.ArgumentParser(description="convert Agisoft XML export to nerf format transforms.json")
	parser.add_argument("--xml_in", default="/data/data/wangda-9.19/pos/shangyou.xml", help="specify xml file location")
	parser.add_argument("--out", default="/data/data/wangda-9.19/transforms.json", help="output path")
	parser.add_argument("--imgfolder", default="images", help="location of folder with images")
	parser.add_argument("--imgtype", default="jpg", help="type of images (ex. jpg, png, ...)")
	parser.add_argument("--scale", default=0.25,type=float, help="type of images (ex. jpg, png, ...)")
	parser.add_argument("--altitude", default=20,type=float, help="type of images (ex. jpg, png, ...)")
	args = parser.parse_args()
	return args



def get_calibration(root):
	for sensor in root[0][0]:
		for child in sensor:
			if child.tag == "calibration":
				return child
	print("No calibration found")	
	return None

def reflectZ():
	return [[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]]

def reflectY():
	return [[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]

def matrixMultiply(mat1, mat2):
	return np.array([[sum(a*b for a,b in zip(row, col)) for col in zip(*mat2)] for row in mat1])

def angle2matrix(w,p,k):

    rotMat = np.array([
        [np.cos(p)*np.cos(k),np.cos(w)*np.sin(k)+np.sin(w)*np.sin(p)*np.cos(k),np.sin(w)*np.sin(k)-np.cos(w)*np.sin(p)*np.cos(k)],
        [-np.cos(p)*np.sin(k), np.cos(w)*np.cos(k)-np.sin(w)*np.sin(p)*np.sin(k), np.sin(w)*np.cos(k)+np.cos(w)*np.sin(p)*np.sin(k)],
        [np.sin(p), -np.sin(w)*np.cos(p),  np.cos(w)*np.cos(p)]])

    return rotMat


if __name__ == "__main__":
	args = parse_args()
	XML_LOCATION = args.xml_in
	IMGTYPE = args.imgtype
	IMGFOLDER = args.imgfolder
	OUTPATH = args.out

	out = dict()
	aabb_scale = 128
	with open(XML_LOCATION, "r") as f:
		root = ET.parse(f).getroot()
		#print(root[0][0][0].tag)

		if len(root[1][3])==1 or True: # single camera
			w = float(root[1][3][0][3][0].text)
			h = float(root[1][3][0][3][1].text)
			focal_length = float(root[1][3][0][6].text)  # 物理焦距
			ccdw = float(root[1][3][0][7].text) #ccdw
			fl_x = focal_length / ccdw * w
			fl_y = fl_x
			cx=float(root[1][3][0][9][0].text)
			cy=float(root[1][3][0][9][1].text)
			camera_angle_x = math.atan(float(w) / (float(fl_x) * 2)) * 2
			camera_angle_y = math.atan(float(h) / (float(fl_y) * 2)) * 2
			out.update({"camera_angle_x": camera_angle_x})
			out.update({"camera_angle_y": camera_angle_y})
			out.update({"fl_x": fl_x})
			out.update({"fl_y": fl_y})
			out.update({"w": w})
			out.update({"h": h})
			out.update({"cx":cx})
			out.update({"cy": cy})	
			out.update({"aabb_scale": aabb_scale})
			frames = list()	
			len =1
			for i in range(len):
				for frame in root[1][3][i]:
					current_frame = dict()
					if frame.tag != 'Photo':
						continue
					child_count = sum(1 for _ in frame.iter())
					# print("子标签个数：", child_count)
					# print(len(frame))
					if child_count < 10 :
						continue
					path_img = frame[1].text
					# filename1 = path_img.split('/')[-2]
					filename = path_img.split('/')[-1]
					filename = filename.replace(".JPG", ".jpg")
					# if '(' in filename:
					# 	continue
					imagePath = os.path.join(IMGFOLDER,filename)
					current_frame.update({"file_path": imagePath})
					# current_frame.update({"sharpness":sharpness(imagePath)})
					# omega = float(frame[2][0][0].text)
					# phi = float(frame[2][0][1].text)
					# kappa = float(frame[2][0][2].text)
					# rotation_matrix = angle2matrix(omega*np.pi/180.0,phi*np.pi/180.0,kappa*np.pi/180.0)
					# print(frame[0].text)
					
					rotation_matrix = np.array([
						[float(frame[3][0][0].text),float(frame[3][0][1].text),float(frame[3][0][2].text)],
						[float(frame[3][0][3].text),float(frame[3][0][4].text),float(frame[3][0][5].text)],
						[float(frame[3][0][6].text),float(frame[3][0][7].text),float(frame[3][0][8].text)]
					])
					center = np.array([[float(frame[3][1][0].text)],[float(frame[3][1][1].text)],[float(frame[3][1][2].text)]])
					w2c = np.concatenate([np.concatenate([rotation_matrix,np.dot(rotation_matrix,-center)],axis=1),np.array([[0,0,0,1]])],axis=0)
					# c2w = np.linalg.inv(np.dot(np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]),w2c))
					c2w = np.linalg.inv(w2c)				
					current_frame.update({"transform_matrix":c2w} )
					
					frames.append(current_frame)

			
				

			print(frames)
			out.update({"frames": frames})
		else: # multi cameras
			out.update({"aabb_scale": aabb_scale})
			frames = list()
			for camera in root[1][3]:
				w = float(camera[1][0].text)*args.scale
				h = float(camera[1][1].text)*args.scale
				fl_x = float(camera[3].text)*args.scale
				fl_y = fl_x
				cx=float(camera[4][0].text)*args.scale
				cy=float(camera[4][1].text)*args.scale
				camera_angle_x = math.atan(float(w) / (float(fl_x) * 2)) * 2
				camera_angle_y = math.atan(float(h) / (float(fl_y) * 2)) * 2
				for frame in camera:
					if frame.tag != 'Photo':
						continue
					current_frame = dict()
					current_frame.update({"camera_angle_x": camera_angle_x})
					current_frame.update({"camera_angle_y": camera_angle_y})
					current_frame.update({"fl_x": fl_x})
					current_frame.update({"fl_y": fl_y})
					current_frame.update({"w": w})
					current_frame.update({"h": h})
					current_frame.update({"cx":cx})
					current_frame.update({"cy": cy})	

					path_img = frame[1].text
					imagePath = IMGFOLDER+path_img.split('\\')[-1][:-4]+'_IMGP.jpg'
					print(imagePath)
					current_frame.update({"file_path": imagePath})
					# current_frame.update({"sharpness":sharpness(imagePath)})
					omega = float(frame[2][0][0].text)
					phi = float(frame[2][0][1].text)
					kappa = float(frame[2][0][2].text)
					rotation_matrix = angle2matrix(omega*np.pi/180.0,phi*np.pi/180.0,kappa*np.pi/180.0)
					center = np.array([[float(frame[2][1][0].text)],[float(frame[2][1][1].text)],[float(frame[2][1][2].text)]]) 
					w2c = np.concatenate([np.concatenate([rotation_matrix,np.dot(rotation_matrix,-center)],axis=1),np.array([[0,0,0,1]])],axis=0)
					# c2w = np.linalg.inv(np.dot(np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]),w2c))
					c2w = np.linalg.inv(w2c)				
					current_frame.update({"transform_matrix":c2w} )
					
					frames.append(current_frame)
			out.update({"frames": frames})
		flip_mat = np.array([
			[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]
		])
		total=np.zeros(3)
		index = 0
		for f in out["frames"]:
			f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
			total += f["transform_matrix"][0:3,3]
			index += 1
		print("index == ",index)
		local_center = total/index
		local_center = np.concatenate([local_center[0:2],[local_center[2]-args.altitude]])
		# max_x = -1000
		# max_y = -1000
		# max_z = -1000
		# min_x = 1000
		# min_y = 1000
		# min_z = 1000
		for f in out["frames"]:
			f["transform_matrix"][0:3,3] = (f["transform_matrix"][0:3,3] - local_center)*0.5
		# 	max_x = max_x if f["transform_matrix"][0,3] < max_x else f["transform_matrix"][0,3]
		# 	max_y = max_y if f["transform_matrix"][1,3] < max_y else f["transform_matrix"][1,3]
		# 	max_z = max_z if f["transform_matrix"][2,3] < max_z else f["transform_matrix"][2,3]
		# 	min_x = min_x if f["transform_matrix"][0,3] > min_x else f["transform_matrix"][0,3]
		# 	min_y = min_y if f["transform_matrix"][1,3] > min_y else f["transform_matrix"][1,3]
		# 	min_z = min_z if f["transform_matrix"][2,3] > min_z else f["transform_matrix"][2,3]
		# print([max_x,max_y,max_z])
		# print([min_x,min_y,min_z])
	out = central_point(out)


	with open(OUTPATH, "w") as f:
		json.dump(out, f, indent=4)
