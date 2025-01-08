# Training script for PointRend combineds Panels
#import packages
import os
import cv2
import json
import pickle
import random
import numpy as np

import shutil
import time

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from shapely.geometry import Polygon

# import PointRend project
# from detectron2.projects import point_rend


#for lr_scheduler
from fvcore.common.param_scheduler import CosineParamScheduler
from detectron2.solver.lr_scheduler import LRMultiplier, WarmupParamScheduler
from detectron2.solver import build_lr_scheduler

#eval hook
from hooks import LossEvalHook
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_test_loader

NUM_GPUS = 1
# List of  Panels:
train_classes = ['fake_lp' ,'real_lp_no_hsrp' ,	'real_lp' ,'hsrp' ,'fake_lp_ioi' ]

label_map = {"fake_lp":0,"fake_lp_ioi":1,'real_lp_no_hsrp':2,'real_lp':2,'hsrp':3}
class_list = ['fake_lp','fake_lp_ioi','real_lp','hsrp']

def countdown(t):
    columns = shutil.get_terminal_size().columns
    while t:
        mins, secs = divmod(t, 60)
        hrs, mins = divmod(mins,60)
        timer = '{:02d}:{:02d}:{:02d}'.format(hrs,mins, secs)
        print(f"Starting training in : {timer}".center(columns), end="\r")
        time.sleep(1)
        t -= 1

def get_panels_dicts(img_dir):
	if os.path.exists(os.path.join(img_dir,'preloaded_annotations.pkl')):
		with open(os.path.join(img_dir,'preloaded_annotations.pkl'),'rb') as f:
			dataset_dicts = pickle.load(f)
		print("Pre Saved annotations loaded and returned")
		return dataset_dicts
	
	dataset_dicts = []  # final list of dictionaries (one dict will have information of one image)
	
	# reading inside mutiple folders of on "train" and "val" folder
	total_folders = len(os.listdir(img_dir))
	for find, inside_folder in enumerate(os.listdir(img_dir)):
		json_file = os.path.join(img_dir, inside_folder, "via_region_data.json")
		
		if not os.path.exists(json_file): 
			continue
		
		with open(json_file) as f:
			imgs_anns = json.load(f)
			if type(imgs_anns) == str:
				imgs_anns = eval(imgs_anns)  #if jsons are created from any code changes

		for idx, v in enumerate(imgs_anns.values()):
			record = {}
			filename = os.path.join(img_dir, inside_folder, v["filename"])
			filename = filename.replace('https://cq-workflow.s3.ap-south-1.amazonaws.com/', '')
			filename = filename.replace('https://cq-workflow.s3.amazonaws.com','')
			print("Folder: ",find+1,'/',total_folders,"Image: ",idx,'/',len(imgs_anns),' : ',filename)
			
			try:
				cv_height, cv_width = cv2.imread(filename).shape[:2]
			except Exception as e:
				print("Image: ", filename, str(e))
				continue
 
			pil_width, pil_height = Image.open(filename).size
			if pil_width == cv_width and pil_height == cv_height:
				record["file_name"] = filename
				record["image_id"] = idx
				record['height'] = cv_height
				record['width'] = cv_width
				annos = v["regions"]
				objs = []

				for anno in annos:
					#Check if "region attributes" is present or not
					if "region_attributes" in anno.keys():
						cate = anno["region_attributes"]['identity']

						if cate in train_classes:
							#Check if "shape attributes" is present or not
							if "shape_attributes" in anno.keys():
								anno = anno["shape_attributes"]

								if "all_points_x" in anno.keys() and "all_points_y" in anno.keys():
									px = anno["all_points_x"]
									py = anno["all_points_y"]

									if len(px) == len(py):
										poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
										poly = [p for x in poly for p in x]

										if len(poly) >= 6:
											#Now checking if it can form a polygon or not
											points = []
											for i in range(0, len(px)):
												points.append([px[i], py[i]])
						
											anno_poly = Polygon(points)
											if anno_poly.is_valid:     #To check if this is a valid polygon
												obj = {
													"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
													"bbox_mode": BoxMode.XYXY_ABS,
													"segmentation": [poly],
													"category_id": label_map[cate],
													"iscrowd": 0
												}
												objs.append(obj)

				if objs != []:
					record["annotations"] = objs
					dataset_dicts.append(record)
	
	with open(os.path.join(img_dir, 'preloaded_annotations.pkl'), 'wb') as f:
		pickle.dump(dataset_dicts,f)
		print("annotations saved")

	return dataset_dicts

class Trainer(DefaultTrainer):
	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
		return COCOEvaluator("msil_lp_val", cfg, False, output_dir = './msil_training_dir/exp2')
	
	@classmethod
	def build_lr_scheduler(cls,cfg,optimizer):
		name = cfg.SOLVER.LR_SCHEDULER_NAME
		if name not in ['Cosine','Exponential']:
			return build_lr_scheduler(cfg,optimizer)
		else:
			if name == 'Cosine':
				sched = CosineParamScheduler(1,0.000001)
			elif name == 'Exponential':
				sched = ExponentialParamScheduler(1,0.002)
			sched = WarmupParamScheduler(sched,
			cfg.SOLVER.WARMUP_FACTOR,
			min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
			cfg.SOLVER.WARMUP_METHOD,
			)
			return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)

	def build_hooks(self):
		hooks = super().build_hooks()
		hooks.insert(-1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD,self.model,build_detection_test_loader(self.cfg,self.cfg.DATASETS.TEST[0],DatasetMapper(self.cfg,True))))
		return hooks


# registering dataset for training process
print("Dataset Registering....")
for d in ["train", "val"]:
	DatasetCatalog.register("msil_lp_" + d, lambda d=d: get_panels_dicts("msil_lp/" + d))
	MetadataCatalog.get("msil_lp_" + d).set(thing_classes=class_list)


# adjusting the training configurations
def setup():
	cfg = get_cfg()
	cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
	cfg.DATASETS.TRAIN = ("msil_lp_train",)
	cfg.DATASETS.TEST = ("msil_lp_val",)
	cfg.DATALOADER.NUM_WORKERS = 2

	cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
	cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 1.33, 1.5, 2.0]]
	cfg.MODEL.WEIGHTS = 'msil_training_dir/exp1/model_final.pth'
	#Let training initialize from pre-trained
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)
	
	cfg.INPUT.RANDOM_FLIP = "none"
	cfg.INPUT.MIN_SIZE_TRAIN = (512,)
	cfg.INPUT.MAX_SIZE_TRAIN = 800
	cfg.INPUT.MIN_SIZE_TEST = 512
	cfg.INPUT.MAX_SIZE_TEST = 800
	cfg.SOLVER.BASE_LR = 0.00125/2
	cfg.SOLVER.CHECKPOINT_PERIOD = 20000
	cfg.SOLVER.LR_SCHEDULER_NAME = 'Cosine'
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.MAX_ITER = 200000
	cfg.SOLVER.WARMUP_ITERS = 2000
	cfg.TEST.EVAL_PERIOD = 20000

	cfg.OUTPUT_DIR = './msil_training_dir/exp2'

	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	print(cfg, flush=True)

	return cfg

# Defining "trainer" object to start the training process
def main():
	# countdown(18000)
	cfg = setup()
	# print(cfg)
	# exit()
	trainer = Trainer(cfg)
	
	# "True" to resume training from previous step else False for fresh training
	trainer.resume_or_load(resume=False)
	print("Training Started...", flush=True)
	trainer.train()

	print("Training is done....", flush=True)

if __name__ == "__main__":
	launch(main, num_gpus_per_machine=NUM_GPUS, dist_url="auto")
