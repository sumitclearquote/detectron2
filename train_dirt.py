import shutil
import time
import os
import pickle
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T

from augs import random_apply_augmentations

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
#train_classes = ['dirt', 'bird_dropping']

#label_map = {"dirt":0,"fake_lp_ioi":1,'real_lp_no_hsrp':2,'real_lp':2,'hsrp':3}

class_list = ['dirt']
exp_dir = './mahindra_dirt/exp3'



def get_dataset_dicts(img_dir):
    if os.path.exists(os.path.join(img_dir,'preloaded_annotations.pkl')):
        with open(os.path.join(img_dir,'preloaded_annotations.pkl'),'rb') as f:
            dataset_dicts = pickle.load(f)
        print(f"Pre Saved annotations loaded and returned for {img_dir.split('/')[-1]}")
        return dataset_dicts


class Trainer(DefaultTrainer):
	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
		return COCOEvaluator("dirt_val", cfg, False, output_dir = exp_dir)
	
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
		
	@classmethod
	def build_train_loader(cls, cfg): # For augmentations
		return build_detection_train_loader(
			cfg,
			mapper=DatasetMapper(cfg, is_train=True, augmentations=[random_apply_augmentations])
		)

	def build_hooks(self):
		hooks = super().build_hooks()
		hooks.insert(-1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD,self.model,build_detection_test_loader(self.cfg,self.cfg.DATASETS.TEST[0],DatasetMapper(self.cfg,True))))
		return hooks


# registering dataset for training process
print("Dataset Registering....")


# register_coco_instances("dirt_train", {}, "datasets/project_mahindra_dirt/dirt_train.json", "datasets/project_mahindra_dirt")
# register_coco_instances("dirt_val", {}, "datasets/project_mahindra_dirt/dirt_val.json", "datasets/project_mahindra_dirt")

# for d in ["train", "val"]:
#     my_dataset_train_metadata = MetadataCatalog.get(f"dirt_{d}").set(thing_classes = class_list)
#     dataset_dicts = DatasetCatalog.get(f"dirt_{d}")

for d in ["val", "train"]:
    DatasetCatalog.register("dirt_" + d, lambda d=d:get_dataset_dicts(f"datasets/project_mahindra_dirt/{d}"))
    metadata = MetadataCatalog.get("dirt_" + d).set(thing_classes=class_list)



# adjusting the training configurations
def setup():
	cfg = get_cfg()
	cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml") # ======== architecture?
	cfg.DATASETS.TRAIN = ("dirt_train",)
	cfg.DATASETS.TEST = ("dirt_val",)
	cfg.DATALOADER.NUM_WORKERS = 2

	cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
	cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 1.33, 1.5, 2.0]]
	
	cfg.MODEL.WEIGHTS = 'mahindra_dirt/exp2/model_0008099.pth'
	#Let training initialize from pre-trained
	
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)
	
	cfg.INPUT.RANDOM_FLIP = "horizontal"
	# cfg.INPUT.MIN_SIZE_TRAIN = (512,) # default: [800, 850  ... ]
	# cfg.INPUT.MAX_SIZE_TRAIN = 800    # default: [1333, ... ]
	# cfg.INPUT.MIN_SIZE_TEST = 512
	# cfg.INPUT.MAX_SIZE_TEST = 800

	cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False #Keeping this FALSE includes images with empty annotations in training.
	cfg.SOLVER.BASE_LR = 0.0000125 # divide 0.00125(default) by 2 since bsize was / 2 and further divide it by 4 since we're starting from previous best model
	cfg.SOLVER.CHECKPOINT_PERIOD = 500 #2000 # =========================== 
	cfg.SOLVER.LR_SCHEDULER_NAME = 'Cosine'
	cfg.SOLVER.IMS_PER_BATCH = 2 # =========================== 5 epochs
	cfg.SOLVER.MAX_ITER = 1000 #16000 #===================================== 50 epochs=27000 iters with 16 Batch size. 16000 iters = ~15 epoch
	cfg.SOLVER.WARMUP_ITERS = 50 #500
	cfg.TEST.EVAL_PERIOD = 500 #2000 # =========================== same as 'cfg.SOLVER.CHECKPOINT_PERIOD'

	cfg.OUTPUT_DIR = exp_dir #./mahindra_dirt/exp2

	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	print(cfg, flush=True)

	return cfg

# Defining "trainer" object to start the training process
def main():
	# countdown(18000)
	cfg = setup()
	# print(cfg)
	# exit()


	# Custom trainer for adding augmentations
	# class AugTrainer(DefaultTrainer):
	# 	@classmethod
	# 	def build_train_loader(cls, cfg):
	# 		return build_detection_train_loader(
	# 			cfg,
	# 			mapper=DatasetMapper(cfg, is_train=True, augmentations=[random_apply_augmentations])
	# 		)


	trainer = Trainer(cfg)
	#trainer = AugTrainer(cfg)
	
	# "True" to resume training from previous step else False for fresh training
	trainer.resume_or_load(resume = True)
	
	print("Training Started...", flush=True)
	trainer.train()

	print("Training is done....", flush=True)

if __name__ == "__main__":
	launch(main, num_gpus_per_machine=NUM_GPUS, dist_url="auto")
	


# directory structure:
# detectron2
#           train_dirt.py
#           datasets
#                 project_mahindra_dirt
#                         train
#                         val
#                         train.json #coco
#                         val.json #coco


