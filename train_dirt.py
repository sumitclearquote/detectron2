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
exp_dir = './mahindra_dirt/exp1'



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

for d in ["train", "val"]:
    imgdir = f"datasets/project_mahindra_dirt/{d}"
    DatasetCatalog.register("dirt_" + d, lambda d=d:get_dataset_dicts(imgdir))
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
	
	cfg.MODEL.WEIGHTS = 'R-101.pkl'
	#Let training initialize from pre-trained
	
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)
	
	cfg.INPUT.RANDOM_FLIP = "horizontal"
	# cfg.INPUT.MIN_SIZE_TRAIN = (512,) # default: [800, 850  ... ]
	# cfg.INPUT.MAX_SIZE_TRAIN = 800    # default: [1333, ... ]
	# cfg.INPUT.MIN_SIZE_TEST = 512
	# cfg.INPUT.MAX_SIZE_TEST = 800
	cfg.SOLVER.BASE_LR = 0.00125
	cfg.SOLVER.CHECKPOINT_PERIOD = 540#2700 # =========================== Change this
	cfg.SOLVER.LR_SCHEDULER_NAME = 'Cosine'
	cfg.SOLVER.IMS_PER_BATCH = 16 # =========================== Change this
	cfg.SOLVER.MAX_ITER = 540#27000 #===================================== 50 epochs=27000 for 50 epochs with 16 Batch size
	cfg.SOLVER.WARMUP_ITERS = 10 #2700
	cfg.TEST.EVAL_PERIOD = 540 #2700 # =========================== same as 'cfg.SOLVER.CHECKPOINT_PERIOD'

	cfg.OUTPUT_DIR = exp_dir #./mahindra_dirt/exp1

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
	


# directory structure:
# detectron2
#           train_dirt.py
#           datasets
#                 project_mahindra_dirt
#                         train
#                         val
#                         train.json #coco
#                         val.json #coco


