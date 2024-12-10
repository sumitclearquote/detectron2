from detectron2.data import transforms as T


from detectron2.data.transforms import Augmentation
from detectron2.data.transforms.augmentation import Transform
import numpy as np
import random

class ChannelShuffle(Augmentation):
    def __init__(self):
        super().__init__()

    def get_transform(self, image):
        # Ensure the input is in HWC format (height, width, channels)
        assert image.ndim == 3 and image.shape[2] == 3, "Image must be in HWC format with 3 channels"
        channels = [0, 1, 2]
        random.shuffle(channels)  # Shuffle channel order
        return ChannelShuffleTransform(channels)

class ChannelShuffleTransform(Transform):
    def __init__(self, channel_order):
        super().__init__()
        self.channel_order = channel_order

    def apply_image(self, img):
        return img[:, :, self.channel_order]

    def apply_coords(self, coords):
        return coords  # No change to coordinates
    

augmentations = [
    T.RandomBrightness(0.7, 1.3),
    T.RandomContrast(0.7, 1.3),
    ChannelShuffle(),  # Custom channel shuffle
    T.RandomFlip(horizontal=True),
]

random_apply_augmentations = T.RandomApply(augmentations, p = 0.5)


