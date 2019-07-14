import numpy as np
import cv2
import os
from PIL import Image
from torchvision.datasets import VisionDataset


class CocoDetection(VisionDataset):
    """
    COCO dataset for M2Det

    """
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        new_target = []
        for t in target:
            b = t['bbox']
            c = t['category_id']
            new_target.append(np.array([float(b[0]), float(b[1]), float(b[0]) + float(b[2]), float(b[1]) + float(b[3]), c]))

        new_target = np.array(new_target)

        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
        if self.transforms is not None:
            img, new_target = self.transforms(img, new_target)

        return img, new_target

    def __len__(self):
        return len(self.ids)
