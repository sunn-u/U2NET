# Coding by SunWoo(tjsntjsn20@gmail.com)

import cv2


# for DUTS dataset : http://saliencydetection.net/duts/
class BasicDataset:
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, mask_dir = self.samples[idx]

        img = cv2.imread(img_dir)
        mask = cv2.cvtColor(cv2.imread(mask_dir), cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
