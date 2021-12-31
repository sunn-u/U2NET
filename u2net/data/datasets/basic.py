# Coding by SunWoo(tjsntjsn20@gmail.com)

from ..utils import load_image


class BasicDataset(object):
    def __init__(self, samples: list, type: str, transform=None):
        self.samples = samples
        self.type = type
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, mask_dir = self.samples[idx]

        img = load_image(img_dir, type=self.type)
        mask = load_image(mask_dir, type='GRAY')

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
