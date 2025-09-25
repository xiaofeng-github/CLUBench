

import torch
import tqdm
import torch.distributed as dist


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform1, transform2=None):
        self.transform1 = transform1
        self.transform2 = transform1 if transform2 is None else transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

    def __str__(self):
        return f'transform1 {str(self.transform1)} transform2 {str(self.transform2)}'
class MemTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform1 if transform2 is None else transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform1(x), self.transform2(x)]

    def __str__(self):
        return f'transform1 {str(self.transform1)} transform2 {str(self.transform2)}'