import os

import tqdm

from PIL import Image
from torchvision import transforms as TR

in_folder = "img_align_celeba"
out_folder = "img_squared128_celeba"

t = TR.Compose([TR.CenterCrop(150),
                TR.Resize((128, 128), Image.BICUBIC)])

for filename in tqdm.tqdm(os.listdir(in_folder)):
    img = Image.open(os.path.join(in_folder, filename))
    img = t(img)
    img.save(os.path.join(out_folder, filename))

