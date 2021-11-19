from PIL import Image
import os
#path to high resolution data
data_gt_path="testsets/BSDS100/original/"
#path to low resolution data
data_lq_path="testsets/BSDS100/"
downscale_ratio=2
data_lq_path=data_lq_path+"X"+str(downscale_ratio)+"/"
data=os.listdir(data_gt_path)
if(not os.path.isdir(data_lq_path)):
    os.mkdir(data_lq_path)
for img_file in data:
    img=Image.open(data_gt_path+img_file)
    w,h=img.size
    img=img.resize((int(w/downscale_ratio),int(h/downscale_ratio)),Image.BICUBIC)
    img.save(data_lq_path+img_file)
    
