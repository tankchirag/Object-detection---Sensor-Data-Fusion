#######
# Parser functions to get: 
# YOLO Model output result format 
# JSON file format  
# YOLO train input format (from JSON format)
#######

import os , sys 
import argparse 
from pathlib import Path 
import json 
from collections import defaultdict
import numpy as np


def coco2yolo(src:str, dst:str):
    """
    Converts COCO format to YOLO format 
    Saves in .txt file 
    Works by taking a single image file 
    """
    src = Path(src)
    fn = Path(dst)
    fn.mkdir(exist_ok=True) 
    with open(src) as fin:
        data = json.load(fin)
    # Get the Image id and Name 
    # TODO: Solve conflict for different scenes but same image id 
    images = {"%g" % data["image"]["id"] : data["image"]}
    img2ann = defaultdict(list) # complex
    for ann in data["annotations"]:
        img2ann[ann["image_id"]].append(ann)
    
    # Official rep used: tqdm -> to visualize progress (TODO: Checkout)
    for img_id, anns in img2ann.items():
        img = images["%g" % img_id]
        h, w, f = img["height"], img["width"], img["file_name"]
        bboxes = [] 
        for ann in anns: 
            box = np.array(ann["bbox"], dtype=np.float64) 
            box[:2] += box[2:] / 2 # Finding centre co-ordinate
            box[[0,2]] /= w # 
            box[[1,3]] /= h
            if box[2] <= 0 or box[3] <= 0: continue  # (to ensure that width / height are not negative nums)
            
            cls = ann["category_id"] - 1 # This is to convert the COCO format (1 - n classes) to YOLO format (0 - n classes)
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
        
        output_file = (fn / f).with_suffix(".txt")
        if output_file.exists():
            output_file.unlink()

        # This writes the final data to a txt file (expected format for YOLO) 
        with open(output_file,"a") as file:
            for i in range(len(bboxes)): # Iterate for all the bounding boxes 
                line = (*(bboxes[i]),) # bboxes has a list of box's with {cls + list of bbox co-ordinate}
                file.write(("%g " * len(line)).rstrip() % line + "\n")
                

#coco2yolo(Path(src), dst)
list(map(lambda x:coco2yolo(x,dst), sorted(Path(src).rglob('*.json'))))