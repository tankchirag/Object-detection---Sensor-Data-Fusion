# File to read json files of in a given directory
# Store the results of the bounding-box in parallel to the image data
# use this to plot the bounding box using open-cv
# --------------------- 
# TODO: Fix the multi-class multi-color scheme
# ----------------------


import os
import cv2
import json
import time 
import numpy
import argparse
from pathlib import Path

def arg_parse() -> argparse.Namespace:
    """
    Function to parse comman-line arguments of img_dir, json_dir and save_dir
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('image_directory', type=str, help='Directory containing images for first dataset')
    parser.add_argument('json_directory', type=str, help='Directory containing images for second dataset')
    parser.add_argument('save_directory', type=str, help='Directory containing images for third dataset')
    return parser.parse_args() 

def read_one_image(dir:Path) -> numpy.ndarray:
    """
    Reads an image using cv2 from it's POSIX Path
    """
    return cv2.imread(str(dir))

def read_one_json(dir:Path)->tuple:
    """
    Parse and read a single JSON file provided as Path input
    """
    with open(dir, 'r') as f:
        file = json.load(f)
        labels = []
        bboxes = [] 
        for c, b in zip(file['categories'], file['annotations']):
            labels.append(c['name'])
            bboxes.append(b['bbox'])
    return (labels, list(map(xywh2xyxy, bboxes)))

# Map xywh (from json) to xyxy
def xywh2xyxy(bbox) -> list:
    """
    Converts COCO co-ordinates (top-x-top-y-w-h) to 
    (top-x-top-y-bottom-x-bottom-y)
    """
    top_x, top_y, w, h = bbox
    return [(int(top_x), int(top_y)), (int(top_x + w), int(top_y + h))]

def draw_one_image(image_file:numpy.ndarray, json_file:tuple) -> numpy.ndarray:
    """
    Draws a single image containing all the info from json labesl passed for that image 
    Works
    """
    if image_file is None:
        print(f"Error: Unable to read image file: {image_file}")
        return None  
    new_img = image_file.copy() # creating a copy to protect original image
    for name, bbox in zip(json_file[0], json_file[1]):
        cv2.rectangle(new_img,bbox[0],bbox[1], (0,0,255), 3)
        cv2.putText(new_img, name,(bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255), 3)
    return new_img 

def save_images(img_files:str,json_files:str, dst:str) -> None:
    """Saves all the drawn images with particular name"""
    start = time.perf_counter()
    json_labels = list(map(read_one_json, sorted(Path(json_files).rglob('*.json')))) # List of tuples (labels, bounding boxes)
    imgs_path = list(sorted(Path(img_files).rglob('*.png'))) # List of POSIX image paths 
    imgs = list(map(read_one_image ,imgs_path)) # List of images in numpy.ndarray format
    imgs_with_labels = list(map(draw_one_image, imgs, json_labels)) # List of images with bounding boxes an labels
    for img_path, img_with_label in zip(imgs_path, imgs_with_labels): 
        save_path = os.path.join(dst, img_path.name)
        cv2.imwrite(save_path, img_with_label)
    end = time.perf_counter()
    print(f"Saved the images \ntime: {end - start} secs")

def main():
    args = arg_parse()
    save_images(args.image_directory, args.json_directory,args.save_directory) 

if __name__ == '__main__':
    main() 
