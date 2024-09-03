###################
# Python script to visualize a scene (of sorted images)
##################
import cv2
import argparse
from pathlib import Path

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Directory containing images')
    parser.add_argument('-f', '--frame_rate', type=int, default=10, help='Frame rate(frames per second)')
    return parser.parse_args() 

def read_one_image(dir:Path):
    return cv2.imread(str(dir))

def read_images(directory:str) -> list:
    image_path = sorted(Path(directory).rglob('*.png'))
    im_list = map(read_one_image, image_path)
    return im_list

def visualize(images:list, fps:int):
    cv2.namedWindow('Scene', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Scene', 1200, 1200)
    for image in images:
        cv2.imshow('Scene', image)
        cv2.waitKey(100//fps) # can't take fps as float hence performed integer division
    cv2.destroyAllWindows()


def main():
    args = arg_parse()
    images=read_images(args.directory)
    visualize(images, args.frame_rate)

if __name__=="__main__":
    main()
