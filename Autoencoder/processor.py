import numpy as np
import glob
import os
import cv2
from keras.preprocessing.image import img_to_array, load_img
import argparse

stored_images = []

arg_parser = argparse.ArgumentParser(description='Specify the video folder path')
arg_parser.add_argument('videos_path', type=str)
arguments = arg_args.parse_args()

video_folder = arguments.videos_path
frame_rate = 10

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clear_images(path):
    files = glob.glob(os.path.join(path, "*.png"))
    for file in files:
        os.remove(file)

def save_image(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (227, 227))
    gray_image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    stored_images.append(gray_image)

video_files = os.listdir(video_folder)

make_directory(video_folder + '/frames')
clear_images(video_folder + '/frames')

frame_directory = video_folder + '/frames'

for video_file in video_files:
    os.system('ffmpeg -i {}/{} -r 1/{} {}/frames/%03d.jpg'.format(video_folder, video_file, frame_rate, video_folder))
    frame_files = os.listdir(frame_directory)
    for frame_file in frame_files:
        full_image_path = frame_directory + '/' + frame_file
        save_image(full_imageia_path)

stored_images = np.array(stored_images)
height, width, num_images = stored_images.shape
stored_images = stored_images.reshape(width, height, num_images)
stored_images = (stored_images - np.mean(stored_images)) / np.std(stored_images)
stored_images = np.clip(stored_images, 0, 1)
np.save('processed_images.npy', stored_images)
os.system('rm -r {}'.format(frame_directory))
