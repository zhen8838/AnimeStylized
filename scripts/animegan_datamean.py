import cv2
import argparse
import os
from glob import glob
from tqdm import tqdm
import numpy as np


def read_img(image_path):
  img = cv2.imread(image_path)
  assert len(img.shape) == 3
  B = img[..., 0].mean()
  G = img[..., 1].mean()
  R = img[..., 2].mean()
  return B, G, R


def main(dataset):
  file_list = glob(os.path.join(dataset, '*.jpg'))
  image_num = len(file_list)
  print('image_num:', image_num)

  B_total = 0
  G_total = 0
  R_total = 0
  for f in tqdm(file_list):
    bgr = read_img(f)
    B_total += bgr[0]
    G_total += bgr[1]
    R_total += bgr[2]

  B_mean, G_mean, R_mean = B_total / image_num, G_total / image_num, R_total / image_num
  mean = (B_mean + G_mean + R_mean) / 3

  print('RGB mean diff')
  print(np.asfarray((mean - R_mean, mean - G_mean, mean - B_mean), dtype='float32'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', help="get the mean values of rgb from dataset",
                      type=str, default='')

  args = parser.parse_args()

  main(args.dataset)
