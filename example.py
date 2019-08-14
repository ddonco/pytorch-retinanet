import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
from pathlib import Path

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


def main(model, csv_val, csv_classes):

	dataset_val = CSVDataset(train_file=csv_val, class_list=csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	retinanet = torch.load(model, map_location=torch.device('cpu'))

	use_gpu = False

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, label_name, label_score):

		caption = f"{label_name}: {label_score:.2f}"
		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			st = time.time()
			scores, classification, transformed_anchors = retinanet(data['img'].float())
			print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores>0.7)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				label_score = scores[idxs[0][j]]
				draw_caption(img, (x1, y1, x2, y2), label_name, label_score)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(f"{label_name}: {label_score:.2f}")

			cv2.imshow('img', img)
			cv2.waitKey(0)

if __name__ == '__main__':
    model_path = Path('models')
    label_path = Path('labels')

    model = model_path/'csv_retinanet_35.pt'
    csv_val = label_path/'retinanet_defect_labels_colab-200px-valid-short.csv'
    csv_classes = label_path/'retinanet_defect_classes.csv'
    main(model, csv_val, csv_classes)