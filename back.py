#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import stats

N_FRAMES = 113

frames = [
    cv2.imread(f"../img_1fps/{i:03d}.png", cv2.IMREAD_COLOR)
    for i in range(1, N_FRAMES + 1)
]
frames = np.array(frames)

result = stats.mode(frames, axis=0)
print(result)

cv2.imwrite("back.png", result.mode[0])