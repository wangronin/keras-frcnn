import logging, random, string, sys, os, atexit, \
    errno, time, signal, cv2, re, warnings

import numpy as np

data = './train'
image_path = os.path.expanduser(os.path.expandvars(data))
IMGs = [
    os.path.join(image_path, f) for f in os.listdir(image_path) \
        if re.search(r'.*\.jpg|png|bmp$', f)
]

X = np.array([cv2.imread(_) for _ in IMGs])
print(np.mean(X, axis=3))
