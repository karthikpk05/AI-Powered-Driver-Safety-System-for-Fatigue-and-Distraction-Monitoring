import cv2
import os

folders = ["dataset/eye/open", "dataset/eye/closed"]

for folder in folders:
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        image = cv2.imread(path)
        if image is None:
            print("Removing corrupted:", path)
            os.remove(path)

print("Dataset cleaning completed.")
