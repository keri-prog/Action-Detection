import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
input_size = 256

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

path = "/home/kushojha/Action-Detection/RWF-2000/train/Fight"
# path = "/home/kushojha/Action-Detection/RWF-2000/val/NonFight"
for filename in tqdm(os.listdir(path)):
    cap = cv2.VideoCapture(path + '/' + filename)
    np_path = path + '/' + filename.split(".")[0]
    os.mkdir(path + '/' + filename.split(".")[0])
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize image
            img = frame.copy()
            img = tf.image.resize_with_pad(
                tf.expand_dims(img, axis=0), 320, 512)
            input_img = tf.cast(img, dtype=tf.int32)

            # Detection section
            results = movenet(input_img)
            keypoints_with_scores = results['output_0'].numpy()[
                :, :, :51].reshape((6, 17, 3))

            np.save(np_path+'/'+str(i), keypoints_with_scores)
            i += 1
            # Render keypoints
            # loop_through_people(frame, keypoints_with_scores, EDGES, 0.2)

            # cv2.imshow('Movenet Multipose', frame)
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
