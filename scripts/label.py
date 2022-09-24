import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

# Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
input_size = 256
# image_path = 'PATH_TO_IMAGE.jpg'
# image = tf.io.read_file(image_path)
# image = tf.compat.v1.image.decode_jpeg(image)
# image = tf.expand_dims(image, axis=0)
# # Resize and pad the image to keep the aspect ratio and fit the expected size.
# resized_image, image_shape = keep_aspect_ratio_resizer(image, input_size)
# image_tensor = tf.cast(resized_image, dtype=tf.uint8)
# interpreter = tf.lite.Interpreter(model_path='model.tflite')
# # Output: [1, 6, 56] tensor that contains keypoints/bbox/scores.
# keypoints_with_scores = detect(
#     interpreter, tf.cast(image_tensor, dtype=tf.uint8))

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)
            
# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

path = "/content/RWF-2000/test/FIGHT"
for filename in os.listdir(path):
  cap = cv2.VideoCapture(path+ '/' + filename)
  np_path = path+ '/' + filename.split(".")[0]
  os.mkdir(path+ '/' + filename.split(".")[0])
  i = 0
  while cap.isOpened():
      ret, frame = cap.read()
      if ret:
          # Resize image
          img = frame.copy()
          img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 320,512)
          input_img = tf.cast(img, dtype=tf.int32)
          
          # Detection section
          results = movenet(input_img)
          keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
          
          np.save(np_path+'/'+str(i), keypoints_with_scores)
          i+=1
          # Render keypoints 
          # loop_through_people(frame, keypoints_with_scores, EDGES, 0.2)
          
          # cv2.imshow('Movenet Multipose', frame)
      else:
          break
      
      if cv2.waitKey(10) & 0xFF==ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()
  
  np.load("/content/RWF-2000/test/FIGHT/vid1/0.npy")
  
  
