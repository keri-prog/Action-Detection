import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import math


class MovenetLite:
    def __init__(self) -> None:
        # Optional if you are using a GPU
        if tf.test.is_built_with_cuda():
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        self.interpreter = tf.lite.Interpreter(
            model_path='lite-model_movenet_multipose_lightning_tflite_float16_1.tflite')

    def detect(interpreter, input_tensor):
        """Runs detection on an input image.

        Args:
            interpreter: tf.lite.Interpreter
            input_tensor: A [1, input_height, input_width, 3] Tensor of type tf.float32.
            input_size is specified when converting the model to TFLite.

        Returns:
            A tensor of shape [1, 6, 56].
        """

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
        if is_dynamic_shape_model:
            input_tensor_index = input_details[0]['index']
            input_shape = input_tensor.shape
            interpreter.resize_tensor_input(
                input_tensor_index, input_shape, strict=True)
        interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

        interpreter.invoke()

        keypoints_with_scores = interpreter.get_tensor(
            output_details[0]['index'])
        return keypoints_with_scores

    def keep_aspect_ratio_resizer(image, target_size):
        """Resizes the image.

        The function resizes the image such that its longer side matches the required
        target_size while keeping the image aspect ratio. Note that the resizes image
        is padded such that both height and width are a multiple of 32, which is
        required by the model.
        """
        _, height, width, _ = image.shape
        if height > width:
            scale = float(target_size / height)
            target_height = target_size
            scaled_width = math.ceil(width * scale)
            image = tf.image.resize(image, [target_height, scaled_width])
            target_width = int(math.ceil(scaled_width / 32) * 32)
        else:
            scale = float(target_size / width)
            target_width = target_size
            scaled_height = math.ceil(height * scale)
            image = tf.image.resize(image, [scaled_height, target_width])
            target_height = int(math.ceil(scaled_height / 32) * 32)
        image = tf.image.pad_to_bounding_box(
            image, 0, 0, target_height, target_width)
        return (image,  (target_height, target_width))

    def infer_lite(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        input_size = 256
        frame_keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Resize image
                img = frame.copy()
                img = tf.image.resize_with_pad(
                    tf.expand_dims(img, axis=0), 320, 512)
                input_img = tf.cast(img, dtype=tf.int32)

                resized_image, image_shape = self.keep_aspect_ratio_resizer(
                    input_img, input_size)
                image_tensor = tf.cast(resized_image, dtype=tf.uint8)

                interpreter = tf.lite.Interpreter(
                    model_path='../weights/lite-model_movenet_multipose_lightning_tflite_float16_1.tflite')

                # Output: [1, 6, 56] tensor that contains keypoints/bbox/scores.
                keypoints_with_scores = self.detect(
                    interpreter, tf.cast(image_tensor, dtype=tf.uint8))
                keypoints_with_scores = keypoints_with_scores[:, :, :51].reshape(
                    (6, 17, 3))

                frame_keypoints.append(keypoints_with_scores)
            else:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        frame_keypoints = np.array(frame_keypoints)
        return frame_keypoints


class Movenet:
    def __init__(self) -> None:
        # Optional if you are using a GPU
        if tf.test.is_built_with_cuda():
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        model = hub.load(
            'https://tfhub.dev/google/movenet/multipose/lightning/1')
        self.movenet = model.signatures['serving_default']

    def infer(self, video_path: str):
        frame_keypoints = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Resize image
                img = frame.copy()
                img = tf.image.resize_with_pad(
                    tf.expand_dims(img, axis=0), 320, 512)
                input_img = tf.cast(img, dtype=tf.int32)

                # Detection section
                results = self.movenet(input_img)
                keypoints_with_scores = results['output_0'].numpy()[
                    :, :, :51].reshape((6, 17, 3))
                frame_keypoints.append(keypoints_with_scores)
            else:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        frame_keypoints = np.array(frame_keypoints)

        return frame_keypoints
