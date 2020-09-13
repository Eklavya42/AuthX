import cv2
from lib.src.align import detect_face
import tensorflow as tf
from model import (
facenet_model,
image_size,
config,
images_placeholder,
embeddings,
phase_train_placeholder,
facenet_persistent_session,
pnet,rnet,onet
)

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
from utils import (
    load_model,
    get_face,
    get_faces_live,
    forward_pass,
    save_embedding,
    load_embeddings,
    identify_face,
    allowed_file,
    remove_file_extension,
    save_image
)





class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame_orig = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        # faces = face_cascade.detectMultiScale(image, 1.3, 5)
        frame = cv2.resize(src=frame_orig, dsize=(0, 0), fx=0.5, fy=0.5)
        embedding_dict = load_embeddings()

        frame = frame[:, :, ::-1]

        if frame.size > 0:
            faces, rects = get_faces_live(
                img=frame,pnet=pnet,rnet=rnet,onet=onet,image_size=image_size)

            # If there are human faces detected
            if faces:
                for i in range(len(faces)):
                    face_img = faces[i]
                    rect = rects[i]

                    # Scale coordinates of face locations by the resize ratio
                    rect = [coordinate * 2 for coordinate in rect]

                    face_embedding = forward_pass(
                        img=face_img,
                        session=facenet_persistent_session,
                        images_placeholder=images_placeholder,
                        embeddings=embeddings,
                        phase_train_placeholder=phase_train_placeholder,
                        image_size=image_size
                    )

                    # Compare euclidean distance between this embedding and the embeddings in 'embeddings/'
                    identity = identify_face(
                        embedding=face_embedding,
                        embedding_dict=embedding_dict
                    )

                    cv2.rectangle(
                        img=frame_orig,
                        pt1=(rect[0], rect[1]),
                        pt2=(rect[2], rect[3]),
                        color=(255, 215, 0),
                        thickness=2
                    )

                    W = int(rect[2] - rect[0]) // 2

                    cv2.putText(
                        img=frame_orig,
                        text=identity,
                        org=(rect[0] + W - (W // 2), rect[1]-7),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 215, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )


        ret, jpeg = cv2.imencode('.jpg', frame_orig)
        return jpeg.tobytes()



#
# # Load FaceNet model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
# model_path = 'model/20170512-110547/20170512-110547.pb'
# facenet_model = load_model(model_path)
# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# image_size = 160
# images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
# embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
# phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#
# # Initiate persistent FaceNet model in memory
# facenet_persistent_session = tf.Session(graph=facenet_model, config=config)
#
# # Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
# pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)
