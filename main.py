import cv2  # for web camera
import tensorflow as tf
import os
import datetime
import json
from flask import Flask, request, jsonify, render_template, Response
from processing import recognise_text, crop_aadhar, get_address, get_labels_from_aadhar
from scipy.misc import imread
from lib.src.align import detect_face  # for MTCNN face detection
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from waitress import serve
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


from camera import VideoCamera

# import warnings
# warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["KMP_WARNINGS"] = "FALSE"

app = Flask(__name__)
app.secret_key = os.urandom(24)
# APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# path to upload images
uploads_path = './uploads/'
# path to face embeddings
embeddings_path = './embeddings/'


# route to processing images of id cards
@app.route('/image/upload', methods=['POST'])
def detail():
    ''' Detects text and face in Aadhaar Card '''

    if request.method == 'POST':


        # saving current timestamp
        current_time = str(datetime.datetime.now()).replace('-', '_').replace(':', '_')

        # The type of image i.e. Front or Back image
        image_type1 = 'Front'
        image_type2 = 'Back'

        # Path for Front image and the face image that will be croppped
        filename1 = uploads_path + image_type1 + '/' + current_time + '.jpg'
        photo_path = uploads_path + image_type1 + '/' + 'faces' + '/' + current_time + '.png'

        # Path for Back image and the face image that will be croppped
        filename2 = uploads_path + image_type2 + '/' + current_time + '.jpg'
        crop_path = uploads_path + image_type2 + '/temp/' + current_time + '.png'


        # if the Front folder (in uploads) doesn't already exist, create it
        if not os.path.exists(uploads_path + image_type1):
            os.mkdir(uploads_path + image_type1)
            # directory for saving faces in the id cards
            os.mkdir(uploads_path + image_type1 + '/' + 'faces')

        # if the Back folder (in uploads) doesn't already exist, create it
        if not os.path.exists(uploads_path + image_type2):
            os.mkdir(uploads_path + image_type2)
            os.mkdir(uploads_path + image_type2 + '/temp')

        # variable to store details extracted from card
        details = {}

        # get Front Card Photo from user
        photo1 = request.files['photo-front']
        photo1.save(filename1)

        # get Front Card Photo from user
        photo2 = request.files['photo-back']
        photo2.save(filename2)

        print("Processing Front Image ......")

        # Process The Front Card Image
        data, photo_path = recognise_text(filename1, photo_path)
        details = get_labels_from_aadhar(data)
        print("Processing Front Image ...... DONE")

        print("Processing Back Image .......")

        # Process The Back Card Image
        crop_aadhar(filename2, crop_path)
        data2, photo_path2 = recognise_text(crop_path, 'none')
        details.update(get_address(data2))
        print("Processing Back Image ....... DONE")

        os.remove(crop_path)

        data_dict = {'status':True, 'fields': details, 'image_path_front': filename1,'image_path_back': filename2, 'photo_path': photo_path}

        print("save into json files")
        # the json file where the output must be stored
        with open('myfile.json', 'a+') as out_file:
            json.dump(data_dict, out_file, indent = 6)

        img = imread(name=photo_path, mode='RGB')
        print("Processing Face Image .......")
        # Detect and crop a 160 x 160 image containing a human face in the image file
        img = get_face(
            img=img,
            pnet=pnet,
            rnet=rnet,
            onet=onet,
            image_size=image_size
        )
        embedding = forward_pass(
            img=img,
            session=facenet_persistent_session,
            images_placeholder=images_placeholder,
            embeddings=embeddings,
            phase_train_placeholder=phase_train_placeholder,
            image_size=image_size
        )


        print("Processing Face Image ....... DONE")
        # Save The Face embedding as the name of the Person
        filename = data_dict['fields']['Name']
        filename = secure_filename(filename=filename)
        # Save embedding to 'embeddings/' folder
        save_embedding(
            embedding=embedding,
            filename=filename,
            embeddings_path=embeddings_path
        )

        # Write the Raw and Cleaned Text detected from the Card
        with open('outputs.txt', 'a+') as f:
            f.write("##########################################################################\n\n")
            f.write('######################## Raw Output for Front Card Image #############################\n\n')
            for value in data:
                f.write(str(value) + '\n')
            f.write("##########################################################################\n\n")
            f.write('######################## Raw Output for Back Card Image #############################\n\n')
            for value in data2:
                f.write(str(value) + '\n')
            f.write('\n\n######################## Cleaned Output #############################\n\n')
            for key, value in details.items():
                f.write(str(key) + ' : ' + str(value) + '\n')
            f.write("##########################################################################\n\n")

        return jsonify(data_dict)

    else:
        # if not POST, terminate
        return jsonify({'status':False})


# route to do live face recognition
@app.route("/live", methods=['GET', 'POST'])
def face_detect_live():
    """Detects faces in real-time via Web Camera."""

    embedding_dict = load_embeddings()
    if embedding_dict:
        try:
            cap = cv2.VideoCapture(0)

            while True:
                return_code, frame_orig = cap.read()  # Read frame

                # Resize frame to half its size for faster computation
                frame = cv2.resize(src=frame_orig, dsize=(0, 0), fx=0.5, fy=0.5)

                # Convert the image from BGR color (which OpenCV uses) to RGB color
                frame = frame[:, :, ::-1]

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if frame.size > 0:
                    faces, rects = get_faces_live(
                        img=frame,
                        pnet=pnet,
                        rnet=rnet,
                        onet=onet,
                        image_size=image_size
                    )

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

                        cv2.imshow(winname='Video', mat=frame_orig)
                    # Keep showing camera stream even if no human faces are detected
                    cv2.imshow(winname='Video', mat=frame_orig)
                else:
                    continue

            cap.release()
            cv2.destroyAllWindows()

            return render_template(template_name_or_list='index.html')

        except Exception as e:
            print(e)

    else:
        return render_template(
            template_name_or_list="warning.html",
            status="No embedding files detected! Please upload image files for embedding!"
        )



@app.route("/browse",methods =['GET', 'POST'])
def video_feed():
    return Response(gen(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')




# route to upload images of id cards and the form
@app.route("/details")
def details_page():
    """Renders the 'card-form.html' page for manual image file uploads."""
    return render_template(template_name_or_list="card-from.html")

@app.route("/feed")
def video_feed_live():
    """Renders the 'browser.html' page."""
    return render_template(template_name_or_list="browser.html")

@app.route("/")
def index_page():
    """Renders the 'index.html' page."""
    return render_template(template_name_or_list="index.html")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':
    """Server Run"""

    app.run(host='0.0.0.0', debug=True)
