import cv2
import pytesseract as pyt
import re
from ctpn.demo_pb import get_coords
import numpy as np
# from keras.models import model_from_json
import tensorflow as tf
from keras import backend as k


# function to recognise text from image
def recognise_text(image_path, photo_path):

    # read image and convert to grayscale
    image = cv2.imread(image_path, 0)

    # get coordinates of text using ctpn
    coordinates = get_coords(image_path)

    detected_text = []

    # sorting coordinates from top to bottom
    coordinates = sorted(coordinates, key = lambda coords: coords[1])

    # looping through all the text boxes
    for coords in coordinates:
        # x, y, width, height of the text box
        x, y, w, h = coords

        # cropping image based on the coordinates
        temp = image[y:h, x:w]

        # binarizing image
        _, thresh = cv2.threshold(temp, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # padding the image with 10 pixels for better prediction with tesseract
        thresh = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # get text from the image, lang = english + hindi + marathi, config = use lstm for prediction
        # text = pyt.image_to_string(thresh, lang="eng+hin+mar", config=('--oem 1 --psm 3'))
        text = pyt.image_to_string(thresh, lang="eng", config=('--oem 1 --psm 3'))

        # clean text and remove noise
        text = clean_text(text)

        # ignore text if the length of text is less than 3
        if len(text) < 3:
            continue
        detected_text.append(text)

    # find face in the image
    face, found = get_photo(image)

    # if a face is found save it to faces directory
    if found:
        cv2.imwrite(photo_path, face)
    else:
        photo_path = face

    # return detected text and the face path
    return detected_text, photo_path


# function to remove noise and unnecessary characters from string
def clean_text(text):
    if text != ' ' or text != '  ' or text != '':
        text = re.sub('[^A-Za-z0-9-/,.() ]+', '', text)
        text = text.strip()
        text = re.sub(r'\s{2,}', ' ', text)

    return text


# extract labels from aadhar image
def get_labels_from_aadhar(temp):
    imp = {}

    # reverse list to parse through it starting from the aadhar number
    temp = temp[::-1]
    # parse through the list
    for idx in range(len(temp)):

        try:
            # if string similar to aadhar number is found, use it as a hook to find other details
            if re.search(r"[0-9]{4}\s[0-9]{4}\s[0-9]{4}", temp[idx]):
                try:
                    imp['Aadhar No'] = re.findall(r"[0-9]{4}\s[0-9]{4}\s[0-9]{4}", temp[idx])[0]
                except Exception as _:
                    imp['Aadhar No'] = "Not Found"
                if temp[idx + 1].endswith("Female") or temp[idx + 1].endswith("FEMALE"):
                    imp["Gender"] = "Female"
                elif temp[idx + 1].endswith("Male") or temp[idx + 1].endswith("MALE"):
                    imp["Gender"] = "Male"
                elif temp[idx + 2].endswith("Female") or temp[idx + 2].endswith("FEMALE"):
                    imp["Gender"] = "Female"
                elif temp[idx + 2].endswith("Male") or temp[idx + 2].endswith("MALE"):
                    imp["Gender"] = "Male"
                elif temp[idx + 3].endswith("Female") or temp[idx + 3].endswith("FEMALE"):
                    imp["Gender"] = "Female"
                elif temp[idx + 3].endswith("Male") or temp[idx + 3].endswith("MALE"):
                    imp["Gender"] = "Male"

            elif re.search(r"[0-9]{2}\-|/[0-9]{2}\-|/[0-9]{4}", temp[idx]):
                # if string similar to date is found, use it as a hook to find other details
                try:
                    imp["Date of Birth"] = re.findall(r"[0-9]{2}\-[0-9]{2}\-[0-9]{4}", temp[idx])[0]
                except Exception as _:
                    imp["Date of Birth"] = re.findall(r"[0-9]{2}/[0-9]{2}/[0-9]{4}", temp[idx])[0]
                imp["Name"] = temp[idx + 1]

            elif "Year of Birth" in temp[idx]:
                # handle variation of 'Year of Birth' in place of DOB
                try:
                    imp["Year of Birth"] = re.findall(r"[0-9]{4}", temp[idx])[0]
                except Exception as _:
                    imp["Year of Birth"] = "Not Found"
                imp["Name"] = temp[idx + 1]

            elif re.search(r"[0-9]{4}", temp[idx]):
                # handle exception if Year of Birth is not found but string similar to year is found
                try:
                    imp["Year of Birth"] = re.findall(r"[0-9]{4}", temp[idx])[0]
                except Exception as _:
                    imp["Year of Birth"] = "Not Found"
                imp["Name"] = temp[idx + 1]


        except Exception as _:
            pass
    return imp




# function to find face in the image
def get_photo(image):

    # Image Should be 1920 x 1080 pixels
    scale_factor = 1.1
    min_neighbors = 3
    min_size = (150, 150)
    flags = cv2.CASCADE_SCALE_IMAGE

    # using frontal face haar cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # detect faces of different sizes
    faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
                                          minSize = min_size, flags = flags)

    # crop the face if found
    try:
        x, y, w, h = faces[0]
        face = image[y-50:y+h+40, x-10:x+w+10]
        return face, True
    except Exception as _:
        return "Photo not found!", False

# function to remove border lines
def _lineRemoval(img):
    min_length=140
    # getting matrix of values in all the rows
    matrix = _imgToMatrixR(img)
    # parsing through the matrix row by row
    for i in range(0, len(matrix)):
        row=matrix[i]
        start=-1
        end=0
        conn=0
        for j in range(0, len(row)):
            if (row[j]==0):
                conn=conn+1
                # first point in the line .
                if( start == -1 ):
                    start = j
                # last point in the row .
                if( j == len(row)-1 ):
                    end =j
                    if (conn > min_length):
                        img[i-2:i+4, start:end+1] = 255
                    start = -1
                    end = 0
                    conn = 0
            # end of the line
            else:
                end =j
                if (conn >min_length):
                    img[i-2:i+4, start:end+1] = 255
                start = -1
                end = 0
                conn = 0

    return img


# this function converts image into matrix of image rows
def _imgToMatrixR(img):
    # get dimensions
    height, width = img.shape
    matrix = []
    # getting pixels values for all rows
    for i in range(0, height):
        row = []
        for j in range(0, width):
            row.append(img[i,j])
        matrix.append(row)
    return matrix


# this function convert image into matrix of image columns
def _imgToMatrixC(img):
    # get dimensions
    height, width = img.shape
    matrix = []
    # getting pixels values for all columns
    for i in range(0, width):
        col = []
        for j in range(0, height):
            col.append(img[j, i])
        matrix.append(col)
    return matrix


# this function clears all horizontal boundaries around the input image
def clearBounds_horiz(img):

    height, width = img.shape
    matrix = _imgToMatrixR(img)
    white_counter = _countPixel(matrix,255)

    for i in range (0,height):
        if(white_counter[i]>= width-1):
            img = img[1:height,0:width]
        else:
            break

    new_height, width = img.shape
    for i in range (1,height):
        if(white_counter[height-i]>= width-1):
            img = img[0:new_height-i,0:width]
        else:
            break

    return img


# this function clears all vertical boundaries around the input image
def clearBounds_vert(img):

    height, width = img.shape
    matrix = _imgToMatrixC(img)
    white_counter = _countPixel(matrix,255)

    for i in range (0,width):
        if(white_counter[i]>= height-1):
            img = img[0:height,1:width]
        else:
            break

    height, new_width = img.shape
    for i in range (1,width):
        if(white_counter[width-i]>= height-1):
            img = img[0:height,0:new_width-i]
        else:
            break

    return img


# this function counts a specific value (parameter p) in matrix
def _countPixel(matrix,p):
    counter = []
    for k in range(0, len(matrix)):
        counter.append(matrix[k].count(p))
    return counter


# function to crop aadhar back image
def crop_aadhar(image_path, crop_path):
    image = cv2.imread(image_path, 0)
    height, width = image.shape
    image = image[int(height * (15 / 100)):int(height * (70 / 100)), int(width * (40 / 100)):]
    cv2.imwrite(crop_path, image)


def get_address(details):
    imp = {'Address': ''}

    try:
        if 'Address' in details[0]:
            if details[0].split('Address', 1)[1].strip() != '':
                imp["Address"] = details[0].split('Address', 1)[1].strip()
            for line in details[1:]:
                imp["Address"] += '\n' + line
            imp['Address'] = imp['Address'].strip()

        elif 'Address' in details[1]:
            if details[1].split('Address', 1)[1].strip() != '':
                imp["Address"] = details[1].split('Address', 1)[1].strip()
            for line in details[2:]:
                imp["Address"] += '\n' + line
            imp['Address'] = imp['Address'].strip()

        elif 'Address' in details[2]:
            if details[2].split('Address', 1)[1].strip() != '':
                imp["Address"] = details[2].split('Address', 1)[1].strip()
            for line in details[3:]:
                imp["Address"] += '\n' + line
            imp['Address'] = imp['Address'].strip()

        else:
            imp["Address"] = 'Failed to read Address'
    except Exception as _:
        imp["Address"] = 'Failed to read Address'

    return imp
