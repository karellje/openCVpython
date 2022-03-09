import os
import cv2
import numpy as np
from PIL import Image

###For take images for datasets
def take_img(ID,Name):
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('/Users/jeremiekarell/PycharmProjects/openCVpython/venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite("dataset/ " + Name + "." + ID + '.' + str(sampleNum) + ".jpg",
                        gray[y:y + h, x:x + w])
            cv2.imshow('Frame', img)
            # wait for 100 miliseconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # break if the sample number is morethan 100
        elif sampleNum > 100:
            break
    cam.release()
    cv2.destroyAllWindows()

    print("Images Saved  : " + ID + " Name : " + Name)



###For train the model
def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("/Users/jeremiekarell/PycharmProjects/openCVpython/venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    try:
        global faces, Id
        faces, Id = getImagesAndLabels("dataset")
    except Exception as e:
        print("no dataset folder")

    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("model/trained_model2.yml")
    except Exception as e:
        print("Did not work")


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids

def Facialrecognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('model/trained_model2.yml')
    cascadePath = "/Users/jeremiekarell/PycharmProjects/openCVpython/venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    id_to_pic = 0

    # add the list of names of your dataset here
    names = ['None', 'jeremie', 'Ronny', 'Oscar']
    model, classes, colors, output_layers = load_yolo()
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width, channels = img.shape
        blob, outputs = detect_objects(img, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id_to_pic, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            draw_labels(boxes, confs, colors, class_ids, classes, img)

            # If confidence is less them 100 ==> "0" : perfect match
            if (confidence < 50):
                id_to_pic = names[id_to_pic]
                confidence = "  {0}%".format(round(100 - confidence))
                print(confidence)
            else:
                id_to_pic = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                print(confidence)


            cv2.putText(
                img,
                str(id_to_pic),
                (x + 5, y - 5),
                font,
                1,
                (255, 255, 255),
                2
            )

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

###----------------------------------------#### proj 2 del 2
def load_yolo():
    net = cv2.dnn.readNet("yolov3_training_last-4.weights", "yolov3_testing.cfg")
    classes = []
    with open("classes1.name", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()



print("välkommen till Facial recognition och iphone detektion")
ID = input("Enter id: (between 1-100)")
Name = input("Enter username:")
while True:
    userinput = input("1.take photos: \n2.train model: \n3.show result: \n 4.EXIT")
    if userinput == "1":
        take_img(ID,Name)
        continue

    elif userinput == "2":
        trainimg()
        print("training done: exit now")
        continue

    elif userinput == "3":
        Facialrecognition()
    elif userinput == "4": break
    else:
        print("use numbers please")
        continue



