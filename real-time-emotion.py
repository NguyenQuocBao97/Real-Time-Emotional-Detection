# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import glob
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)
# loop over the frames from the video stream


def trainDataset():
    trainFile = open("train.dat", "w")
    for imgDir in glob.glob('./dataset/*.tiff'):
        trainFile.write(imgDir+'\n')
    	datasetFrame = cv2.imread(imgDir)
    	datasetFrame = imutils.resize(datasetFrame, width=400)
    	datasetGray = cv2.cvtColor(datasetFrame, cv2.COLOR_BGR2GRAY)
    	datasetRects = detector(datasetGray, 0)
        for rect in datasetRects:
            shape = predictor(datasetGray, rect)
            shape = face_utils.shape_to_np(shape)
            if len(shape):
                for (x,y) in shape:
                    trainFile.write(str(x)+' '+str(y)+'\n')
    trainFile.close()


def readTrain():
    trainFile = open("train.dat", "r")
    global globDir, pointDataset
    tmp = []
    count = 0
    lines = trainFile.readlines()

    for line in lines:
        if count%69:
            x,y = line.split()
            tmp.append([int(x),int(y)])       
        else:
            if len(tmp):
                pointDataset.append(tmp)  
                tmp = []
            globDir.append(line)
        count += 1
    pointDataset.append(tmp)


if __name__ == "__main__":
    globDir = []
    pointDataset = []
    
    trainDataset()
    readTrain()
    for i in pointDataset:
        if len(i) != 68:
            print len(i)
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image

            
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # show the frame
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
