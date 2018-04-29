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
import collections as co
APPROXIMATE_CATE = 20
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-t", "--train", type=int, default=0,
                help="set 1 when to re-train the data")

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


def getSVM(datasetFrame):
    datasetFrame = imutils.resize(datasetFrame, width=400)
    datasetGray = cv2.cvtColor(datasetFrame, cv2.COLOR_BGR2GRAY)
    datasetRects = detector(datasetGray, 0)
    dist = []
    for rect in datasetRects:
        shape = predictor(datasetGray, rect)
        shape = face_utils.shape_to_np(shape)
        listx = []
        listy = []
        sumx = 0
        sumy = 0
        for (x, y) in shape:
            listx.append(x)
            listy.append(y)
            sumx += x
            sumy += y
        meanx = sumx / 68
        meany = sumy / 68
        
        for x, y in zip(listx, listy):
            dist.append (round(((x-meanx)**2 + (y-meany)**2)**.5, 4))
        minDist = min(dist)
        maxDist = max(dist)
        difMinMax = maxDist-minDist
        normDist = []
        for index in range(len(dist)):
            dist[index] = round((dist[index]-minDist)/difMinMax,4)
    return [datasetFrame,dist]


def trainDataset():
    trainFile = open("train.dat", "w")
    for imgDir in glob.glob('./dataset/*.tiff'):
        trainFile.write(imgDir+'\n')
        datasetFrame = cv2.imread(imgDir)
        dist = getSVM(datasetFrame)[1]
        for i in dist:
            trainFile.write(str(i)+' ')
        trainFile.write('\n')

    trainFile.close()


def readTrain():
    trainFile = open("train.dat", "r")
    global globDir, svmDataset
    count = 0
    lines = trainFile.readlines()
    tmp = []
    for line in lines:
        if count % 2:
            tmp = map(float, line.split())
            svmDataset.append(tmp)
        else:
            globDir.append(line)
        count += 1

def findLessDifOffset(sumdif):
    mindif = sumdif[0]
    index = 0
    for i,v in enumerate(sumdif):
        if v < mindif:
            mindif,index = v, i
    return index

def findCategory(imageDir): #happy, sad, neutral, suprise, angry, fear, digust
    return (imageDir.split('.')[2])[:2]
    
def getKeyWithMaxValue(d):
    maxV = -1
    key = ""
    for k,v in d.items():
        if v > maxV:
            key = k
            maxV = v
    if key == "HA":
        return "HAPPY"
    elif key == "AN":
        return "ANGRY"
    elif key == "DI":
        return "DIGUST"
    elif key == "SA":
        return "SAD"
    elif key == "NE":
        return "NEUTRAL"
    elif key == "SU":
        return "SUPRISE"
    elif key == "FE":
        return "FEAR"
    return key


if __name__ == "__main__":

    globDir = []
    svmDataset = []
    if args["train"]:
        trainDataset()
    readTrain()
    count = APPROXIMATE_CATE
    d = co.defaultdict(lambda: 0)
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame, vsSVM = getSVM(frame)
        if vsSVM != []:
            dif = []
            for svmFrame in svmDataset:
                sumdif = 1.0
                for difIndex in range(len(svmFrame)):
                    sumdif += abs(vsSVM[difIndex]-svmFrame[difIndex])
                dif.append(round(sumdif,4))
            if count:
                d[findCategory(globDir[findLessDifOffset(dif)])] += 1
                count -= 1
            else:
                count = APPROXIMATE_CATE
                print getKeyWithMaxValue(d)
                
                d = co.defaultdict(lambda: 0)

        # show the frame
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
