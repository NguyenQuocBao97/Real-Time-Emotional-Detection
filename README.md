# Real Time Emotional Detection
Use this application to find the emotion of people with your camera

This project used OpenCV 3.3.0 as a main framework to develop, and required python plugin to build.

To run this application, make sure to have:
- Visual Studio Code 
Install here: https://code.visualstudio.com/
- OpenCV 3.0 (If you are using OpenCV 2, you will have to reinstall it to a newer version)
Install here: https://opencv.org/releases.html
- Dlib (an open-source library written by C++)
Find it here: http://dlib.net/

## After cloning this project, please follow these instructions to run this program

- $python real-time-emotion.py -p shape_predictor_68_face_landmarks.dat

If your database changed, and you need the train.dat file to update:
- $python real-time-emotion.py -p shape_predictor_68_face_landmarks.dat -t 1

## Hope you enjoy!
  
# Reference Algorithm
[Emotion Recognition using Facial Landmarks, Python, DLib and OpenCV](http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/)

### Note:
- If you have any issues, comment and I'll help you
