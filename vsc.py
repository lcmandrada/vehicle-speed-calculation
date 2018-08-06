'''
Optimization of Vehicle Speed Calculation
on Raspberry Pi Using Sparse Random Projection

Luke Clark M. Andrada
Joseph Merlin F. Hernandez
Eduardo C. Garcia
'''

import cv2
import time
from datetime import datetime
import numpy as np
from math import ceil
from math import tan
from math import atan
from math import radians
from sklearn import random_projection as rp
from picamera.array import PiRGBArray
from picamera import PiCamera

def init():
    '''
    initialize system

    variables
    x - higher quality when lower, higher resolution when lower
    d - higher quality when higher
    r - higher quality when higher
    '''

    # declare global variables
    global camera, rawCapture, opt, components, xdiv, ydiv, gaussian
    global fcount, ftrack, ttrack, frate, vtime, stamp, vspeed, fzone, dzone
    global gamma, theta, phi, d, z
    global measured, predicted, ppredicted, kalman

    # initialize camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (480, 368)
    camera.framerate = 60
    rawCapture = PiRGBArray(camera, size=(480, 368))

    # set vertical and horizontal divisions
    ydiv = 368
    xdiv = 16 #23

    # set number of components
    components = 1

    # init dimensionality reduction with sparse random projection
    opt = rp.SparseRandomProjection(n_components=components)
    orig = np.zeros((ydiv, xdiv), np.uint8)
    opt.fit(orig)

    # init object detection with gaussian mixture model
    gaussian = cv2.createBackgroundSubtractorMOG2()

    # init frame count
    fcount = 0

    # init entry and exit frames and timestamps
    ftrack = [0, 0]
    ttrack = [0, 0]

    # init A and B timestamps
    vtime = [0, 0]
    stamp = [0, 0]

    # init effective frame rate and vehicle speed
    gamma = [0, 0]
    theta = 0
    phi = 0
    z = [0, 0]
    frate = 0
    vspeed = 0

    # init entry and exit flags
    fzone = [True, True]
    dzone = [True, True]

    # init object tracking with kalman filter
    kalman = cv2.KalmanFilter(4, 2)

    # set kalman filter parameters
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003

    # init measured, predicted and previously predicted coordinates
    measured = np.array((2, 1), np.float32)
    predicted = np.zeros((2, 1), np.float32)
    ppredicted = np.zeros((2, 1), np.float32)

def srp(img):
    ''' optimize using sparse random projection '''

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # init projection width tracker
    pwidth = int(100 / xdiv)

    # init output
    out = np.zeros((height, ceil(width * components / xdiv)), dtype=np.uint8)

    # iterate each division on region of interest
    for i in range(100, 400, xdiv):
        for j in range(0, height, ydiv):
            # cut division
            img = gray[j:j + ydiv, i:i + xdiv]

            # project division
            proj = opt.transform(img)

            # normalize projection within gray
            if proj.max() != 0:
                proj = np.multiply(np.divide(proj, proj.max()), 255)

            # stitch division
            out[j:j + ydiv, pwidth:pwidth + components] = proj
        else:
            # update projection width tracker
            pwidth += components

    return out

def morph(img):
    ''' reduce noise and fill holes using morphological operations '''

    # set kernel for noise
    kernel = np.ones((15, ceil(15 / xdiv)), np.uint8)

    # apply opening for noise reduction
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # set kernel for hole
    kernel = np.ones((50, ceil(40  / xdiv)), np.uint8)

    # apply closing for hole filling
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # return inverted output
    return cv2.bitwise_not(img)

def contour(img, orig):
    ''' detect blobs using contours '''

    # find contours for blobs
    _, contours, heirarchy = cv2.findContours(img, 1, 2)

    # iterate every contour
    for contour in contours:
        # get box parameters
        x, y, w, h = cv2.boundingRect(contour)

        # filter contours
        if w < (430 / xdiv) and h < 318 and w > (100 / xdiv) and h > 15:
            x, w = (x * xdiv, w * xdiv)

            # track vehicle
            orig = track(x, y, w, h, orig)

            # calculate effective frame rate and vehicle speed
            calculate(measured, ppredicted)

            # draw boxes
            cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # return frames
    return orig

def track(x, y, w, h, orig):
    ''' track vehicle using kalman filter '''

    # declare global variables
    global measured, predicted, ppredicted

    # set measured coordinates
    measured = np.array([[np.float32(x + (w / 2))], [np.float32(y + h)]])

    # update corrected states
    kalman.correct(measured)

    # set previous predicted coordinates
    ppredicted = predicted

    # set predicted coordinates
    predicted = kalman.predict()

    # return original
    return orig

def calculate(m, pp):
    ''' calculate vehicle speed using pinhole model '''

    # declare global variables
    global fcount, ftrack, ttrack, frate, vtime, stamp, vspeed, fzone, dzone
    global gamma, theta, phi, d, z

    # if measured and predicted coordinates are within threshold, if vehicle is tracked
    if abs(int((m[0]) - (pp[0]))) < 20 and abs(int((m[1]) - (pp[1]))) < 100:
        # if vehicle entered fzone
        if fzone[0]:
            # save F1 and T1
            ftrack[0] = fcount
            ttrack[0] = time.time() - start
            print("F1: %.2f f\nT1: %.2f s" % (ftrack[0], ttrack[0]))

            # disable flag
            fzone[0] = False

        # if vehicle entered dzone
        if dzone[0] and int(m[1]) >= 100:
            # save t1
            vtime[0] = time.time() - start
            stamp[0] = str(datetime.now().time())
            print("t1: %.2f s %s" % (vtime[0], stamp[0]))
            z[0] = abs(184 - int(m[1]))
            gamma[0] = atan((z[0] * (2.76 / 368)) / 3.04)
            theta = radians(24.4) - gamma[0]

            # disable flag
            dzone[0] = False

        # if vehicle exited dzone
        if dzone[1] and int(m[1]) >= 260:
            # save t2
            vtime[1] = time.time() - start
            stamp[1] = str(datetime.now().time())
            print("t2: %.2f s %s" % (vtime[1], stamp[1]))
            z[1] = abs(int(m[1]) - 184)
            gamma[1] = atan((z[1] * (2.76 / 368)) / 3.04)
            phi = radians(24.4) + gamma[1]

            # calculate vehicle speed
            d = 3.1 * (tan(radians(40) + phi) - tan(radians(40) + theta))
            vspeed = (d / (vtime[1] - vtime[0])) * 3.6
            print("S: %.2f kph" % (vspeed))

            # disable flag
            dzone[1] = False

        # if vehicle exited fzone
        if fzone[1] and int(m[1]) > 261:
            # save F2 and T2
            ftrack[1] = fcount
            ttrack[1] = time.time() - start
            print("F2: %.2f f\nT2: %.2f s" % (ftrack[1], ttrack[1]))

            # calculate effective frame rate
            frate = (ftrack[1] - ftrack[0]) / (ttrack[1] - ttrack[0])
            print("E: %.2f fps" % (frate))

            # save to file
            with open("output_vsc+.txt", "a") as f:
                f.write("F1: %.2f f\nT1: %.2f s\nF2: %.2f f\nT2: %.2f s\nE: %.2f fps\n\
                        d: 4.47 m\nt1: %.2f s %s\nt2: %.2f s %s\nS: %.2f kph\n\n" \
                        % (ftrack[0], ttrack[0], ftrack[1], ttrack[1],
                           frate, vtime[0], stamp[0], vtime[1], stamp[1], vspeed))

            # disable flag
            fzone[1] = False

def main():
    ''' main function '''

    # init variables
    init()

    # declare global variables
    global fcount, width, height

    # set output video parameters
    width = 480
    height = 368
    fps = 10

    # set output video writer
    out = cv2.VideoWriter('vsc+out.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))

    # warm camera up
    time.sleep(0.1)

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image,
        # then initialize the timestamp and occupied/unoccupied text
        orig = frame.array

        # count frame
        fcount += 1

        # apply sparse random projection
        proj = srp(orig)

        # apply background subtraction
        fg = gaussian.apply(proj)

        # apply morphological operations
        fg = morph(fg)
        fg[0:10, 0:479] = 255
        fg[357:367, 0:479] = 255
        fg[0:367, 0:10] = 255
        fg[0:367, 469:479] = 255

        # apply blob detection
        orig = contour(fg, orig)

        # draw A and B
        cv2.line(orig, (0, 100), (479, 100), (255, 255, 255), 2)
        cv2.line(orig, (0, 260), (479, 260), (255, 255, 255), 2)

        # write vehicle speed to orig
        cv2.putText(orig, "%.2f" % (vspeed), (10, 357),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # display
        cv2.imshow('orig', orig)

        # save video
        out.write(orig)

        # terminate
        rawCapture.truncate(0)

        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destroy
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    global start
    start = time.time()
    main()
    elapsed = time.time() - start
    print('T: %.2f' % elapsed)
