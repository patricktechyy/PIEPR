# thank you pupilext authors!!!
# (づ ￣ ³￣)づ


import pypupilext as pp
import cv2
import pandas as pd
import time
import matplotlib.pyplot as plt

#testPath = "/Users/honganh/Documents/nerd folder/smpf thing/implement/videoImplement/frames/left/frame0.bmp"
testPath = "frames/left/frame0.bmp"

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def detect(imagePath):
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    pupilClass = pp.Pupil()
    assert pupilClass.confidence == -1

    pure = pp.PuReST()
    pure.maxPupilDiameterMM = 7

    im_reized = img
    pupil = pure.runWithConfidence(im_reized)
    diam = pupil.diameter()
    data = pd.DataFrame([{'Outline Conf': pupil.outline_confidence, 'PupilDiameter': diam}])
    print(data)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if diam is not None and diam >= 0:
        img_plot = cv2.ellipse(
            img_bgr,
            (int(pupil.center[0]), int(pupil.center[1])),
            (int(pupil.minorAxis()/2), int(pupil.majorAxis()/2)),
            pupil.angle,
            0,
            360,
            (0, 0, 255),
            1,
        )
    else:
        img_plot = img_bgr

    resize = ResizeWithAspectRatio(img_plot, width=800)

    return resize, pupil.outline_confidence, diam

#fig = plt.figure(figsize=(20, 8))
#ax1 = plt.subplot(1, 2, 1)
#im1 = ax1.imshow(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
#fig.tight_layout()
#
#plt.show()
# If you want to show the image using an opencv window instead of matplotlib
#cv2.imshow("window", resize)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1)