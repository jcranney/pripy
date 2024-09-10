# -*- coding: utf-8 -*-
"""
Usage: test_slm.py [options]

    -s, --slm               Use the SLM
    -m, --monitor <id>      The monitor to use [default: 1]
    -r, --radius <pixels>   Radius of pupil to use in pixels [default: 500]
    -x, --xoffset <pixels>  x offset in pixels [default: 0]
    -y, --yoffset <pixels>  y offset in pixels [default: 0]
"""

from docopt import docopt
import cv2
import numpy as np
import slmpy

if __name__ == "__main__":
    doc = docopt(__doc__)

    if doc["--slm"]:
        monitor_id = int(doc["--monitor"])
    else:
        monitor_id = -1
    radius = int(doc["--radius"])
    xoffset = int(doc["--xoffset"])
    yoffset = int(doc["--yoffset"])

    # construct the argument parse and parse the arguments
    # generate phase mask for LG beamimg_res_x

    def generate_phase_piston():
        image = np.zeros([img_res_y, img_res_x])+1
        image8bit = normalize_image(image)
        return image8bit

    def generate_phase_tt(x):
        xx, yy = np.mgrid[:img_res_y, :img_res_x]/img_res_y-0.5
        image = xx*x[0]+yy*x[1]
        image += 0.5
        image8bit = normalize_image(image)
        return image8bit

    def normalize_image(image):
        """normalize image to range [0, 1]
        """
        image[image < 0] = 0
        image[image > 1] = 1
        image8bit = np.round(image*255).astype('uint8')
        return image8bit

    if monitor_id >= 0:
        # create the object that handles the SLM array
        slm = slmpy.SLMdisplay(monitor=monitor_id, isImageLock=True)
        # retrieve SLM resolution (defined in monitor options)
        img_res_x, img_res_y = slm.getSize()
    else:
        img_res_x = 1920//2
        img_res_y = 1200//2

    img_center_x = img_res_x//2
    img_center_y = img_res_y//2

    x = np.linspace(0, img_res_x, img_res_x)
    y = np.linspace(0, img_res_y, img_res_y)

    # initialize image matrix
    xx, yy = np.meshgrid(x, y)

    xx = xx - img_center_x
    yy = yy - img_center_y

    # generate circular window mask
    mask_circle = np.zeros((img_res_y, img_res_x), dtype="uint8")
    cv2.circle(mask_circle, (img_center_x, img_center_y), radius, 255, -1)
    mask_circle = normalize_image(mask_circle)

    state = np.array([0.5, 0.5])

    if monitor_id < 0:
        image8bit = generate_phase_tt(state)
        print(image8bit.shape)
        image8bit = cv2.bitwise_and(image8bit, image8bit, mask=mask_circle)
        cv2.imshow('phase hologram', image8bit)
        cv2.waitKey()
    else:
        while True:
            image8bit = generate_phase_tt(state)
            image8bit = cv2.bitwise_and(image8bit, mask_circle)

            image = cv2.resize(
                image8bit, (320, 200), interpolation=cv2.INTER_CUBIC
            )
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                image, "press q to exit...", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

            # display image on window
            cv2.imshow('Phase mask', image)

            # send image to SLM
            slm.updateArray(image8bit)

            # press 'q' to exit
            key = cv2.waitKey(33)
            if key == ord('q'):
                break

        slm.close()
        cv2.destroyAllWindows()
