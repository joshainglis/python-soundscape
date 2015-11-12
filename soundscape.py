# Python 3 program for soundscape generation. (C) P.B.L. Meijer 2015
# Direct port of the hificode.c C program
# Last update: October 6, 2015; released under the Creative
# Commons Attribution 4.0 International License (CC BY 4.0),
# see http://www.seeingwithsound.com/im2sound.htm for details
#
# Beware that this program runs excruciatingly slowly under Python,
# while the PyPy python JIT compiler does not (yet) support OpenCV

import math
import os
import struct
import sys
import wave

import cv2 as cv
import numpy as np

file_name = 'hificode.wav'  # User-defined parameters
min_frequency = 500  # Lowest  frequency (Hz) in soundscape
max_frequency = 5000  # Highest frequency (Hz)
sample_frequency = 44100  # Sample  frequency (Hz)
image_to_sound_conversion_time = 1.05  # Image to sound conversion time (s)
use_exponential = False  # Linear|Exponential=0|1 distribution
hifi = 1  # 8-bit|16-bit=0|1 sound quality
stereo = 1  # Mono|Stereo=0|1 sound selection
delay = 1  # Nodelay|Delay=0|1 model   (stereo=1)
relative_fade = 1  # Relative fade No|Yes=0|1  (stereo=1)
diffraction = 1  # Diffraction No|Yes=0|1    (stereo=1)
use_b_spline = 1  # Rectangular|B-spline=0|1 time window
gray_levels = 0  # 16|2-level=0|1 gray format in P[][]
use_camera = 1  # Use OpenCV camera input No|Yes=0|1
use_screen = 1  # Screen view for debugging No|Yes=0|1


class Soundscape(object):
    IR = 0
    IA = 9301
    IC = 49297
    IM = 233280
    TwoPi = 6.283185307179586476925287
    WHITE = 1.00
    BLACK = 0.00

    def __init__(self, file_name='hificode.wav', min_frequency=500, max_frequency=5000, sample_frequency=44100,
                 image_to_sound_conversion_time=1.05, is_exponential=False, hifi=True, stereo=True, delay=True,
                 relative_fade=True, diffraction=True, use_b_spline=True, gray_levels=16, use_camera=True,
                 use_screen=True):
        """

        :param file_name:
        :type file_name: str
        :param min_frequency:
        :type min_frequency: int
        :param max_frequency:
        :type max_frequency: int
        :param sample_frequency:
        :type sample_frequency: int
        :param image_to_sound_conversion_time:
        :type image_to_sound_conversion_time: float
        :param is_exponential:
        :type is_exponential: bool
        :param hifi:
        :type hifi: bool
        :param stereo:
        :type stereo: bool
        :param delay:
        :type delay: bool
        :param relative_fade:
        :type relative_fade: bool
        :param diffraction:
        :type diffraction: bool
        :param use_b_spline:
        :type use_b_spline: bool
        :param gray_levels:
        :type gray_levels: int
        :param use_camera:
        :type use_camera: bool
        :param use_screen:
        :type use_screen: bool
        :return:
        :rtype:
        """

        self.file_name = file_name
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.sample_frequency = sample_frequency
        self.image_to_sound_conversion_time = image_to_sound_conversion_time
        self.is_exponential = is_exponential
        self.hifi = hifi
        self.stereo = stereo
        self.delay = delay
        self.relative_fade = relative_fade
        self.diffraction = diffraction
        self.use_b_spline = use_b_spline
        self.gray_levels = gray_levels
        self.use_camera = use_camera
        self.use_screen = use_screen
        self.hist = (1 + self.hifi) * (1 + self.stereo)

        if use_camera:
            self.num_columns = 176
            self.num_rows = 64
        else:
            self.num_columns = 64
            self.num_rows = 64

        self.k = 0
        self.b = 0
        self.num_frames = 2 * int(0.5 * self.sample_frequency * self.image_to_sound_conversion_time)
        self.frames_per_column = int(self.num_frames / self.num_columns)
        self.sso = 0 if self.hifi else 128
        self.ssm = 32768 if self.hifi else 128
        self.scale = 0.5 / math.sqrt(self.num_rows)
        self.dt = 1.0 / self.sample_frequency
        self.v = 340.0  # v = speed of sound (m/s)
        self.hs = 0.20  # hs = characteristic acoustical size of head (m)
        self.w = np.arange(self.num_rows, dtype=np.float)
        self.phi0 = np.zeros(self.num_rows, dtype=np.float)
        self.A = np.zeros((self.num_columns, self.num_rows), dtype=np.uint8)

# Coefficients used in rnd()
IR = 0
IA = 9301
IC = 49297
IM = 233280

TwoPi = 6.283185307179586476925287
HIST = (1 + hifi) * (1 + stereo)
WHITE = 1.00
BLACK = 0.00

if use_camera:
    num_columns = 176
    num_rows = 64
else:
    num_columns = 64
    num_rows = 64

# if gray_levels:
# else:

try:
    # noinspection PyUnresolvedReferences
    import winsound
except ImportError:
    def playsound(frequency, duration):
        # sudo dnf -y install beep
        os.system('beep -f %s -l %s' % (frequency, duration))
else:
    def playsound(frequency, duration):
        winsound.Beep(frequency, duration)


# def playSound(file):
#     if sys.platform == "win32":
#         winsound.PlaySound(file, winsound.SND_FILENAME)  # Windows only
#         # os.system('start %s' %file)                    # Windows only
#     elif sys.platform.startswith('linux'):
#         print("No audio player called for Linux")
#     else:
#         print("No audio player called for your platform")


def wi(file_object, i):
    b0 = int(i % 256)
    b1 = int((i - b0) / 256)
    file_object.write(struct.pack('B', b0 & 0xff))
    file_object.write(struct.pack('B', b1 & 0xff))


def wl(fp, l):
    i0 = l % 65536
    i1 = (l - i0) / 65536
    wi(fp, i0)
    wi(fp, i1)


def rnd():
    global IR, IA, IC, IM
    IR = (IR * IA + IC) % IM
    return IR / (1.0 * IM)


def main():
    current_frame = 0
    b = 0
    num_frames = 2 * int(0.5 * sample_frequency * image_to_sound_conversion_time)
    frames_per_column = int(num_frames / num_columns)
    sso = 0 if hifi else 128
    ssm = 32768 if hifi else 128
    scale = 0.5 / math.sqrt(num_rows)
    dt = 1.0 / sample_frequency
    v = 340.0  # v = speed of sound (m/s)
    hs = 0.20  # hs = characteristic acoustical size of head (m)
    w = np.arange(num_rows, dtype=np.float)
    phi0 = np.zeros(num_rows)
    A = np.zeros((num_columns, num_rows), dtype=np.uint8)
    # w = [0 for i in range(num_rows)]
    # phi0 = [0 for i in range(num_rows)]
    # A = [[0 for j in range(num_columns)] for i in range(num_rows)]  # num_rows x num_columns pixel matrix

    # Set lin|exp (0|1) frequency distribution and random initial phase
    freq_ratio = max_frequency / float(min_frequency)
    if use_exponential:
        w = TwoPi * min_frequency * np.power(freq_ratio, w / (num_rows - 1))
        for i in range(0, num_rows):
            w[i] = TwoPi * min_frequency * pow(freq_ratio, 1.0 * i / (num_rows - 1))
    else:
        for i in range(0, num_rows):
            w[i] = TwoPi * min_frequency + TwoPi * (max_frequency - min_frequency) * i / (
        num_rows - 1)
    for i in range(0, num_rows): phi0[i] = TwoPi * rnd()

    cam_id = 0  # First available OpenCV camera
    # Optionally override ID from command line parameter: python hificode_OpenCV.py cam_id
    if len(sys.argv) > 1:
        cam_id = int(sys.argv[1])

    try:
        # noinspection PyArgumentList
        cap = cv.VideoCapture(cam_id)
        if not cap.isOpened():
            raise ValueError('camera ID')
    except ValueError:
        print("Could not open camera", cam_id)
        raise

    # Setting standard capture size, may fail; resize later
    cap.read()  # Dummy read needed with some devices
    # noinspection PyUnresolvedReferences
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 176)
    # noinspection PyUnresolvedReferences
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 144)
    if use_screen:  # Screen views only for debugging
        cv.namedWindow('Large', cv.WINDOW_AUTOSIZE)
        cv.namedWindow('Small', cv.WINDOW_AUTOSIZE)

    key = 0
    while key != 27:  # Escape key
        ret, frame = cap.read()

        if not ret:
            # Sometimes initial frames fail
            print("Capture failed\n")
            key = cv.waitKey(100)
            continue

        tmp = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if frame.shape[1] != num_rows or frame.shape[0] != num_columns:
            # cv.resize(tmp, gray, Size(num_columns,num_rows))
            gray = cv.resize(tmp, (num_columns, num_rows), interpolation=cv.INTER_AREA)
        else:
            gray = tmp

        if use_screen:  # Screen views only for debugging
            cv.imwrite("hificodeLarge.jpg", frame)
            cv.imshow('Large', frame)
            cv.moveWindow('Large', 20, 20)
            cv.imwrite("hificodeSmall.jpg", gray)
            cv.imshow('Small', gray)
            cv.moveWindow('Small', 220, 20)

        key = cv.waitKey(10)

        if use_camera:  # Set live camera image
            mVal = gray / 16
            A[mVal == 0] = 0
            A[mVal > 0] = np.power(10.0, (mVal[mVal > 0] - 15) / 10.0)

        # Write 8/16-bit mono/stereo .wav file
        with open(file_name, 'wb') as nf:
            fp = wave.open(nf)
            fp.setnchannels(2 if stereo else 1)
            fp.setframerate(sample_frequency)
            fp.setsampwidth(2 if hifi else 1)


            tau1 = 0.5 / w[num_rows - 1]
            tau2 = 0.25 * (tau1 * tau1)
            y = yl = yr = z = zl = zr = 0.0

            while current_frame < num_frames and not stereo:
                if use_b_spline:
                    q = 1.0 * (current_frame % frames_per_column) / (frames_per_column - 1)
                    q2 = 0.5 * q * q
                j = int(current_frame / frames_per_column)
                j = num_columns - 1 if j > num_columns - 1 else j
                s = 0.0
                t = current_frame * dt
                if current_frame < num_frames / (5 * num_columns):
                    s = (2.0 * rnd() - 1.0) / scale  # "click"
                else:
                    for i in range(0, num_rows):
                        if use_b_spline:  # Quadratic B-spline for smooth C1 time window
                            if j == 0:
                                a = (1.0 - q2) * A[i][j] + q2 * A[i][j + 1]
                            elif j == num_columns - 1:
                                a = (q2 - q + 0.5) * A[i][j - 1] + (0.5 + q - q2) * A[i][j]
                            else:
                                a = (q2 - q + 0.5) * A[i][j - 1] + (0.5 + q - q * q) * A[i][j] + q2 * A[i][j + 1]
                        else:
                            a = A[i][j]  # Rectangular time window
                        s += a * math.sin(w[i] * t + phi0[i])

                yp = y
                y = tau1 / dt + tau2 / (dt * dt)
                y = (s + y * yp + tau2 / dt * z) / (1.0 + y)
                z = (y - yp) / dt
                l = sso + 0.5 + scale * ssm * y  # y = 2nd order filtered s
                if l >= sso - 1 + ssm: l = sso - 1 + ssm
                if l < sso - ssm: l = sso - ssm
                ss = int(l) & 0xFFFFFFFF  # Make unsigned int
                if hifi:
                    wi(fp, ss)
                else:
                    fp.write(struct.pack('B', ss & 0xff))
                current_frame += 1
            while current_frame < num_frames and stereo:
                if use_b_spline:
                    q = 1.0 * (current_frame % frames_per_column) / (frames_per_column - 1)
                    q2 = 0.5 * q * q
                j = int(current_frame / frames_per_column)
                j = num_columns - 1 if j > num_columns - 1 else j
                r = 1.0 * current_frame / (num_frames - 1)  # Binaural attenuation/delay parameter
                theta = (r - 0.5) * TwoPi / 3
                x = 0.5 * hs * (theta + math.sin(theta))
                tl = tr = current_frame * dt
                if delay:
                    tr += x / v  # Time delay model
                x = abs(x)
                sl = sr = 0.0
                hrtfl = hrtfr = 1.0
                for i in range(0, num_rows):
                    if diffraction:
                        # First order frequency-dependent azimuth diffraction model
                        hrtf = 1.0 if (TwoPi * v / w[i] > x) else TwoPi * v / (x * w[i])
                        if theta < 0.0:
                            hrtfl = 1.0
                            hrtfr = hrtf
                        else:
                            hrtfl = hrtf
                            hrtfr = 1.0
                    if relative_fade:
                        # Simple frequency-independent relative fade model
                        hrtfl *= (1.0 - 0.7 * r)
                        hrtfr *= (0.3 + 0.7 * r)
                    if use_b_spline:
                        if j == 0:
                            a = (1.0 - q2) * A[i][j] + q2 * A[i][j + 1]
                        elif j == num_columns - 1:
                            a = (q2 - q + 0.5) * A[i][j - 1] + (0.5 + q - q2) * A[i][j]
                        else:
                            a = (q2 - q + 0.5) * A[i][j - 1] + (0.5 + q - q * q) * A[i][j] + q2 * A[i][j + 1]
                    else:
                        a = A[i][j]
                    sl += hrtfl * a * math.sin(w[i] * tl + phi0[i])
                    sr += hrtfr * a * math.sin(w[i] * tr + phi0[i])

                sl = (2.0 * rnd() - 1.0) / scale if (current_frame < num_frames / (5 * num_columns)) else sl  # Left "click"
                if tl < 0.0: sl = 0.0;
                if tr < 0.0: sr = 0.0;
                ypl = yl
                yl = tau1 / dt + tau2 / (dt * dt)
                yl = (sl + yl * ypl + tau2 / dt * zl) / (1.0 + yl)
                zl = (yl - ypl) / dt
                ypr = yr
                yr = tau1 / dt + tau2 / (dt * dt)
                yr = (sr + yr * ypr + tau2 / dt * zr) / (1.0 + yr)
                zr = (yr - ypr) / dt
                l = sso + 0.5 + scale * ssm * yl
                if l >= sso - 1 + ssm: l = sso - 1 + ssm
                if l < sso - ssm: l = sso - ssm
                ss = int(l) & 0xFFFFFFFF
                # Left channel
                if hifi:
                    wi(fp, ss)
                else:
                    fp.write(struct.pack('B', ss & 0xff))
                l = sso + 0.5 + scale * ssm * yr
                if l >= sso - 1 + ssm: l = sso - 1 + ssm
                if l < sso - ssm: l = sso - ssm
                ss = int(l) & 0xFFFFFFFF
                # Right channel
                if hifi:
                    wi(fp, ss)
                else:
                    fp.write(struct.pack('B', ss & 0xff))
                current_frame += 1

            fp.close()

            playSound("hificode.wav")  # Play the soundscape

            current_frame = 0  # Reset sample count

    cap.release()
    cv.destroyAllWindows()
    return 0


main()
