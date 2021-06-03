from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.graphics.texture import Texture

import cv2
import numpy as np
import pyramids
import pywt
import heartrate
import processing

from kivy.clock import Clock

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
__version__ = "1.0.0"


def getROI(self, frame):
    classifier_face = cv2.CascadeClassifier("haarcascade_frontalface_alt0.xml")
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faceRects_face = classifier_face.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
    if len(faceRects_face) > 0:
        print("ROI SUCESS")
        # 检测到人脸
        # for faceRect_face in faceRects_face:
        x, y, w, h = faceRects_face[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 获取图像x起点,y起点,宽，高
        boxh = int(float(h / 2.1))
        xmin = int(x + w / 6)
        ymin = int(y)
        boxw = int(w * 4 / 6)

        cv2.rectangle(frame, (xmin, ymin + boxh), (xmin + boxw, ymin + boxh + int(h / 5)), (0, 255, 0), 2)
        bbox = (int(xmin), int(ymin + boxh), int(boxw), int(h / 6))
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture1
        return bbox, True
    else:
        print("ROI Failed")
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture1
        return (), False


def Tracking(out):
    # Frequency range for Fast-Fourier Transform
    freq_min = 0.8
    freq_max = 1.8
    # Build Laplacian video py ramid
    fps = 30
    frame_ct = len(out)
    lap_video = pyramids.build_video_pyramid(out)
    for i, video in enumerate(lap_video):
        if i == 0 or i == len(lap_video) - 1:
            continue
        # Eulerian magnification with temporal FFT filtering
        result, fft, frequencies = processing.fft_filter(video, freq_min, freq_max, fps)
        lap_video[i] += result

    # Collapse laplacian pyramid to generate final video

    amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)

    for frame in amplified_frames:
        cv2.imshow("frame", frame)
        cv2.waitKey(20)

    average1 = []
    for frame in amplified_frames:
        average1.append(processing.Imeang(frame))

    coeffs = pywt.wavedec(average1, 'sym8', level=4)  # 4阶小波分解
    yd4 = pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0]).tolist(), 'sym8')
    yd3 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0]).tolist(), 'sym8')
    CD = [yd3, yd4]
    final = pywt.waverec(CD, 'sym8')
    fft, frequencies = processing.fft_filter1(final, freq_min, freq_max, fps)
    # Calculate heart rate
    heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)
    heart_rate = round(heart_rate, 2)
    return heart_rate


class IndexPage(FloatLayout):
    def __init__(self, **kwargs):
        super(IndexPage, self).__init__(**kwargs)

    def page_go(*args):
        App.get_running_app().screen_manager.current = "Video_page"
        App.get_running_app().screen_manager.transition.direction = 'left'


class MiddlePage(BoxLayout):
    heartrate = StringProperty()

    def __init__(self, **kwargs):
        super(MiddlePage, self).__init__(**kwargs)
        self.heartrate = ''

    def show(self):
        heartbeat = Tracking(video)
        self.heartrate = "heartrate:  " + str(heartbeat) + " bpm"

    def page_go(*args):
        App.get_running_app().screen_manager.current = "Index_page"
        App.get_running_app().screen_manager.transition.direction = 'left'


class KivyCamera(Image):
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None
        self.clock_event = None

    def start(self, capture, fps=30):
        global video
        video = []
        self.video = []
        self.capture = capture
        self.count = 0
        global tracker
        tracker = cv2.TrackerMIL_create()

        self.clock_event = Clock.schedule_interval(self.update, 1.0 / fps)
        self.clock_event1 = Clock.schedule_interval(self.update1, 1.0 / fps)
        video = self.video

    def update(self, dt):
        ok, frame = self.capture.read()
        if ok:
            if self.count == 0:
                self.bbox, judge = getROI(self, frame)
                if self.bbox != ():
                   tracker.init(frame, self.bbox)
                   self.count = 1
                else:
                    self.count = 0

        if self.count == 1:
            Clock.unschedule(self.clock_event)

    def update1(self, dt):
        ret, frame = self.capture.read()
        if ret:
            area1 = []
            # Update tracker
            ok, bbox = tracker.update(frame)
            if ok:
                self.count = self.count + 1
                # Draw bonding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                area = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                area = cv2.resize(area, (250, 250))
                area1 = np.ndarray(shape=area.shape, dtype="float")
                area1[:] = area * (1. / 255)
                self.video.append(area1)
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tostring()
                texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.texture = texture1
                self.bbox = bbox
            else:
                print("error")
                self.count = self.count + 1
                # Draw bonding box
                p1 = (int(self.bbox[0]), int(self.bbox[1]))
                p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                area = frame[int(self.bbox[1]):int(self.bbox[1] + self.bbox[3]),
                       int(self.bbox[0]):int(self.bbox[0] + self.bbox[2])]
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                area = cv2.resize(area, (250, 250))
                area1 = np.ndarray(shape=area.shape, dtype="float")
                area1[:] = area * (1. / 255)
                self.video.append(area1)
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tostring()
                texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.texture = texture1

            if self.count == 600:
                self.back_index_1()
                self.stop()
        else:
            self.video = []
            self.count = 1

    def stop(self):
        Clock.unschedule(self.clock_event1)
        self.capture.release()

    def back_index_1(*args):
        App.get_running_app().screen_manager.current = "Middle_Page"
        App.get_running_app().screen_manager.transition.direction = 'left'


class VideoPage(FloatLayout):
    word = StringProperty()

    def __init__(self, **kwargs):
        super(VideoPage, self).__init__(**kwargs)
        self.word = "The page is loading , please waiting..."

    def dostart(self, *largs):
        self.word = ""
        self.capture = cv2.VideoCapture(0)
        self.ids.cv2cam_l.start(self.capture)


class heartDetectApp(App):
    def build(self):
        self.icon = "./static/icon.ico"
        self.title = "光电容积脉搏波描记法测心率App"
        self.load_kv("./index.kv")
        self.load_kv("./middle.kv")
        self.load_kv("./video.kv")
        self.screen_manager = ScreenManager()
        pages = {"Index_page": IndexPage(), "Video_page": VideoPage(), "Middle_Page": MiddlePage()}
        for item, page in pages.items():
            self.default_page = page
            # 添加页面
            screen = Screen(name=item)
            screen.add_widget(self.default_page)
            # 向屏幕管理器添加页面
            self.screen_manager.add_widget(screen)
        return self.screen_manager


if __name__ == "__main__":
    heartDetectApp().run()
