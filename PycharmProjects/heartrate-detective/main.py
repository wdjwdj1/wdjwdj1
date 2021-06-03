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
from objectTracking import objectTracking
from Videocapture import getfirstROI
import ICA
from kivy.clock import Clock

__version__ = "1.0.0"

def Tracking(target):
    out = []
    frame = 600
    # Frequency range for Fast-Fourier Transform
    freq_min = 0.8
    freq_max = 1.8
    cap = cv2.VideoCapture("cool.avi")
    objectTracking(cap,frame,target,out)
    # Build Laplacian video py ramid
    fps = 30
    frame_ct = len(out)
    lap_video = pyramids.build_video_pyramid(out)
    for i, video in enumerate(lap_video):
        if i == 0 or i == len(lap_video)-1:
            continue
        # Eulerian magnification with temporal FFT filtering
        result, fft, frequencies = processing.fft_filter(video, freq_min, freq_max, fps)
        lap_video[i] += result

    # Collapse laplacian pyramid to generate final video

    amplified_frames = pyramids.collapse_laplacian_video_pyramid(lap_video, frame_ct)

    for frame in amplified_frames:
        cv2.imshow("frame", frame)
        cv2.waitKey(20)

    mubaio = ICA.fastICA(amplified_frames)
    coeffs = pywt.wavedec(mubaio, 'db5', level=5) # 5阶小波分解
    yd4 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0, 0]).tolist(), 'db5')
    yd3 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0, 0]).tolist(), 'db5')
    CD = [yd3,yd4]
    final = pywt.waverec(CD, 'db5')
    fft, frequencies = processing.fft_filter1(final, freq_min, freq_max, fps)
    # Calculate heart rate
    heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)
    heart_rate = round(heart_rate,2)
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
        heartbeat = Tracking(target)
        self.heartrate = "heartrate:  "+str(heartbeat)+" bpm"

    def page_go(*args):
        App.get_running_app().screen_manager.current = "Index_page"
        App.get_running_app().screen_manager.transition.direction = 'left'

class KivyCamera(Image):
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None
        self.clock_event = None

    def start(self, capture, fps=30):
        global target
        target = []
        self.count = 0
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.video = cv2.VideoWriter("cool.avi", fourcc, 30, (640, 480))
        self.capture = capture
        self.clock_event = Clock.schedule_interval(self.update, 1.0 / fps)
        self.clock_event1 = Clock.schedule_interval(self.update1, 1.0 / fps)

    def update(self,dt):
        ret, frame = self.capture.read()
        if ret:
            if getfirstROI(frame, target) == True:
                self.video.write(frame)
                buf1 = cv2.flip(frame, 0)
                buf = buf1.tostring()
                texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.texture = texture1
                return False


    def update1(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.count = self.count + 1
            self.video.write(frame)
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture1
            if self.count == 600:
                self.back_index_1()
                self.stop()

        else:
            self.count = 0
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video = cv2.VideoWriter("cool.avi", fourcc, 30, (640, 480))
            self.clock_event = Clock.schedule_interval(self.update, 1.0 / 30)
            self.clock_event1 = Clock.schedule_interval(self.update1, 1.0 / 30)



    def stop(self):
        Clock.unschedule(self.clock_event1)
        self.capture.release()
        self.video.release()


    def back_index_1(*args):
        App.get_running_app().screen_manager.current = "Middle_Page"
        App.get_running_app().screen_manager.transition.direction = 'left'





class VideoPage(FloatLayout):
    word = StringProperty()
    def __init__(self, **kwargs):
        super(VideoPage, self).__init__(**kwargs)
        self.word = "The page is loading , please waiting..."

    def dostart(self,*largs):
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
        pages ={"Index_page":IndexPage(), "Video_page":VideoPage(),"Middle_Page":MiddlePage()}
        for item, page in pages.items():
            self.default_page = page
            #添加页面
            screen = Screen(name=item)
            screen.add_widget(self.default_page)
            #向屏幕管理器添加页面
            self.screen_manager.add_widget(screen)
        return self.screen_manager

if __name__ == "__main__":

    heartDetectApp().run()


