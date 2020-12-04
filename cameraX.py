import traitlets
from traitlets.config.configurable import SingletonConfigurable
import atexit
import cv2
import threading
import numpy as np


class CameraX(SingletonConfigurable):
    
    value = traitlets.Any()
    
    # config
    # width = traitlets.Integer(default_value=224).tag(config=True)
    # height = traitlets.Integer(default_value=224).tag(config=True)
    # fps = traitlets.Integer(default_value=21).tag(config=True)
    # capture_width = traitlets.Integer(default_value=3280).tag(config=True)
    # capture_height = traitlets.Integer(default_value=2464).tag(config=True)
    width = traitlets.Integer(default_value=224).tag(config=True)
    height = traitlets.Integer(default_value=224).tag(config=True)
    fps = traitlets.Integer(default_value=30).tag(config=True)
    capture_width = traitlets.Integer(default_value=1280).tag(config=True)
    capture_height = traitlets.Integer(default_value=720).tag(config=True)
    cap = None
    isVideo = False
    
    def __init__(self, *args, **kwargs):
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        super(CameraX, self).__init__(*args, **kwargs)
    
        try:
            # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not kwargs.get('cap'):
                self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            else:
                self.isVideo = True
                self.cap = kwargs['cap']
            
            re, image = self.cap.read()

            if not re:
                raise RuntimeError('Could not read image from camera.')

            self.value = image
            self.start()
            
            print('Shape:', self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 'FPS:', self.cap.get(cv2.CAP_PROP_FPS))
            print('Buffer:', self.cap.get(cv2.CAP_PROP_BUFFERSIZE))
        except:
            self.stop()
            raise RuntimeError('Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):
        while True:
            re, image = self.cap.read()
            if re:
                self.value = image
            else:
                if self.isVideo:
                    self.value = None
                break
                
    def _gst_str(self):
#         return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
#                 self.capture_width, self.capture_height, self.fps, self.width, self.height)
        return ('nvarguscamerasrc ! ' # num-buffers=1
            'video/x-raw(memory:NVMM), '
            'width=%d, height=%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'queue max-size-buffers=1 max-size-bytes=1 max-size-time=1 ! '
            'queue max-size-time=1 min-threshold-time=1 ! '
            'appsink' % (self.capture_width, self.capture_height, self.fps, self.width, self.height)) # fakesink sync=true

    def start(self):
        if not self.cap.isOpened():
            print('start cap not isOpened')
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'thread'):
            self.thread.join()
            
    def restart(self):
        self.stop()
        self.start()