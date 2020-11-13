#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import time
# from sklearn.metrics.pairwise import cosine_similarity
from argparse import ArgumentParser
import traceback
import nanocamera as nano
from PID import PID
# import select
# import v4l2capture
from camera import Camera

def draw_lanes(img, lines):
    distance_x = None
    # a. 劃分左右車道
    left_lines, right_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # print(x1, y1, x2, y2)
        sx = (x2 - x1)
        if sx <= 0:  # 跳過x相差為0,避免除0
            # print('跳過x相差為0,避免除0')
            continue

        k = (y2 - y1) / sx  # 斜率
        if k < 0:
            left_lines.append(line)
        else:
            right_lines.append(line)

    # print('left_lines:', len(left_lines), 'right_lines:', len(right_lines))
    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        # print('沒有數值')
        if len(left_lines) > 0:
            return distance_x, True # 向右轉
        else:
            return distance_x, False # 向左轉

    # b. 清理異常數據，迭代計算斜率均值，排除掉與差值差異較大的數據
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    # print(left_lines)
    # print(right_lines)

    # c. 得到左右車道線點的集合，擬合直線
    # left_points = [(line[0], line[1]) for line in left_lines]
    # left_points = left_points + [(line[2], line[3]) for line in left_lines]
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    minY = min(left_points[1])
    # print(left_points)

    # right_points = [(line[0], line[1]) for line in right_lines]
    # right_points = right_points + [(line[2], line[3]) for line in right_lines]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
    minY = min(minY, min(right_points[1]))
    # print(minY)

    minY = 80
    # 最小二乘法擬合
    left_results = least_squares_fit(left_points, minY, img.shape[1])
    right_results = least_squares_fit(right_points, minY, img.shape[1])

    if args.draw_lines:
        cv2.line(img, (left_results[0][0], left_results[0][1]), (left_results[1][0], left_results[1][1]), (0, 0, 255), 2)      # 左線
        cv2.line(img, (right_results[0][0], right_results[0][1]), (right_results[1][0], right_results[1][1]), (0, 0, 255), 2)  # 右線
        # print('left_results:', left_results)
        # print('right_results:', right_results)

        # 實際中心線
        car_center1 = [int((left_results[1][0] + right_results[1][0]) / 2),
                       int((left_results[1][1] + right_results[1][1]) / 2)]  # 下中的一個點
        car_center2 = [int((left_results[0][0] + right_results[0][0]) / 2),
                       int((left_results[0][1] + right_results[0][1]) / 2)]  # 上中一個點
        car_center1[1] = min(car_center1[1], img.shape[0])  # 大於高度設定成高度
        car_center2[1] = min(car_center2[1], img.shape[0])  # 大於高度設定成高度
#         print('car_center1:', car_center1)
#         print('car_center2:', car_center2)
        cv2.line(img, tuple(car_center1), tuple(car_center2), (0, 0, 255), 2) # 實際中心線

        view_center1 = (int(img.shape[1] / 2), int(img.shape[0]))  # 下中的一個點
        view_center2 = (int(img.shape[1] / 2), 0)  # 上中一個點

        cv2.line(img, view_center1, view_center2, (0, 0, 255), 2)  # Camera中心線上下
        # cv2.line(img, (0, int(img.shape[0]/2)), (int(img.shape[1]), int(img.shape[0]/2)) , (0, 255, 255), 2) # Camera中心線左右

        car_center_point = [int((car_center1[0] + car_center2[0]) / 2), int((car_center1[1] + car_center2[1]) / 2)]
        camera_center_point = [int(img.shape[1] / 2), int(img.shape[0] / 2)]
        # cv2.circle(img, tuple(car_center_point), 2, (0, 255, 255), -1)  # 實際中心點
        # cv2.circle(img, tuple(camera_center_point), 2, (0, 255, 255), -1)  # Camera中心點

        distance_x = car_center_point[0] - camera_center_point[0]
#         cv2.putText(img, 'DST: {}'.format(distance_x), (10, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1,
#                     cv2.LINE_AA)
# #         cos = cosine_similarity([list(car_center1) + list(car_center2)], [list(view_center1) + list(view_center2)]).squeeze()
# #         cv2.putText(img, '{:.2f}'.format(cos), (10, 300), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

    # 注意這裡點的順序
    #     print(left_results)
    #     print(right_results)

    # vtxs = np.array([[left_results[0], right_results[0], left_results[1], right_results[1]]])
    # #     print("-------------------")
    # # d. 填充車道區域
    # #     cv2.fillPoly(img, vtxs, (100, 200, 100))
    return distance_x, True


def clean_lines(lines, threshold):
    # 迭代計算斜率均值，排除掉與差值差異較大的數據
    # slope = [(line[3] - line[1]) / (line[2] - line[0]) for line in lines]
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]

    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def least_squares_fit(point_list, ymin, ymax):
    # 最小二乘法擬合
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]

    # polyfit第三個參數為擬合多項式的階數，所以1代表線性
    fits = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fits)  # 獲取擬合的結果

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))
    # print(xmin, ymin, xmax, ymax)
    return [[xmin, ymin], [xmax, ymax]]


def gstreamer_pipeline(capture_width=640, capture_height=480, display_width=640, display_height=480, framerate=30,
                       flip_method=0):
    # return ('nvarguscamerasrc ! '
    #         'video/x-raw(momory:NVMM), '
    #         'width=(int)%d, height=(int)%d, '
    #         'format=(string)NV12, framerate=(fraction)%d/1 ! '
    #         'nvvidconv flip-method=%d ! '
    #         'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    #         'videoconvert ! '
    #         'video/x-raw, format=(string)BGR ! appsink' % (capture_width, capture_height, framerate, flip_method, display_width, display_height))
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=%d, height=%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'queue max-size-buffers=1 max-size-bytes=1 max-size-time=1 ! '
            'queue max-size-time=1 min-threshold-time=1 ! '
            'appsink' % (capture_width, capture_height, framerate, display_width, display_height))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-w", "--write_video", default='result.avi', type=str)

    parser.add_argument("-dt", "--delay_time", default=0, type=int)
    parser.add_argument("-rt", "--runtime", default=-1, type=int)
    parser.add_argument("-c", "--camera", default=0, type=int)
    parser.add_argument("-disca", "--display_camera", default=True, type=bool)

    parser.add_argument("-d", "--draw_lines", default=True, type=bool)
    parser.add_argument("-f", "--draw_fps", default=True, type=bool)

    parser.add_argument("-ctl", "--controller", default=True, type=bool)
    parser.add_argument("-fwd", "--forward", default=0.5, type=float)

    args = parser.parse_args()

    # cap = cv2.VideoCapture('../demo.avi')
    # cap = cv2.VideoCapture(args.camera)
    # cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=320, capture_height=240, flip_method=0, framerate=5), cv2.CAP_GSTREAMER)

    # video = v4l2capture.Video_device("/dev/video0")
    # size_x, size_y = video.set_format(640, 480, fourcc='MJPG')
    # print("device chose {0}x{1} res".format(size_x, size_y))
    # # Create a buffer to store image data in. This must be done before
    # # calling 'start' if v4l2capture is compiled with libv4l2. Otherwise
    # # raises IOError.
    # video.create_buffers(30)
    # # Send the buffer to the device. Some devices require this to be done
    # # before calling 'start'.
    # video.queue_all_buffers()
    # video.start()
    # # The rest is easy :-)
    # image_data = video.read_and_queue()
    # frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.cv.CV_LOAD_IMAGE_COLOR)

    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # cap.set(cv2.CAP_PROP_FPS, 20)

    camera = Camera()
    # camera = Camera.instance()
    # camera = nano.Camera(camera_type=2, device_id=0, flip=0, width=640, height=480, fps=30)

    if args.write_video:
        out = cv2.VideoWriter(args.write_video, cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (320, 480))

    # 控制方向
    if args.controller:
        from jetbot import Robot
        robot = Robot()

        # 100像素點當最大值
        pid = PID(0.1, 0.01, 0.00015)
        # pid = PID(0.02, 0.01, 0.00015)
        # left_default, right_default = 0.34, 0.34
        left_default, right_default = 0.35, 0.35
        # robot.set_motors(left_default, right_default)
        # time.sleep(0.1)
        # left_default, right_default = 0.28, 0.28

    sys_start = time.time()
    value_left_pre = 0.0
    value_right_pre = 0.0
    distance_x_pre = 0.0

    while(True):
        start = time.time()

        # ret, frame = cap.read()
        # if ret:
        # if camera.isReady():
        #     frame = camera.read()

        if True:
            # # Wait for the device to fill the buffer.
            # select.select((video,), (), ())
            # image_data = video.read_and_queue()
            # frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.cv.CV_LOAD_IMAGE_COLOR)
            frame = camera.value

            # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # f1 = cv2.bilateralFilter(gray, 9, 75, 75)  # 去噪音(模糊)
            # erosion = cv2.erode(f1, np.ones((2, 2), np.uint8), iterations=1)
            # dilation = cv2.dilate(erosion, np.ones((2, 2), np.uint8), iterations=1)
            # _, th1 = cv2.threshold(dilation, 180, 225, cv2.THRESH_TOZERO)
            ## canny = cv2.cuda.Canny(dilation, 100, 200)

            gpu_img = cv2.cuda_GpuMat(frame)
            gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2GRAY)
            f1 = cv2.cuda.bilateralFilter(gray, 9, 75, 75)  # 去噪音(模糊)
            openIt = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, f1.type(), np.eye(2)).apply(f1)
            _, th1 = cv2.cuda.threshold(openIt, 180, 225, cv2.THRESH_TOZERO)
            # th1 = cv2.cuda.createCannyEdgeDetector(0, 100).detect(th1)

            # lines = cv2.cuda.createHoughSegmentDetector(1.0, np.pi / 180.0, 150, 5).detect(th1).download()
            # if lines is not None:
            #     # print(lines.shape)
            #     # try:
            #     #     for line in lines[0]:
            #     #         x1, y1, x2, y2 = line
            #     #         # if y1 < 50:
            #     #         #     line[1] = 150
            #     #         # if y2 < 50:
            #     #         #     line[3] = 150
            #     #         cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 線條
            #     # except:
            #     #     traceback.print_exc()
            #     #     pass
            #     distance_x = draw_lanes(frame, lines)

            th1 = th1.download()
            lines = cv2.HoughLinesP(th1, 1, np.pi / 180, 150, 5)
            distance_x = None
            if lines is not None:
                # print(lines.shape)
                try:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # if y1 < 50:
                        #     line[1] = 150
                        # if y2 < 50:
                        #     line[3] = 150
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  # 線條
                except:
                    traceback.print_exc()
                    pass
                distance_x, direction = draw_lanes(frame, lines)

            if distance_x is not None:
                distance_x_pre = distance_x
                output = pid.update(distance_x)
                turn = output / 100.0
                value_left = left_default - turn / 2.0
                value_right = right_default + turn / 2.0
                value_left_pre = value_left
                value_right_pre = value_right

                ratio = -0.005 * abs(distance_x) + 1
                value_left *= ratio
                value_right *= ratio

                # print('turn:', turn, 'value_left:', value_left, 'value_right:', value_right)
                cv2.putText(frame, 'RAT: {:.0f}'.format(ratio), (10, 100), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'DST: {:.0f}'.format(distance_x), (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255),
                            1, cv2.LINE_AA)
                cv2.putText(frame, 'OUT: {:.0f}'.format(output), (10, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255),
                            1, cv2.LINE_AA)
                cv2.putText(frame, 'Turn: {:.3f}'.format(turn), (10, 350), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255),
                            1, cv2.LINE_AA)
                cv2.putText(frame, 'Left: {:.3f}'.format(value_left), (10, 400), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Right: {:.3f}'.format(value_right), (10, 450), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)

                # if time.time() - sys_start > args.delay_time:
                robot.set_motors(value_left, value_right)
            else:
                # output = pid.update(distance_x_pre)

                # cv2.putText(frame, 'Left: {:.3f}'.format(robot.right_motor.value), (10, 300), cv2.FONT_HERSHEY_TRIPLEX, 1,
                #             (0, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(frame, 'Right: {:.3f}'.format(robot.left_motor.value), (10, 350), cv2.FONT_HERSHEY_TRIPLEX, 1,
                #             (0, 255, 255), 1, cv2.LINE_AA)

                # robot.set_motors(value_right_pre * 0.5, value_left_pre * 0.5)
                if direction: # 向右轉
                    robot.set_motors(value_left_pre * 0.8, 0)
                else: # 向左轉
                    robot.set_motors(0, value_left_pre * 0.8)

                cv2.putText(frame, 'Left: {:.3f}'.format(value_right_pre), (10, 400), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Right: {:.3f}'.format(value_left_pre), (10, 450), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (0, 255, 255), 1, cv2.LINE_AA)

            if args.display_camera:
                if args.draw_fps:
                    end = time.time()
                    # 計算FPS
                    fps = 1 / (end - start)
                    cv2.putText(frame, 'FPS: {:.0f}'.format(fps), (10, 250), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255),
                                1, cv2.LINE_AA)
                    # print("\rFPS: {:.0f}".format(fps), end='\n')
                # cv2.imshow('frame', frame)

            if args.write_video:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
                break
            if args.runtime != -1 and time.time() - sys_start >= args.runtime:
                break
        else:
            break

    if args.controller:
        robot.stop()
    # cap.release()
    # camera.release()
    camera.stop()
    if args.write_video:
        out.release()
