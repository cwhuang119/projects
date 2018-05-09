import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os
from matplotlib import pyplot as plt

def img_capture(video_path,fn,save_path,obj,threshold,sec_per_frame,crop):
    option = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolov2.weights',
        'threshold': threshold,
        'gpu': 1.0
    }
    tfnet = TFNet(option)
    video = video_path + fn
    capture = cv2.VideoCapture(video)
    frame_count = 0
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while capture.isOpened():
        res, frame = capture.read()
        frame_count+=1
        if frame_count%300==0:

            print('progressing:{} %'.format(str((frame_count/video_length)*100)[:5]))
            print('----------')
        if res:
            if frame_count%sec_per_frame == 0  :
                results = tfnet.return_predict(frame)
                for result in results:
                    if result['label']==obj:
                        x1 = result['topleft']['x']
                        y1 = result['topleft']['y']
                        x2 = result['bottomright']['x']
                        y2 = result['bottomright']['y']
                        width = x2-x1
                        hight = y2-y1
                        #min = str(int(capture.get(cv2.CAP_PROP_POS_MSEC)/1000/60))
                        sec = str(int(capture.get(cv2.CAP_PROP_POS_MSEC)/1000))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        # print(min,sec,result['label'])
                        if crop == True:
                            cv2.imwrite(save_path+'/'+sec+'.jpg',frame[y1:y2,x1:x2,:])
                        else :
                            cv2.imwrite(save_path+'/'+sec+'.jpg',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        capture.release()
                        cv2.destroyAllWindows()
                        break
        else:
            capture.release()
            cv2.destroyAllWindows()
            break
if __name__ =='__mian__':
    img_capture(video_path,fn,save_path,obj,threshold,sec_per_frame)
