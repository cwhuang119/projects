
import math
import numpy as np
import cv2
import tensorflow as tf
from face_recognition import facenet
from face_recognition import detect_face
from scipy import misc
from scipy.spatial import distance
from face_recognition import visualization_utils as vis_utils
import pickle
import os


def detect_person(video_path,fn,save_path,sec_per_frame,name):
    minsize = 20  # 最小的臉部的大小
    threshold = [0.6, 0.7, 0.7]  # 三個網絡(P-Net, R-Net, O-Net)的閥值
    factor = 0.709  # scale factor
    margin = 44 # 在裁剪人臉時的邊框margin

    modeldir =  'face_recognition/model/facenet/20170512-110547/20170512-110547.pb' #'/..Path to Pre-trained model../20170512-110547/20170512-110547.pb'
    facenet.load_model(modeldir)

    DATA_PATH = 'face_recognition/custom_data'
        # "人臉embedding"的資料
    with open(os.path.join(DATA_PATH,'lfw_emb_features.pkl'), 'rb') as emb_features_file:
        emb_features =pickle.load(emb_features_file)

            # "人臉embedding"所對應的標籤(label)的資料
    with open(os.path.join(DATA_PATH,'lfw_emb_labels.pkl'), 'rb') as emb_lables_file:
        emb_labels =pickle.load(emb_lables_file)

    # "標籤(label)對應到人臉名稱的字典的資料
    with open(os.path.join(DATA_PATH,'lfw_emb_labels_dict.pkl'), 'rb') as emb_lables_dict_file:
        emb_labels_dict =pickle.load(emb_lables_dict_file)
        #create emb_dict
    emb_dict = {}
    for feature,label in zip(emb_features, emb_labels):
            # 檢查key有沒有存在
            # print(label)
        if label in emb_dict:
            emb_dict[label].append(feature)   #接著每項的第一個label產生的
        else:
            emb_dict[label] = [feature]       #每個label的第一個是這樣產生的

    def is_same_person(face_emb, face_label, threshold=1.1):
        emb_distances = []
        emb_features = emb_dict[face_label]
        for i in range(len(emb_features)):
            emb_distances.append(calc_dist(face_emb, emb_features[i]))
        # 取得平均值
        if np.mean(emb_distances) > threshold: # threshold <1.1 代表兩個人臉非常相似
            return False,np.mean(emb_distances)
        else:
            return True,np.mean(emb_distances)


    # 計算兩個人臉特徵（Facenet Embedding 128 bytes vector)的歐式距離
    def calc_dist(face1_emb, face2_emb):
        return distance.euclidean(face1_emb, face2_emb)

    # 創建Tensorflow Graph物件
    tf_g = tf.Graph().as_default()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    # 創建Tensorflow Session物件

    tf_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    # 把這個Session設成預設的session
    tf_sess.as_default()


    # 載入MTCNN模型 (偵測人臉位置)
    pnet, rnet, onet = detect_face.create_mtcnn(tf_sess,'face_recognition/model/mtcnn/' )
    classifier_filename = 'face_recognition/custom_data/lfw_svm_classifier.pkl'
    with open(classifier_filename, 'rb') as svm_model_file:
        (face_svc_classifier, face_identity_names) = pickle.load(svm_model_file)
        HumanNames = face_identity_names
    print('load classifier file-> %s' % classifier_filename)
    print(face_svc_classifier)
    # print(face_identity_names)

    video = video_path + fn
    capture = cv2.VideoCapture(video)
    frame_count = 0
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('length:',video_length)
    while capture.isOpened():
        res, frame = capture.read()
        frame_count+=1
        # print(frame_count)
        if res:

            video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            percentage = str((frame_count/video_length)*100)
            if frame_count%300==0:
                print('progressing:{} %'.format(percentage[:5]))
                print('')
                print('----------')
            if frame_count%sec_per_frame == 0  :

                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                #numbers of faces detected
                nrof_faces = bounding_boxes.shape[0]
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                if nrof_faces > 0: # 如果有偵測到人臉
                    det = bounding_boxes[:, 0:4].astype(int) # 取出邊界框座標
                    img_size = np.asarray(frame.shape)[0:2] # 原圖像大小 (height, width)
                    # 人臉圖像前處理的暫存
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)
                    # 步驟 #2.擷取人臉特徵
                    for i in range(nrof_faces):
                        # print("faces#{}".format(i))
                        emb_array = np.zeros((1, embedding_size))
                        x1 = bb[i][0] = det[i][0]
                        y1 = bb[i][1] = det[i][1]
                        x2 = bb[i][2] = det[i][2]
                        y2 = bb[i][3] = det[i][3]
                        input_image_size = 160
                        image_size = 182

                        try:
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                # print('face is out of range!')
                                continue
                            # **人臉圖像的前處理 **

                            # 根據邊界框的座標來進行人臉的裁剪
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}

                            # 進行臉部特0.5徵308 257擷取
                            emb_array[0, :] = tf_sess.run(embeddings, feed_dict=feed_dict)
                            # 步驟 #3.進行人臉識別分類
                            face_id_idx = face_svc_classifier.predict(emb_array)
                            face_id_name = HumanNames[int(face_id_idx)]
                            resu,dis = is_same_person(emb_array, int(face_id_idx), threshold=1.0)
                            # print(face_id_name)
                            # print(dis)

                            if resu:
                                    face_id_name = HumanNames[int(face_id_idx)]
                                    # print(dis)
                                    # print(face_id_name)
                                    if face_id_name==name:
                                        min = str(int(capture.get(cv2.CAP_PROP_POS_MSEC)/1000/60))
                                        sec = str(int(capture.get(cv2.CAP_PROP_POS_MSEC)/1000))
                                    if not os.path.exists(save_path):
                                        os.makedirs(save_path)
                                         # print(min,sec,result['label'])
                                    cv2.imwrite(save_path+'/'+sec+'.jpg',frame)


                        except:
                            pass

                # det = bounding_boxes[:, 0:4].astype(int) # 取出邊界框座標
                # bb = np.zeros((nrof_faces,4), dtype=np.int32)
                #
                # # print(nrof_faces)
                # if nrof_faces > 0:
                #     for box in bounding_boxes:
                #         conf = box[4]
                #         y1 = int(box[0])
                #         x1 = int(box[1])
                #         y2 = int(box[2])
                #         x2 = int(box[3])
                #         #capture face
                #         face_frame = frame[x1:x2,y1:y2,:]
                #         #tf sess create
                #         images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                #         embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                #         phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                #         embedding_size = embeddings.get_shape()[1]
                #
                #         # load svm model
                #         classifier_filename = 'face_recognition/custom_data/lfw_svm_classifier.pkl'
                #         with open(classifier_filename, 'rb') as svm_model_file:
                #             (face_svc_classifier, face_identity_names) = pickle.load(svm_model_file)
                #             HumanNames = face_identity_names
                #         # print(HumanNames)
                #         input_image_size = 160
                #         image_size = 182
                #
                #         emb_array = np.zeros((1, embedding_size))
                #         bb = np.zeros((4,), dtype=np.int32)
                #         bb[0] = det[0]
                #         bb[1] = det[1]
                #         bb[2] = det[2]
                #         bb[3] = det[3]
                #
                #
                #         face_frame = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                #
                #
                #
                #         #scale face_frame
                #         try:
                #             face_frame = facenet.flip(face_frame, False)
                #             scaled = misc.imresize(face_frame, (image_size, image_size), interp='bilinear')
                #             scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                #                                interpolation=cv2.INTER_CUBIC)
                #             scaled= facenet.prewhiten(scaled)
                #             scaled_reshape=scaled.reshape(-1,input_image_size,input_image_size,3)
                #             feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                #             emb_array[0, :] = tf_sess.run(embeddings, feed_dict=feed_dict)
                #
                #             #predict based on extract embedding emb_array
                #             face_id_idx = face_svc_classifier.predict(emb_array)

        else:
            capture.release()
            cv2.destroyAllWindows()
            break


if __name__ =='__mian__':


    face_mtcnn(video_path,fn,save_path,threshold,sec_per_frame)
