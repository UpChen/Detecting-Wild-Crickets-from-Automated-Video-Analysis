import numpy as np
import onnx
import onnxruntime
import os
import sys
import cv2
from video_parser import *
from odutil import visual

sys.path.append(os.getcwd())


class Detector():
    '''
    Detector
    '''
    def __init__(self, onnx_path):
        """
        Loading onnx model
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def infer(self, image_tensor):
        '''
        Model reasoning
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        outputs = self.onnx_session.run(None, input_feed)
        objects = self._nms(outputs)
        return objects

    def _nms(self, outputs, conf_thresh=0.3, nms_thresh=0.5):
        '''
        NMS
        '''
        def iou(bbox_1, bbox_2):
            """
            Caculate IOU
            bbox_i: (xmin, ymin, xmax, ymax)
            """

            inter_xmin = max(bbox_1[0], bbox_2[0])
            inter_xmax = min(bbox_1[2], bbox_2[2])
            inter_ymin = max(bbox_1[1], bbox_2[1])
            inter_ymax = min(bbox_1[3], bbox_2[3])
            if(inter_xmin >= inter_xmax or inter_ymin >= inter_ymax):
                return 0
            inter_area = (inter_xmax-inter_xmin)*(inter_ymax-inter_ymin)
            union_area = (bbox_1[2]-bbox_1[0])*(bbox_1[3]-bbox_1[1]) + \
                (bbox_2[2]-bbox_2[0])*(bbox_2[3]-bbox_2[1])-inter_area
            return inter_area/union_area

        objs = []
        for batch in outputs:
            for img in batch:
                for box in img:
                    if max(box[4:]) > conf_thresh:
                        objs.append(box)

        objs.sort(key=lambda x: max(x[4:]), reverse=True)
        objects = []
        for i, obj in enumerate(objs):
            if obj[-1] == -1:
                continue
            bbox_1 = (obj[0]-obj[2]/2,
                      obj[1]-obj[3]/2,
                      obj[0]+obj[2]/2,
                      obj[1]+obj[3]/2,
                      obj[4],
                      obj[5])
            objects.append(bbox_1)
            for j in range(i+1, len(objs)):
                bbox_2 = (objs[j][0]-objs[j][2]/2,
                          objs[j][1]-objs[j][3]/2,
                          objs[j][0]+objs[j][2]/2,
                          objs[j][1]+objs[j][3]/2,
                          objs[j][4],
                          objs[j][5])
                if iou(bbox_1, bbox_2) > nms_thresh:
                    objs[j][-1] = -1
        return objects


def track(video_path, output_dir, model_path='model.onnx', frame_size=(1280, 800)):
    '''
    Perform full-process video analysis, detection, and synthesis
    '''
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    track_frame_path = os.path.join(output_dir, video_basename, 'track_frame')
    if not os.path.isdir(track_frame_path):
        os.makedirs(track_frame_path)

    # Parsing
    parser = ViderParser(video_path, frame_size)
    # Load the model
    model = Detector(model_path)

    track_record = []
    pbar = tqdm(len(parser.frame_data))
    for i, img in enumerate(parser.frame_data):
        x = np.expand_dims(img, axis=0).astype(np.float32)
        # Inference to get the bbox result of each picture
        outputs = model.infer(x)
        track_record.append(outputs)
        # Keep records of insect activity
        np.save(os.path.join(output_dir, '{}_track_record.npy'.format(
            video_basename)), track_record, allow_pickle=True)
        # Draw result
        for box in outputs:
            cls_ = 'crickt' if box[4] > box[5] else 'ID'
            cv2.rectangle(img,
                          (int(box[0]),
                           int(box[1])),
                          (int(box[2]),
                           int(box[3])),
                          (0, 255, 255),
                          4)
            cv2.putText(img,
                        cls_,
                        (int(box[0]), int(box[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        1)
        cv2.imwrite(os.path.join(track_frame_path,
                                 '{}_{}.jpg'.format(video_basename, i)), img)
        pbar.update()
    pbar.close()
    # Composite video
    compose_dir(track_frame_path, os.path.join(
        output_dir, '{}_track_result.avi'.format(video_basename)))
