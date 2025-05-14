import numpy as np
from filterpy.kalman import KalmanFilter


class SimpleTracker:

    def __init__(self, frame_lim=5, iou_tr=0.5):
        self.frame_lim = frame_lim
        self.rt_objs = []
        self.iou_tr = iou_tr

    def create_objs(self, bboxes:np.ndarray):
        if len(bboxes) != 0:
            for bbox in bboxes:
                self.rt_objs.append(TrackedObject(bbox, len(self.rt_objs), self.frame_lim))

    def clear_objs(self):
        pass

    def get_iou(self, bbox1, bboxes2):
        '''
        bbox1 size: (4,)
        bboxes2 size: (N, 4)
        '''
        x1 = np.maximum(bbox1[0], bboxes2[:, 0])
        y1 = np.maximum(bbox1[1], bboxes2[:, 1])
        x2 = np.minimum(bbox1[2], bboxes2[:, 2])
        y2 = np.minimum(bbox1[3], bboxes2[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_bbox = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area_bboxes = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        union = area_bbox + area_bboxes - intersection
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)

    def get_max_iou_bbox(self, bboxes, iou):
        bboxes_iou = bboxes[iou > self.iou_tr]
        if len(bboxes_iou) == 0:
            return None
        elif len(bboxes_iou) == 1:
            return bboxes_iou
        else:
            return bboxes_iou.max()
            
    def get_new_bboxes(self, tracked_bboxes, bboxes):
        pass

    def update(self, bboxes:np.ndarray):
        if len(self.rt_objs) == 0:
            self.create_objs(bboxes)
            return bboxes
        else:
            out_bboxes = []
            tracked_bboxes = []
            for obj in self.rt_objs:
                iou = self.get_iou(obj.xyxy, bboxes)
                iou_bbox = self.get_max_iou_bbox(bboxes, iou) #bboxes[iou>self.iou_tr]
                if iou_bbox is not None:
                    tracked_bboxes.append(iou_bbox)
                bbox_track = obj.update(iou_bbox)
                out_bboxes.append(bbox_track)
            self.clear_objs()
            new_bboxes = self.get_new_bboxes(tracked_bboxes, bboxes)
            self.create_objs(new_bboxes)
            return out_bboxes


class TrackedObject:
    def __init__(self, bbox, track_id, frame_lim):
        self.frame_lim = frame_lim 
        self.curr_rate = frame_lim
        self.track_id = track_id
        self.kf = create_kalman_filter()
        initial_z = convert_bbox_to_z(bbox)
        self.kf.x[:4] = np.expand_dims(initial_z, axis=-1)  # Инициализация состояния
    
    @property
    def xyxy(self):
        xywh = self.kf.x[:4]
        xyxy = convert_z_to_bbox(xywh)
        return np.squeeze(xyxy)
    
    def update(self, bbox:list=None):
        self.kf.predict()
        if bbox:
            z = convert_bbox_to_z(bbox)
            self.kf.update(z)
            self.curr_rate = self.frame_lim
        else:
            self.curr_rate -= 1
        smoothed_z = self.kf.x[:4]  
        return convert_z_to_bbox(smoothed_z)
    

def create_kalman_filter():
    # Инициализация фильтра Калмана (8D: x, y, w, h, vx, vy, vw, vh)
    kf = KalmanFilter(dim_x=8, dim_z=4)  # 8 состояний, 4 измерения (x, y, w, h)
    
    # Матрица перехода (предполагаем движение с постоянной скоростью)
    kf.F = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx*dt
        [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy*dt
        [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw*dt
        [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh*dt
        [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
        [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
        [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
        [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
    ])
    
    # Матрица измерений (измеряем только x, y, w, h)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ])
    
    # Ковариация процесса (шум модели)
    kf.Q = np.eye(8) * 0.01
    
    # Ковариация измерений (шум датчика)
    kf.R = np.eye(4) * 0.1
    
    # Начальная ковариация (неопределённость)
    kf.P = np.eye(8) * 10
    
    return kf


def convert_bbox_to_z(bbox):
    """Конвертирует [x1, y1, x2, y2] в [x_center, y_center, w, h]"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    return np.array([x_center, y_center, w, h])


def convert_z_to_bbox(z):
    """Конвертирует [x_center, y_center, w, h] обратно в [x1, y1, x2, y2]"""
    x_center, y_center, w, h = z
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return np.array([x1, y1, x2, y2])

