from utility import *
from model import *
import cv2
from copy import deepcopy
import torchvision.transforms as transforms

mu, std = .50774, .21187  # calculated already
test_trans = transforms.Compose([
    transforms.TenCrop(40),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mu, std)(crop) for crop in crops])),
])


class interface:
    """
    This class is used to generate all the APIs for images or videos.
    """
    post_video_path = './output_video.mp4'

    def __init__(self, classifier, device, color, size):
        self.locator = cv2.CascadeClassifier(cv2.data.haarcascades + r'./haarcascade_frontalface_default.xml')
        self.classifier = classifier.to(device)
        self.device = device
        self.color = color
        self.size = size
        self.pre_img = None
        self.post_img = None

    def clear(self):
        self.pre_img = None
        self.post_img = None

    def change_color_size(self, color, size):
        self.color = color
        self.size = size

    def process_single_img(self, img):
        """
        Used to process an input image.
        :param img: the file_path of image_path
        :return: img, whether faces are detected or not
        """
        post_img = deepcopy(img)
        gray_mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = self.locator.detectMultiScale(gray_mat, scaleFactor=1.2, minNeighbors=5)
        if len(rect) == 0:
            return cv2.resize(post_img, self.size), False
        test_batch = []
        for item in rect:
            test_batch.append(
                cv2.resize(gray_mat[item[1]: item[1] + item[3], item[0]: item[0] + item[2]], (48, 48))
            )
        pre = quick_test(self.classifier, test_batch, self.device, test_trans, is_tenCrop=True)
        for idx, item in enumerate(rect):
            font_scale = float(int(np.sqrt(post_img.shape[0] * post_img.shape[1]) / 100)) / 10 + .2
            thickness = 1 if font_scale < 0.5 else 2
            cv2.rectangle(post_img, (item[0], item[1]), (item[0] + item[2], item[1] + item[3]),
                          lineType=1, color=self.color)
            cv2.putText(post_img, label_dict[pre[idx]], (item[0], item[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.color, thickness)
        return cv2.resize(post_img, self.size), True

    def get_post_img(self, img_path):
        self.pre_img = cv2.imread(img_path)
        self.post_img, is_face = self.process_single_img(self.pre_img)
        return is_face

    def get_post_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        size = self.size
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        modifier = cv2.VideoWriter(interface.post_video_path, fourcc, fps, size, True)
        while cap.isOpened():
            success, frame = cap.read()
            if success is None or frame is None:
                break
            post_frame, _ = self.process_single_img(frame)
            modifier.write(post_frame)
        cap.release()
        modifier.release()
        return True
