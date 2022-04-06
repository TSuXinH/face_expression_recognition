from img_video_API import *
from settings_form import *
from start_form import *
from image_form import *
from video_form import *
import sys
import numpy as np
from PyQt5.QtCore import QThread, QUrl
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QIcon, QImage
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget


def set_style_BTN(BTN, tip=None):
    BTN.setStyleSheet('QPushButton{color:black}'
                      'QPushButton:hover{font: bold}'
                      'QPushButton{background-color:white}'
                      'QPushButton{border:2px}'
                      'QPushButton{border-radius:10px}'
                      'QPushButton{padding:2px 4px}')
    if tip is not None:
        BTN.setToolTip(tip)


def set_style_bg(Form, bg_path='./ui_relatives/bg.jpg', icon_path='./ui_relatives/icon.ico'):
    Form.setStyleSheet('QTabWidget:pane {border-top:0px solid #e8f3f9;background:  transparent; }')
    palette = QPalette()
    image = QPixmap(bg_path).scaled(Form.width(), Form.height())
    palette.setBrush(QPalette.Background, QBrush(image))
    Form.setPalette(palette)
    if icon_path:
        icon = QIcon()
        icon.addPixmap(QPixmap(icon_path))
        Form.setWindowIcon(icon)


def array2pix(img):
    Y, X = img.shape[: 2]
    temp = np.zeros_like(img)
    temp[:, :, 0] = img[:, :, 2]
    temp[:, :, 1] = img[:, :, 1]
    temp[:, :, 2] = img[:, :, 0]
    temp = np.uint8(temp)
    q_img = QImage(temp, X, Y, QImage.Format_RGB888)
    q_pix = QPixmap.fromImage(q_img)
    return q_pix


class my_thread(QThread):
    signal = QtCore.pyqtSignal(bool)

    def __init__(self, input_interface, video_path):
        super(my_thread, self).__init__()
        self.interface = input_interface
        self.video_path = video_path

    def run(self):
        result = self.interface.get_post_video(self.video_path)
        self.signal.emit(result)


class start_form(QWidget, Ui_start_form):
    def __init__(self):
        super(start_form, self).__init__()
        self.setupUi(self)
        self.size = (640, 480)
        self.color = (0, 255, 0)
        self.image_form = image_form(self, self.color, self.size)
        self.video_form = video_form(self, self.color, self.size)
        self.settings_form = settings_form(self)
        set_style_BTN(self.quit_BTN, 'Get out')
        set_style_BTN(self.image_BTN, 'Go into the page of image process')
        set_style_BTN(self.video_BTN, 'Go into the page of video process')
        set_style_BTN(self.settings_BTN, 'Set color or size played')
        set_style_bg(self)

        self.image_BTN.clicked.connect(self.image_BTN_clicked)
        self.video_BTN.clicked.connect(self.video_BTN_clicked)
        self.quit_BTN.clicked.connect(self.close)
        self.settings_BTN.clicked.connect(self.settings_BTN_clicked)
        self.settings_form.signal.connect(lambda input_tuple: self.get_color_size(input_tuple))

    def image_BTN_clicked(self):
        self.image_form.show()
        self.image_form.set_color_size(self.color, self.size)
        self.hide()

    def video_BTN_clicked(self):
        self.video_form.show()
        self.video_form.set_color_size(self.color, self.size)
        self.hide()

    def settings_BTN_clicked(self):
        self.settings_form.show()
        self.hide()

    def get_color_size(self, input_tuple):
        self.color, self.size = input_tuple


class settings_form(QWidget, Ui_settings_form):
    signal = QtCore.pyqtSignal(tuple)

    def __init__(self, previous_form):
        super(settings_form, self).__init__()
        self.setupUi(self)
        self.previous_form = previous_form
        set_style_bg(self)
        set_style_BTN(self.back_BTN,
                      'Change color or size\n'
                      'Changing size is strongly not recommended\n'
                      'It may cause disharmony'
                      )
        self.back_BTN.clicked.connect(self.back_BTN_clicked)

    def back_BTN_clicked(self):
        color = (self.B_SB.value(), self.G_SB.value(), self.R_SB.value())
        size = (self.W_SB.value(), self.H_SB.value())
        self.signal.emit((color, size))
        self.hide()
        self.previous_form.show()


class image_form(QWidget, Ui_image_form):
    def __init__(self, previous_form, color, size):
        super(image_form, self).__init__()
        self.interface = interface(model, device, color, size)
        self.previous_form = previous_form
        self.setupUi(self)
        set_style_BTN(self.import_BTN, 'Import an image')
        set_style_BTN(self.back_BTN, 'Go back to the start page')
        set_style_bg(self)

        self.back_BTN.clicked.connect(self.back_BTN_clicked)
        self.import_BTN.clicked.connect(self.import_BTN_clicked)

    def set_color_size(self, color, size):
        self.interface.change_color_size(color, size)

    def import_BTN_clicked(self):
        directory, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Choose an image', './', 'All Files (*);;')
        if directory == '':
            return
        if directory[len(directory) - 3:] != 'jpg':
            msg = QMessageBox.warning(self, 'file type error', 'Please try other images', QMessageBox.Cancel)
            if msg == QMessageBox.Cancel:
                return
        is_face = self.interface.get_post_img(directory)
        if is_face is False:
            msg = QMessageBox.information(self, 'No face detected', 'Please try other images', QMessageBox.Ok)
            if msg == QMessageBox.Ok:
                pass
        post_img = self.interface.post_img
        post_img = array2pix(post_img)
        self.image_label.setPixmap(post_img)

    def back_BTN_clicked(self):
        self.previous_form.show()
        self.hide()
        self.image_label.setPixmap(QPixmap(''))
        self.interface.clear()


class video_form(QWidget, Ui_video_form):
    def __init__(self, previous_form, color, size):
        super(video_form, self).__init__()
        self.setupUi(self)
        self.interface = interface(model, device, color, size)
        self.previous_form = previous_form
        self.raw_video_path = ''
        self.processed_path = ''
        self.thread = None
        self.change_import_back_BTN(True)
        self.change_play_BTN(False)
        set_style_bg(self)
        self.player_video = QVideoWidget(self.player_WGT)
        self.player = QMediaPlayer()

        self.back_BTN.clicked.connect(self.back_BTN_clicked)
        self.import_BTN.clicked.connect(self.import_BTN_clicked)
        self.play_BTN.clicked.connect(self.play_BTN_clicked)

    def set_color_size(self, color, size):
        self.interface.change_color_size(color, size)

    def import_BTN_clicked(self):
        directory, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Choose a video', './', 'All Files (*);;'
        )
        if directory == '':
            return
        if directory[len(directory) - 3:] != 'mp4':
            msg = QMessageBox.warning(self, 'file type error', 'Please try other videos', QMessageBox.Cancel)
            if msg == QMessageBox.Cancel:
                return
        msg = QMessageBox.information(self,
                                      'Processing or Cancel',
                                      'Click "OK" to start processing\n'
                                      'Please wait for a period if processing',
                                      QMessageBox.Ok | QMessageBox.Cancel
                                      )
        if msg == QMessageBox.Ok:
            self.thread = my_thread(self.interface, directory)
            self.thread.signal.connect(lambda input_flag: self.thread_end(input_flag))
            self.change_import_back_BTN(False)
            self.thread.start()
        else:
            return

    def thread_end(self, flag):
        if flag:
            self.change_import_back_BTN(True)
            self.change_play_BTN(True)
            # Behind the window: flash
            # Minimized: jump out
            self.activateWindow()
            self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
            self.showNormal()

    def back_BTN_clicked(self):
        self.previous_form.show()
        self.hide()
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.stop()
        self.change_play_BTN(False)
        self.player_WGT.hide()
        self.player_video.hide()

    def play_BTN_clicked(self):
        self.player_WGT.setGeometry(QtCore.QRect(
            int((1000 - self.interface.size[0]) // 2), 100,
            self.interface.size[0], self.interface.size[1])
        )
        self.player_video.setGeometry(QtCore.QRect(
            0, 0, self.interface.size[0], self.interface.size[1])
        )  # relative to the label
        self.player_WGT.show()
        self.player_video.show()
        self.player.setVideoOutput(self.player_video)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.interface.post_video_path)))
        self.player.play()

    def change_import_back_BTN(self, flag):
        self.import_BTN.setEnabled(flag)
        self.back_BTN.setEnabled(flag)
        if flag:
            set_style_BTN(self.import_BTN, 'Import a video')
            set_style_BTN(self.back_BTN, 'Go back to the start page')
        else:
            self.import_BTN.setStyleSheet('QPushButton{color:white}'
                                          'QPushButton{background-color:gray}')
            self.import_BTN.setToolTip('Temporarily not allowed')
            self.back_BTN.setStyleSheet('QPushButton{color:white}'
                                        'QPushButton{background-color:gray}')
            self.back_BTN.setToolTip('Temporarily not allowed')

    def change_play_BTN(self, flag):
        if flag:
            set_style_BTN(self.play_BTN, 'You can play it Now')
            self.play_BTN.setEnabled(True)
        else:
            self.play_BTN.setStyleSheet('QPushButton{color:white}'
                                        'QPushButton{background-color:gray}')
            self.play_BTN.setToolTip('Please input a video')
            self.play_BTN.setEnabled(False)

    def closeEvent(self, event):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            msg = QMessageBox.question(self, 'Unfinished',
                                       'Do you want to quit?', QMessageBox.Yes | QMessageBox.No)
            if msg == QMessageBox.Yes:
                self.player.stop()
                event.accept()
            else:
                event.ignore()
                self.player.play()
        else:
            event.accept()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load('./c_net.pkl', map_location=device).to(device)
    app = QtWidgets.QApplication(sys.argv)
    start_form = start_form()
    start_form.show()
    sys.exit(app.exec_())
