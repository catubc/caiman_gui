# Caiman GUI
#

import sys
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QSlider
from PyQt5.uic import loadUi
import numpy as np

class MainW(QMainWindow):
    def __init__(self):
        super(MainW,self).__init__()
        loadUi('main.ui',self)
        
        self.image=None
        
        # Qt Creator makes sliders with unique names
        # reassign names to unique MainWindow objects
        # nb: not necessary, can just use names directly
        self.slider1 = self.Slider1
        self.slider1.valueChanged.connect(self.value_changed)
        
    # *****************************************************
    # **************** IMAGE INSIDE WINDOW ****************
    # *****************************************************
    @pyqtSlot()
    def on_loadpicture_clicked(self):
        fname = '/home/cat/Dropbox/murphy_lab.jpg'
        self.image=cv2.imread(fname)
        self.displayImage()
        
    def displayImage(self):
        qformat=QImage.Format_Indexed8
        
        if len(self.image.shape)==3:
            if(self.image.shape[2])==4:
                qformat=QImage.Format_RBA8888
            else:
                qformat=QImage.Format_RGB888
        
        img=QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.strides[0],qformat)
        #img=img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        #self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        
    # *****************************************************
    # **************** MOVIE INSIDE WINDOW ****************
    # *****************************************************
    @pyqtSlot()
    def on_loadmovie_clicked(self):
        ''' Load movie from disk for demo purposes
            Note: can also use opencv directly via a pop up window 
        '''
        print ("Loading movie")
        filename = '/home/cat/Downloads/5Cell.avi'
        camera = cv2.VideoCapture(filename)

        # load movie
        movie = []
        while True:
            (grabbed, frame) = camera.read()
            if not grabbed: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            movie.append(frame)

        self.movie = np.array(movie)
        print ("Finished loading")
   
        
    @pyqtSlot()
    def value_changed(self):
        frame = self.slider1.value()
        print ("FD: ",frame)
        
        #cv2.imshow('image', self.movie[frame])
        self.image = self.movie[frame]
        print (self.image.shape)
        
        qformat=QImage.Format_Indexed8
        if len(self.image.shape)==3:
            if(self.image.shape[2])==4:
                qformat=QImage.Format_RBA8888
            else:
                qformat=QImage.Format_RGB888
        
        print (self.image.shape[1],self.image.shape[0],self.image.strides[0])

        img=QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.strides[0],qformat)

        #img=img.rgbSwapped()
        pixmap = QtGui.QPixmap(img)
        pixmap4 = pixmap.scaled(500,500)

        self.imgLabel.setPixmap(pixmap4)
        self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)


    # *****************************************************
    # **************** MOVIE OUTSIDE WINDOW ****************
    # *****************************************************
    @pyqtSlot()
    def on_loadmovieOpencv_clicked(self):

        def nothing(x):
            pass

        filename = '/home/cat/Downloads/5Cell.avi'
        camera = cv2.VideoCapture(filename)

        # Create a black image, a window
        cv2.namedWindow('image')

        # load movie
        movie = []
        while True:
            # grab the current frame
            (grabbed, frame) = camera.read()

            # if we are viewing a video and we did not grab a
            # frame, then we have reached the end of the video
            if not grabbed: 
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            movie.append(frame)

        movie = np.array(movie)

        # create trackbars for color change
        cv2.createTrackbar('slider','image',0,len(movie)-1,nothing)

        r_old=0
        cv2.imshow('image',movie[0])

        while True:

            cv2.waitKey(1)

            # close window on clicking X top left corner
            if cv2.getWindowProperty('image',1) == -1 :
                cv2.destroyAllWindows()
                break

            # get current positions of four trackbars
            r = cv2.getTrackbarPos('slider','image')

            # if slider moved; update movie image
            if r!=r_old: 
                cv2.imshow('image', movie[r])
                r_old = r
            
app = QApplication(sys.argv)
window=MainW()
window.setWindowTitle('Test')
window.show()
sys.exit(app.exec_())
