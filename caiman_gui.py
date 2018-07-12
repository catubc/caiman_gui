# Caiman GUI
#

import sys
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QApplication, QMainWindow, QSlider,
                              QFileDialog, QTabWidget)
from PyQt5.uic import loadUi
import numpy as np

class MainW(QMainWindow):
    def __init__(self):
        super(MainW,self).__init__()
        loadUi('main.ui',self)
        
        self.image=[]
        self.movie=[]
        self.playmovie_flag=True
        # execute when tabs changed
        #self.tabs = QTabWidget()
        #self.tabsWidget.blockSignals(True) #just for not showing the initial message
        self.tabsWidget.currentChanged.connect(self.onChange) #changed!

        # Cat: TODO: is it possible to send slider object to callback?
        #       if so, then we can just have a generic callback for all sliders
        #       that plot movies
        
        # initialize load tab sliders
        self.loadSliderFrame.valueChanged.connect(self.loadSliderFrame_func)
        
        # initalize motion tab sliders
        self.motionSliderFrame.valueChanged.connect(self.motionSliderFrame_func)
        
        
        
    # *****************************************************
    # **************** BUTTONS  ***************************
    # *****************************************************
    ''' Syntax for buttons is: "on_" + name of button + '_clicked".
        Convention: each button name begins with the tab name (e.g. "load", 
        "motion") in non-capitalized letters.
        New buttons created through Qt Designer must have unique names, 
        and then have a @pyqtSlot() decorator above their function call.
    '''

    @pyqtSlot()
    def on_loadFiles_clicked(self):
        self.files = QFileDialog.getOpenFileNames(self,  'Open file','/home/cat/',
          ("Images (*.avi *.txt)"))[0]
        
        if len(self.files)>0:
            print ("files selected: ")
            for k in range(len(self.files)):
                print (self.files[k])
                self.loadList.addItem(self.files[k])

    @pyqtSlot()
    def on_loadMovieLoad_clicked(self):
        # send slider, load list and target screen
        self.generic_movie_load(self.loadSliderFrame, self.loadList, 
                                                        self.loadScreen)

    @pyqtSlot()
    def on_loadPlayMovie_clicked(self):
        # TODO
        print ("play movie not implemented")
        #self.worker = Worker()
        #self.worker.playflag = True
        #self.worker.do_stuff()
        
    @pyqtSlot()
    def on_loadStopMovie_clicked(self):
        print ("stop movie not impplemented")
        
    @pyqtSlot()
    def on_motionRunRigid_clicked(self):
        print ("Running rigid motion correction")
        self.motionList.addItem("Berlusconi " +
                    str(np.random.randint(2016,2040,1)))
        
    @pyqtSlot()
    def on_motionMovieLoad_clicked(self):
        # send slider, load list and target screen
        self.generic_movie_load(self.motionSliderFrame, self.motionList, 
                                                        self.motionScreen)


    # *****************************************************
    # **************** FUNCTIONS **************************
    # *****************************************************
    ''' Goal of functions is to direct all tab/screen input through 
        generic functions.
    '''

    def loadmovie(self,fname):
        ''' Load movie from disk using opencv
            TO DO: may need to write .tif readers etc.
                    OR use caiman only to read imaging files
        ''' 

        print ("Loading movie: ", fname)
        if "Berlusconi" in fname: 
            self.movie = np.random.random((300,300,300))
            return
            
        camera = cv2.VideoCapture(fname)
        movie = []
        while True:
            (grabbed, frame) = camera.read()
            if not grabbed: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            movie.append(frame)

        self.movie = np.array(movie)
        print ("Finished loading")
        
    def generic_movie_load(self, slider_widget, list_widget, 
                                                        screen_widget):
        ''' Load movie, initialize first frame in screen widget and
            find length of image stack and assign to slider range.
        '''
        
        if len(list_widget.selectedIndexes())>0:
            fname = list_widget.selectedIndexes()[0].data()
            self.loadmovie(fname)
            self.displayImage(self.movie[0], screen_widget)
            
            # set slider range also
            slider_widget.setRange(0,self.movie.shape[0]-1)

        else:
            print (list_widget.selectedIndexes())
            print ("No file selected...")


    def motionSliderFrame_func(self):
        # call the generic slider with slider ID and target widget ID
        self.generic_slider_func(self.motionSliderFrame, self.motionScreen)

    def loadSliderFrame_func(self):
        # call the generic slider with slider ID and target widget ID
        self.generic_slider_func(self.loadSliderFrame, self.loadScreen)

    def generic_slider_func(self, slider_widget, screen_widget):
        ''' Takes slider widget and its value and displays in target
            screen widget
        '''
        # check to see if movie loaded
        if len(self.movie)==0:
            print ("No movie loaded ...")
            return
        
        # load frame from slider and grab movie frame
        image = self.movie[slider_widget.value()]

        self.displayImage(image, screen_widget)

    def displayImage(self, image, screen_widget):
        ''' Displays single image in a target screenwidget
        '''
        # standardized code for convering image to 
        # Cat: TODO: is all this formatting necessary? 
        # Cat: TODO: also, can we just cast opencv imshow to the widget?
        qformat=QImage.Format_Indexed8
        if len(image.shape)==3:
            if(image.shape[2])==4:
                qformat=QImage.Format_RBA8888
            else:
                qformat=QImage.Format_RGB888
        
        # convert from opencv format to pyqt QImage format
        img=QImage(image,image.shape[1],image.shape[0],image.strides[0],qformat)
        pixmap = QtGui.QPixmap(img)

        screen_widget.setPixmap(pixmap)        
        
        
    # *****************************************************
    # ************ TAB CHANGE - EXECUTE CODE  *************
    # *****************************************************
    def onChange(self,i): #changed!
        print ("Current Tab: ", i)

class Worker(QtCore.QObject):
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.t1 = QtCore.QThread()
        self.moveToThread(self.t1)
        self.t1.start()

    def do_stuff(self):
        ctr=0
        while self.playflag:
            if ctr%1000000==0:
                print ("loop: ", ctr)
            ctr+=1
            
    
    

app = QApplication(sys.argv)
window=MainW()
window.setWindowTitle('Test')
window.show()
sys.exit(app.exec_())






# Cat: OPENCV FUNCTION WORKS, BUT NOT USED RIGHT NOW
## *****************************************************
## **************** MOVIE OUTSIDE WINDOW ****************
## *****************************************************
#@pyqtSlot()
#def on_loadmovieOpencv_clicked(self):

    #def nothing(x):
        #pass

    #filename = 'tests/5Cell.avi'
    #camera = cv2.VideoCapture(filename)

    ## Create a black image, a window
    #cv2.namedWindow('image')

    ## load movie
    #movie = []
    #while True:
        ## grab the current frame
        #(grabbed, frame) = camera.read()

        ## if we are viewing a video and we did not grab a
        ## frame, then we have reached the end of the video
        #if not grabbed: 
            #break
        #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #movie.append(frame)

    #movie = np.array(movie)

    ## create trackbars for color change
    #cv2.createTrackbar('slider','image',0,len(movie)-1,nothing)

    #r_old=0
    #cv2.imshow('image',movie[0])

    #while True:

        #cv2.waitKey(1)

        ## close window on clicking X top left corner
        #if cv2.getWindowProperty('image',1) == -1 :
            #cv2.destroyAllWindows()
            #break

        ## get current positions of four trackbars
        #r = cv2.getTrackbarPos('slider','image')

        ## if slider moved; update movie image
        #if r!=r_old: 
            #cv2.imshow('image', movie[r])
            #r_old = r
            
