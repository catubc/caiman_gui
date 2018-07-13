# Caiman GUI
#

import sys
import cv2
import numpy as np
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QApplication, QMainWindow, QSlider,
                              QFileDialog, QTabWidget)
from PyQt5.uic import loadUi

import caiman as cm
from caiman.motion_correction import MotionCorrect

class MainW(QMainWindow):
    def __init__(self):
        super(MainW,self).__init__()
        loadUi('main.ui',self)
 
        # initialize all required variables
        self.image = []
        self.movie = []
        self.files = []
        self.movie_ctr=0
        self.playmovie_flag=True
        self.data_min = None
        self.data_max = None
        self.slider_min = 0
        self.slider_max = 99
        # execute when tabs changed
        #self.tabs = QTabWidget()
        #self.tabsWidget.blockSignals(True) #just for not showing the initial message
        self.tabsWidget.currentChanged.connect(self.onChange) #changed!

        # Cat: TODO: is it possible to send slider object to callback?
        #       if so, then we can just have a generic callback for all sliders
        #       that plot movies
        
        # initialize load tab sliders
        self.loadSliderMaxIntensity.setValue(99)
        self.loadSliderMinIntensity.setValue(0)
        self.loadSliderFrame.valueChanged.connect(self.loadSliderFrame_func)
        self.loadSliderMinIntensity.valueChanged.connect(self.loadSliderMinIntensity_func)
        self.loadSliderMaxIntensity.valueChanged.connect(self.loadSliderMaxIntensity_func)

        # initalize motion tab sliders
        self.motionSliderFrame.valueChanged.connect(self.motionSliderFrame_func)
        
        # initialize postProcessing data
        self.show_postProcScreenTraces()
        
    ''' *********************************************************
        *********************************************************
        ******************** MOVIE FUNCTIONS ********************
        *********************************************************
        *********************************************************
    '''
    ''' Syntax for buttons is: "on_" + name of button + '_clicked".
        Convention: each button name begins with the tab name (e.g. "load", 
        "motion") in non-capitalized letters.
        New buttons created through Qt Designer must have unique names, 
        and then have a @pyqtSlot() decorator above their function call.
    '''

    @pyqtSlot()
    def on_loadFiles_clicked(self):
        self.files = QFileDialog.getOpenFileNames(self,  'Open file','./',
                                                  "Images (*.avi *.tif *.hdf5 *.mmap)")[0]
        
        self.files.sort()
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

        self.timer = QtCore.QTimer(self)
        self.timer.start(10)
        self.timer.timeout.connect(self.playmovie)        
        
    @pyqtSlot()
    def on_loadStopMovie_clicked(self):
        self.timer.stop()


    @pyqtSlot()
    def on_motionRunRigid_clicked(self):
        print ("Running rigid motion correction on the following files")
        print(self.files)
        self.motion_correct_rigid(self.files)
        for fls in self.motion_correct.fname_tot_rig:
            self.motionList.addItem(fls)

    @pyqtSlot()
    def on_motionRunPWRigid_clicked(self):
        print('Running pw-rigid motion correction on the following files')
        if len(self.files)>0:
            print(self.files)
            self.motion_correct_pwrigid(self.files)
            for fls in self.motion_correct.fname_tot_els:
                self.motionList.addItem(fls)
        else:
            print("no files loaded")
        
    @pyqtSlot()
    def on_motionMovieLoad_clicked(self):
        # send slider, load list and target screen
        self.generic_movie_load(self.motionSliderFrame, self.motionList, 
                                                        self.motionScreen)


    ''' *********************************************************
        *********************************************************
        ******************** MOVIE FUNCTIONS ********************
        *********************************************************
        *********************************************************
    '''


    def loadmovie(self,fname):
        ''' Load movie from disk using opencv
            TO DO: may need to write .tif readers etc.
                    OR use caiman only to read imaging files
        ''' 

        # load movie
        print ("Loading movie")
        movie = cm.load('/home/cat/code/CaImAn/example_movies/demoMovie.tif')

        self.movie = movie
        self.data_min = np.float(movie.min())
        self.data_max = np.float(movie.max())

        print(self.movie.shape)
        print ("Finished loading")
    
    def playmovie(self):
        # load next frame and compute movie counter
        self.movie_ctr=(self.movie_ctr+1)%self.movie.shape[0]
        img = self.load_single_image(self.movie_ctr)
        
        # set intensity based on slider values
        image = (np.clip((img-self.slider_min)/
              (self.slider_max-self.slider_min),0,1)*255).astype(np.uint8)
              
              
        # update slider location with current frame index
        self.loadSliderFrame.setValue(self.movie_ctr)
        
        # update slide frame box with current frame index
        self.loadLineEditFrameId.setText(str(self.movie_ctr))
        
        # display image
        self.displayImage(image, self.loadScreen)
    
    def generic_movie_load(self, slider_widget, list_widget, 
                                                        screen_widget):
        ''' Load movie, initialize first frame in screen widget and
            find length of image stack to assign to slider range.
        '''
        
        if len(list_widget.selectedIndexes())>0:
            fname = list_widget.selectedIndexes()[0].data()
            
            print('Loading movie:' + fname)
            self.loadmovie(fname)
            
            # set frame 0 on screen
            img = self.load_single_image(self.loadSliderFrame.value())
                    
            # set intensity based on slider values
            image = (np.clip((img-self.slider_min)/
                  (self.slider_max-self.slider_min),0,1)*255).astype(np.uint8)
              
            self.displayImage(image, screen_widget)
            
            # set slider range based on size of movie
            # Cat: TODO: the shape[0] value may not always be correct
            slider_widget.setRange(0,self.movie.shape[0]-1)

        else:
            print (list_widget.selectedIndexes())
            print ("No file selected...")

    
    ''' *********************************************************
        *********************************************************
        ******************** SLIDER FUNCTIONS********************
        *********************************************************
        *********************************************************
    '''

    def loadSliderFrame_func(self):
        # update movie_ctr based on current frame value
        self.movie_ctr=self.loadSliderFrame.value()
        self.loadLineEditFrameId.setText(str(self.movie_ctr))

        # load img
        img = self.update_image()

        # call the generic slider with slider ID and target widget ID
        self.generic_slider_func(img,self.loadScreen)

    def loadSliderMinIntensity_func(self):
        # call the generic slider with slider ID and target widget ID
        self.slider_min = np.float(self.loadSliderMinIntensity.value())

        img = self.update_image()
        self.generic_slider_func(img,self.loadScreen)

    def loadSliderMaxIntensity_func(self):
        # call the generic slider with slider ID and target widget ID
        self.slider_max = np.float(self.loadSliderMaxIntensity.value())

        img = self.update_image()
        
        self.generic_slider_func(img,self.loadScreen)
        
    def motionSliderFrame_func(self):
        # call the generic slider with slider ID and target widget ID
        self.generic_slider_func(self.motionSliderFrame, self.motionScreen)
        
    def generic_slider_func(self, image, screen_widget):
        ''' Takes slider widget and its value and displays in target
            screen widget
        '''
        # check to see if movie loaded
        if len(self.movie)==0:
            print ("No movie loaded ...")
            return

        # set frame counter value
        #self.loadSliderFrame.setValue(slider_widget.value())

        # load frame from slider and set frame widget
        #image = self.load_single_image(slider_widget.value())

        # display frame
        img = self.update_image()

        self.displayImage(image, screen_widget)


    ''' *********************************************************
        *********************************************************
        ******************** IMAGE FUNCTIONS ********************
        *********************************************************
        *********************************************************
    '''

    def load_single_image(self, index):
        ''' This function loads and normalizes a single image.
            Need to do this outside of display function.
        '''
        
        img = self.movie[index]
        image_out = ((img-self.data_min)/(self.data_max-self.data_min))*99
        return image_out
        
    def update_image(self):
        # load single image
        img = self.load_single_image(self.loadSliderFrame.value())
        
        # set intensity based on slider values
        img = (np.clip((img-self.slider_min)/
              (self.slider_max-self.slider_min),0,1)*255).astype(np.uint8)
        
        return img
        #self.displayImage(img, self.loadScreen)
        
    def displayImage(self, image_raw, screen_widget):
        ''' Displays single image in a target screenwidget
        '''
        
        print (image_raw.shape)
        # standardized code for convering image to 
        # Cat: TODO: is all this formatting necessary? 
        # Cat: TODO: also, can we just cast opencv imshow to the widget?
        
        # convert from opencv format to pyqt QImage format
        qformat=QImage.Format_Grayscale8
        img=QImage(image_raw,image_raw.shape[1],image_raw.shape[0],
                                            image_raw.strides[0],qformat)
        pixmap = QtGui.QPixmap(img)
                    
        # Stretch image to fit screen 
        img_width, img_height = image_raw.shape
        screen_width = screen_widget.frameGeometry().width()
        screen_height = screen_widget.frameGeometry().height()

        if img_width <= img_height: 
            pixmap = pixmap.scaled(screen_width, screen_height/
                                                (img_height/img_width))
        else:
            pixmap = pixmap.scaled(screen_width/(img_width/img_height), 
                                                screen_height)
                                                
        # self.imgLabel.setPixmap(pixmap4)
        #screen_widget.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        screen_widget.setPixmap(pixmap)

        
    ''' *********************************************************
        *********************************************************
        ******************** CAIMAN FUNCTIONS *******************
        *********************************************************
        *********************************************************
    '''


    def motion_correct_rigid(self, fname):
        try:
            dview = None
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

            niter_rig = 1  # number of iterations for rigid motion correction
            max_shifts = (6, 6)  # maximum allow rigid shift
            # for parallelization split the movies in  num_splits chuncks across time
            splits_rig = 56
            # first we create a motion correction object with the parameters specified
            min_mov = cm.load(fname[0], subindices=range(200)).min()
            # this will be subtracted from the movie to make it non-negative

            mc = MotionCorrect(fname, min_mov, dview=dview, max_shifts=max_shifts,
                               niter_rig=niter_rig, splits_rig=splits_rig,
                               border_nan='copy', shifts_opencv=True, nonneg_movie=True)

            mc.motion_correct_rigid(save_movie=True)

            self.motion_correct = mc
        except Exception as e:
            raise e
        finally:
            cm.cluster.stop_server(dview=dview)
            cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

    def motion_correct_pwrigid(self, fname):
        try:
            dview = None
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

            niter_rig = 1  # number of iterations for rigid motion correction
            max_shifts = (6, 6)  # maximum allow rigid shift
            # for parallelization split the movies in  num_splits chuncks across time
            splits_rig = 56
            # start a new patch for pw-rigid motion correction every x pixels
            strides = (48, 48)
            # overlap between pathes (size of patch strides+overlaps)
            overlaps = (24, 24)
            # for parallelization split the movies in  num_splits chuncks across time
            splits_els = 56

            upsample_factor_grid = 4  # upsample factor to avoid smearing when merging patches
            # maximum deviation allowed for patch with respect to rigid shifts
            max_deviation_rigid = 3
            # first we create a motion correction object with the parameters specified
            min_mov = cm.load(fname[0], subindices=range(200)).min()
            # this will be subtracted from the movie to make it non-negative

            mc = MotionCorrect(fname, min_mov, dview=dview, max_shifts=max_shifts,
                               niter_rig=niter_rig, splits_rig=splits_rig,
                               strides=strides, overlaps=overlaps,
                               splits_els=splits_els, border_nan='copy',
                               upsample_factor_grid=upsample_factor_grid,
                               max_deviation_rigid=max_deviation_rigid,
                               shifts_opencv=True, nonneg_movie=True)

            mc.motion_correct_pwrigid(save_movie=True)

            self.motion_correct = mc
        except Exception as e:
            raise e
        finally:
            cm.cluster.stop_server(dview=dview)
            cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

    ''' *********************************************************
        *********************************************************
        ******************** TAB EXECUTE FUNCTION ***************
        *********************************************************
        *********************************************************
    '''
    def onChange(self,i): #changed!
        print ("Current Tab: ", i)


    ''' *********************************************************
        *********************************************************
        **************** MATPLOTLIB FUNCTIONS *******************
        *********************************************************
        *********************************************************
    '''

    def show_postProcScreenTraces(self):
        
        self.postProcScreenTraces.setAlignment(QtCore.Qt.AlignCenter)

        dpi = 100
        self.ctr=0
        width = self.postProcScreenTraces.frameGeometry().width()/float(dpi)*.99
        height = self.postProcScreenTraces.frameGeometry().height()/float(dpi)*.99
        print (width, height)
        sc = MyDynamicMplCanvas(self.postProcScreenTraces, width=width, 
                                                        height=height, dpi=dpi)
                                                        
                                                        
#***********************************************************************
#************************ MATPLOTLIB CLASSES *************************
#***********************************************************************
class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=7, height=2, dpi=43):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self):
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)
        self.axes.plot(t, s)
        self.fig.tight_layout()


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        #timer.timeout.connect(self.update_figure)
        timer.start(10)
        self.ctr=0
        self.axes.set_ylim(0,1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        self.axes.cla()
        print ("Plotting: ", self.ctr)
        length = 10000
        x = np.arange(length)
        for k in range(1):
            y = np.random.randint(0, 1000,length)
            self.axes.plot(x,y,c=np.random.rand(3,))
        self.draw()
        self.ctr+=1
    
    
    

app = QApplication(sys.argv)

window=MainW()
window.setWindowTitle('CaImAn GUI V0.1 (Berlusconi)')
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
            
