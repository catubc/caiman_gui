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
from PyQt5.QtWidgets import (QDialog, QApplication, QMainWindow, QSlider, QMessageBox,
                              QFileDialog, QTabWidget)

from PyQt5.QtQuick import QQuickView
from PyQt5.QtCore import QUrl

from PyQt5.uic import loadUi

import pylab as pl
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.mmapping import save_memmap
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.summary_images import correlation_pnr
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.source_extraction.cnmf.cnmf import CNMF

class MainW(QMainWindow):
    def __init__(self):
        super(MainW,self).__init__()
        loadUi('main.ui',self)

        # initialize all required variables
        self.image = []
        self.movie = []
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
        self.movie_loadScreen=None

        # initalize motion tab sliders
        self.motionSliderMaxIntensity.setValue(99)
        self.motionSliderMinIntensity.setValue(0)
        self.motionSliderFrame.valueChanged.connect(self.motionSliderFrame_func)
        self.motionSliderMinIntensity.valueChanged.connect(self.motionSliderMinIntensity_func)
        self.motionSliderMaxIntensity.valueChanged.connect(self.motionSliderMaxIntensity_func)
        self.movie_motionScreen=None
        
        # initialize postProcessing data
        self.show_postProcScreenTraces()

        self.init_params()

        #
        # for k, d in self.params.items():
        #     d.setText('co')
        #     d.setProperty('enabled',False)

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
        files = QFileDialog.getOpenFileNames(self,  'Open file','./',
                                                  "Images (*.avi *.tif *.hdf5 *.mmap)")[0]

        files.sort()
        if len(files)>0:
            print ("files selected: ")
            for k in range(len(files)):
                print (files[k])
                self.loadList.addItem(files[k])

    @pyqtSlot()
    def on_loadMovieLoad_clicked(self):
        # send slider, load list and target screen
        self.movie_loadScreen = self.generic_movie_load(self.loadSliderFrame, 
                        self.loadList, self.loadScreen)
    @pyqtSlot()
    def on_loadButtonViewCorrImage_clicked(self):
        self.correlation_image = self.movie.local_correlations(eight_neighbours=True,swap_dim=False)
        img = self.normalize_frame(self.correlation_image)
        print("Computing correlation image")
        self.displayImage(img, self.loadScreen)


    @pyqtSlot()
    def on_loadButtonViewAverage_clicked(self):
        img = np.nanmean(self.movie,0)
        img = self.normalize_frame(img)
        print("Computing average image")
        self.displayImage(img, self.loadScreen)

    @pyqtSlot()
    def on_loadPlayMovie_clicked(self):
        self.timer = QtCore.QTimer(self)
        self.timer.start(10)
        self.timer.timeout.connect(self.playmovie_loadscreen)
       
    @pyqtSlot()
    def on_loadStopMovie_clicked(self):
        self.timer.stop()


    @pyqtSlot()
    def on_motionRunRigid_clicked(self):
        self.init_params()
        print ("Running rigid motion correction on the following files")
        files = [self.loadList.item(count).text() for count in range(self.loadList.count())]
        print(files)
        if len(files) > 0:
            self.motion_correct_rigid(files)
            for fls in self.motion_correct.fname_tot_rig:
                self.motionList.addItem(fls)
        else:
            print("no files loaded")

    @pyqtSlot()
    def on_motionRunPWRigid_clicked(self):
        self.init_params()
        print('Running pw-rigid motion correction on the following files')
        files = [self.loadList.item(count).text() for count in range(self.loadList.count())]
        print(files)
        if len(files)>0:
            self.motion_correct_pwrigid(files)
            for fls in self.motion_correct.fname_tot_els:
                self.motionList.addItem(fls)
        else:
            print("no files loaded")

    @pyqtSlot()
    def on_motionMovieLoad_clicked(self):

        if len(self.motionList.selectedIndexes())>0:
            fname = self.loadList.selectedIndexes()[0].data()
            movie1 = self.loadmovie(fname)
            
            fname = self.motionList.selectedIndexes()[0].data()
            movie2 = self.loadmovie(fname)
            
            self.movie_motionScreen = np.concatenate((movie1,movie2), axis=2)

            # set frame 0 on screen
            image = self.normalize_frame(self.movie_motionScreen[self.motionSliderFrame.value()],
                                                            norm_global=True)

            print ("  image size: ", image.shape)
            self.displayImage(image, self.motionScreen)

            # set slider range based on size of movie
            # Cat: TODO: the shape[0] value may not always be correct
            self.motionSliderFrame.setRange(0,self.movie_motionScreen.shape[0]-1)

        else:
            print (list_widget.selectedIndexes())
            QMessageBox.about(self, "Error", "No file selected, please select in text field.")
            print ("No file selected...")

        

    @pyqtSlot(int)
    def on_memmapComboListSelectFiles_currentIndexChanged(self, i):
        if i ==2:
            files = [self.loadList.item(count).text() for count in range(self.loadList.count())]
        elif i==1:
            files = [self.motionList.item(count).text() for count in range(self.motionList.count())]

        if len(files) > 0:
            print("files selected:")
            for k in range(len(files)):
                print(files[k])
                self.memmapListFiles.addItem(files[k])
        else:
            print('No File Loaded')

    @pyqtSlot()
    def on_memmapButtomStartMemmap_clicked(self):
        self.init_params()
        # send slider, load list and target screen
        files = [self.memmapListFiles.item(count).text() for count in range(self.memmapListFiles.count())]

        if len(files) > 0:
            new_file = self.memory_map_files(files)
            self.memmapListFileOutput.addItem(new_file)
        else:
            print("no files loaded")

    @pyqtSlot(int)
    def on_cnmfComboBoxSelectFile_currentIndexChanged(self, i):

        if i == 2:
            files = [self.loadList.item(count).text() for count in range(self.loadList.count())]
        elif i == 1:
            files = [self.memmapListFiles.item(count).text() for count in range(self.memmapListFiles.count())]

        if len(files) > 0:
            print("files selected:")
            for k in range(len(files)):
                print(files[k])
                self.cnmfListFilesInput.addItem(files[k])
        else:
            print('No File Loaded')

    @pyqtSlot()
    def on_cnmfButtonFilesCorrImage_clicked(self):
        file = self.cnmfListFilesInput.item(0).text()

        pl.imshow(cm.load(file).local_correlations(eight_neighbours=True, swap_dim=False, frames_per_chunk=1500,
                                                   order_mean=1), vmax=np.float(self.cnmfMaxCorrImage.text()))

    @pyqtSlot()
    def on_cnmfButtonFilesPNRImage_clicked(self):
        file = self.cnmfListFilesInput.item(0).text()
        Yr, dims, T = cm.load_memmap(file)
        Y = Yr.T.reshape((T,) + dims, order='F')
        print(np.float(self.cnmfMaxCorrImage.text()))
        cn_filter, pnr = correlation_pnr(Y, gSig=np.float(self.gSigFilter.text()), swap_dim=False)
        # inspect the summary images and set the parameters
        inspect_correlation_pnr(cn_filter, pnr)





    @pyqtSlot() # run CNMF full FOV
    def on_cnmfFOVButtonRunCNMF_clicked(self):
        self.init_params()
        files = [self.cnmfListFilesInput.item(count).text() for count in range(self.cnmfListFilesInput.count())]
        if len(files)>0:
            saved_object_path = self.run_cnmf(files, is_patch=False)
            self.cnmfListFilesOutput.addItem(saved_object_path)
        else:
            print('No files Loaded')

    @pyqtSlot()  # run CNMF patches
    def on_cnmfPatchesButtonRunCNMF_clicked(self):
        self.init_params()
        files = [self.cnmfListFilesInput.item(count).text() for count in range(self.cnmfListFilesInput.count())]
        if len(files) > 0:
            saved_object_path = self.run_cnmf(files, is_patch=True)
            self.cnmfListFilesOutput.addItem(saved_object_path)
        else:
            print('No files Loaded')


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
                    
            Note; movie is loaded and pased into its own container
                - i.e. each screen will have its own movie array in general
        '''

        # load movie
        print ("  loading movie...")
        movie = cm.load(fname, in_memory=True)

        # Cat: TODO: fix the mmem map issue her
        print ("  TODO: fix memm map for motion correction files ...")
        temp_fname = '/home/cat/code/caiman_gui/temp.npy'
        np.save(temp_fname, movie)
        movie = np.load(temp_fname)

        self.data_min = np.float(movie.min())
        self.data_max = np.float(movie.max())

        print ("  finished loading movie, size: ", movie.shape)

        return movie
        
    def playmovie_loadscreen(self):
        # load next frame and compute movie counter

        self.movie_ctr = (self.movie_ctr+1)%self.movie_loadScreen.shape[0]
        image = self.normalize_frame(self.movie_loadScreen[self.movie_ctr],norm_global=True)

        # update slider location with current frame index
        self.loadSliderFrame.setValue(self.movie_ctr)
        # update slide frame box with current frame index
        self.loadLineEditFrameId.setText(str(self.movie_ctr))

        # display image
        self.displayImage(image, self.loadScreen)


    def generic_movie_load(self, slider_widget, list_widget, screen_widget):
        ''' Load movie, initialize first frame in screen widget and
            find length of image stack to assign to slider range.
        '''

        if len(list_widget.selectedIndexes())>0:
            fname = list_widget.selectedIndexes()[0].data()

            movie_array = self.loadmovie(fname)
            
            # set frame 0 on screen
            image = self.normalize_frame(movie_array[self.loadSliderFrame.value()],
                                                            norm_global=True)

            self.displayImage(image, screen_widget)

            # set slider range based on size of movie
            # Cat: TODO: the shape[0] value may not always be correct
            slider_widget.setRange(0,movie_array.shape[0]-1)

        else:
            print (list_widget.selectedIndexes())
            QMessageBox.about(self, "Error", "No file selected, please select in text field.")
            print ("No file selected...")

        return movie_array

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
        img = self.update_image(self.loadSliderFrame, self.movie_loadScreen)

        # call the generic slider with slider ID and target widget ID
        self.generic_slider_func(img,self.loadSliderFrame, self.loadScreen, 
                                                    self.movie_loadScreen)

    def loadSliderMinIntensity_func(self):
        # call the generic slider with slider ID and target widget ID
        self.slider_min = np.float(self.loadSliderMinIntensity.value())
        
        # don't allow sliders to overlap
        if self.slider_min>=self.slider_max:
            self.loadSliderMinIntensity.setValue(self.slider_max)
            return

        img = self.update_image(self.loadSliderFrame, self.movie_loadScreen)
        self.generic_slider_func(img, self.loadSliderFrame, 
                                    self.loadScreen, self.movie_loadScreen)

    def loadSliderMaxIntensity_func(self):
        # call the generic slider with slider ID and target widget ID
        self.slider_max = np.float(self.loadSliderMaxIntensity.value())
        
        # don't allow sliders to overlap
        if self.slider_max<=self.slider_min:
            self.loadSliderMaxIntensity.setValue(self.slider_min)
            return
            
        img = self.update_image(self.loadSliderFrame, self.movie_loadScreen)

        self.generic_slider_func(img, self.loadSliderFrame, 
                                        self.loadScreen, self.movie_loadScreen)


    def motionSliderMinIntensity_func(self):
        # call the generic slider with slider ID and target widget ID
        self.slider_min = np.float(self.motionSliderMinIntensity.value())
        
        # don't allow sliders to overlap
        if self.slider_min>=self.slider_max:
            self.motionSliderMinIntensity.setValue(self.slider_max)
            return

        img = self.update_image(self.motionSliderFrame, self.movie_motionScreen)
        self.generic_slider_func(img, self.motionSliderFrame, 
                                self.motionScreen, self.movie_motionScreen)


    def motionSliderMaxIntensity_func(self):
        # call the generic slider with slider ID and target widget ID
        self.slider_max = np.float(self.motionSliderMaxIntensity.value())
        
        # don't allow sliders to overlap
        if self.slider_max<=self.slider_min:
            self.motionSliderMaxIntensity.setValue(self.slider_min)
            return
            
        img = self.update_image(self.motionSliderFrame, self.movie_motionScreen)

        self.generic_slider_func(img,self.motionSliderFrame, 
                                self.motionScreen, self.movie_motionScreen)


    def motionSliderFrame_func(self):
        # call the generic slider with slider ID and target widget ID
        self.movie_ctr=self.motionSliderFrame.value()
        self.motionLineEditFrameId.setText(str(self.movie_ctr))

        # load img
        img = self.update_image(self.motionSliderFrame, self.movie_motionScreen)
        self.generic_slider_func(img, self.motionSliderFrame, 
                            self.motionScreen, self.movie_motionScreen)
                
                        
    def generic_slider_func(self, image, slider_widget, screen_widget,
                            movie_array):
        ''' Takes slider widget and its value and displays in target
            screen widget
        '''
        # check to see if movie loaded
        if len(movie_array)==0:
            print ("No movie loaded ...")
            return

        # display frame
        img = self.update_image(slider_widget, movie_array)

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
        img = ((img-self.data_min)/(self.data_max-self.data_min))*99
        return img
        

    def update_image(self, slider_widget, movie_array, img=None):

        # load single image
        if img is None:
            img = movie_array[slider_widget.value()]

        # set intensity based on slider values
        img = self.normalize_frame(img,norm_global=True)

        return img
        #self.displayImage(img, self.loadScreen)

    def displayImage(self, image_raw, screen_widget):
        ''' Displays single image in a target screenwidget
        '''
        # standardized code for convering image to
        # Cat: TODO: is all this formatting necessary?
        # Cat: TODO: also, can we just cast opencv imshow to the widget?
        
        #image_raw = np.array(image_raw, dtype='uint8')
        
        #fname = '/home/cat/code/caiman_gui/test.npy'
        #np.save(fname, image_raw)
        #image_raw = np.load(fname)
                
        # convert from opencv format to pyqt QImage format
        qformat=QImage.Format_Grayscale8

        img=QImage(image_raw,image_raw.shape[1],image_raw.shape[0],
                                        image_raw.strides[0],qformat)
        #img=QImage(image_raw,image_raw.shape[1],image_raw.shape[0],
        #                                80,qformat)
        
        pixmap = QtGui.QPixmap(img)

        # Stretch image to fit screen
        img_width, img_height = image_raw.shape
        screen_width = screen_widget.frameGeometry().width()
        screen_height = screen_widget.frameGeometry().height()

        if screen_width/img_width*img_height > screen_height: 
            pixmap = pixmap.scaled(screen_width, 
                                   screen_width/img_height*img_width)
        else:
            pixmap = pixmap.scaled(screen_height/img_width*img_height,
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
    def init_params(self):
        self.params = dict()
        # initialize the dictionary of parameters
        for child in self.findChildren(QtWidgets.QWidget):
            try:
                if child.property('par_name') is None:
                    raise Exception()
                self.params[child.property('par_name')] = child
            except:
                print('Could not retrieve parameter information from widget:' + child.objectName())

        print([[k, d.text()] for k, d in self.params.items() if type(d) is QtWidgets.QLineEdit])
        print([[k, d.currentText()] for k, d in self.params.items() if type(d) is QtWidgets.QComboBox])
        print([[k, d.isChecked()] for k, d in self.params.items() if type(d) is QtWidgets.QCheckBox])


    def get_dict_param(self,name,input_type):
        if input_type == 'str':
            if name == 'method_init':
                return self.params[name].currentText()
            else:
                return self.params[name].text()
        elif input_type == 'tuple_int':
            return tuple(map(int, self.params[name].text().split(',')))
        if input_type == 'tuple_float':
            return tuple(map(int, self.params[name].text().split(',')))
        elif input_type == 'single_int':
            return int(self.params[name].text())
        elif input_type == 'single_float':
            return float(self.params[name].text())
        elif input_type == 'slices':
            if self.params[name].text() is not 'None':
                return [slice(*s) for s in eval(self.params[name].text())]
        else:
            raise Exception('Unknown type')

    def set_dict_param(self, name, value):
        if type(value) is tuple:
            self.params[name].setText(str(value)[1:-1])
        elif type == 'single_int':
            self.params[name].setText(str(value))
        else:
            raise Exception('Unknown type')

    def motion_correct_rigid(self, fname):
        dview = None
        try:

            c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

            niter_rig = 1  # number of iterations for rigid motion correction
            max_shifts = self.get_dict_param('max_shifts_rigid','tuple_int')
            # for parallelization split the movies in  num_splits chuncks across time
            splits_rig = self.get_dict_param('splits_rig','single_int')
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

    def motion_correct_pwrigid(self, fname):
        dview = None
        try:
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)

            niter_rig = 1  # number of iterations for rigid motion correction
            max_shifts = self.get_dict_param('max_shifts_pwrigid','tuple_int')  # maximum allow rigid shift
            # for parallelization split the movies in  num_splits chuncks across time
            splits_rig = self.get_dict_param('splits_rig','single_int')
            # start a new patch for pw-rigid motion correction every x pixels
            strides = self.get_dict_param('strides','tuple_int')
            # overlap between pathes (size of patch strides+overlaps)
            overlaps = self.get_dict_param('overlaps','tuple_int')
            # for parallelization split the movies in  num_splits chuncks across time
            splits_els = self.get_dict_param('splits_els','single_int')

            upsample_factor_grid = self.get_dict_param('upsample_factor_grid','single_int') # upsample factor to avoid smearing when merging patches
            # maximum deviation allowed for patch with respect to rigid shifts
            max_deviation_rigid = self.get_dict_param('max_deviation_rigid','single_int')
            # first we create a motion correction object with the parameters specified
            min_mov = cm.load(fname[0], subindices=range(200)).min()
            # this will be subtracted from the movie to make it non-negative


            print(str([max_shifts, splits_rig, strides, overlaps, splits_els, upsample_factor_grid, max_deviation_rigid, min_mov]))
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

    def memory_map_files(self, files):
        n_processes = self.get_dict_param('n_processes_mmap', 'single_int')
        dview = None
        try:
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=n_processes, single_thread=False)

            resize_fact = self.get_dict_param('resize_fact', 'tuple_float')
            add_to_movie = self.get_dict_param('add_to_movie', 'single_float')
            border_to_0 = self.get_dict_param('border_to_0', 'single_int')
            n_chunks = self.get_dict_param('n_chunks', 'single_int')
            slices = self.get_dict_param('slices', 'slices')

            new_file = save_memmap(files, base_name='memmap_', resize_fact=resize_fact, order='C', add_to_movie=add_to_movie, border_to_0=border_to_0, dview=dview,
                        n_chunks=n_chunks, slices=slices)



            return new_file


        except Exception as e:
            raise e

        finally:
            cm.cluster.stop_server(dview=dview)

    def load_params_CNMF(self, file,  is_patch):

        opts_dict = {
            'tsub': self.get_dict_param('tsub', 'single_int'),
            'ssub': self.get_dict_param('ssub', 'single_int'),
            'fnames': file,
            'decay_time': self.get_dict_param('decay_time', 'single_float'),
            'fr': self.get_dict_param('frate', 'single_float'),
            'nb': self.get_dict_param('gnb', 'single_int'),
            'gSig': self.get_dict_param('gSig', 'tuple_int'),
            'method_init': self.get_dict_param('method_init', 'str'),
            'rolling_sum': True,
            'merge_thr': self.get_dict_param('merge_thresh', 'single_float'),
            'n_processes': self.get_dict_param('n_processes_cnmf', 'single_int'),
        }

        if is_patch:
            opts_dict['K'] = self.get_dict_param('k_patch', 'single_int')
            opts_dict['rf'] = self.get_dict_param('rf', 'single_int'),
            opts_dict['stride'] =  self.get_dict_param('stride', 'single_int'),
        else:
            opts_dict['K'] = self.get_dict_param('k', 'single_int')

        return CNMFParams(params_dict=opts_dict)


    def run_cnmf(self, file, is_patch):
        dview = None
        try:
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=self.get_dict_param('n_processes_cnmf', 'single_int'), single_thread=False)


            Yr, dims, T = cm.load_memmap(file[0])
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            opts = self.load_params_CNMF(file, is_patch=is_patch)
            opts.set('temporal', {'p': 0})
            cnm = CNMF(self.get_dict_param('n_processes_cnmf', 'single_int'), params=opts, dview=dview)
            cnm = cnm.fit(images)
            cnm.params.set('temporal', {'p': self.get_dict_param('p', 'single_int')})
            if is_patch:
                cnm2 = cnm.refit(images)
                cnm2.save(file[0][:-4] + 'hdf5')
            else:
                cnm.save(file[0][:-4] + 'hdf5')

            print('SAVED FILE ' + file[0][:-4]+'hdf5')
            return(file[0][:-4]+'hdf5')

        except Exception as e:
            raise e

        finally:
            cm.cluster.stop_server(dview=dview)

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

    def normalize_frame(self,img,norm_global=False):
        if norm_global:
            _min = self.data_min
            _max = self.data_max
        else:
            _min = img.min()
            _max = img.max()

        img = (img - _min) / ( _max- _min) * 99
        img = (np.clip((img-self.slider_min)/
              (self.slider_max-self.slider_min),0,1)*255).astype(np.uint8)
        return img





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
            
