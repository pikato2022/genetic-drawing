import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import DNA
from IPython.display import clear_output

class GeneticDrawing:
    def __init__(self, img_path, seed=0, brushesRange=[[0.1, 0.3], [0.3, 0.7]]):
        self.original_img = cv2.imread(img_path)
        self.img_grey = cv2.cvtColor(self.original_img,cv2.COLOR_BGR2GRAY)
        self.img_grads = self._imgGradient(self.img_grey)
        self.myDNA = None
        self.seed = seed
        self.brushesRange = brushesRange
        self.sampling_mask = None
        
        #start with an empty black img
        self.imgBuffer = [np.zeros((self.img_grey.shape[0], self.img_grey.shape[1]), np.uint8)]
        
    def generate(self, stages=10, generations=100, brushstrokesCount=10, show_progress_imgs=True):
        for s in range(stages):
            #initialize new DNA
            if self.sampling_mask is not None:
                sampling_mask = self.sampling_mask
            else:
                sampling_mask = self.create_sampling_mask(s, stages)
            self.myDNA = DNA(self.img_grey.shape, 
                             self.img_grads, 
                             self.calcBrushRange(s, stages), 
                             canvas=self.imgBuffer[-1], 
                             sampling_mask=sampling_mask)
            self.myDNA.initRandom(self.img_grey, brushstrokesCount, self.seed + time.time() + s)
            #evolve DNA
            for g in range(generations):
                self.myDNA.evolveDNASeq(self.img_grey, self.seed + time.time() + g)
                clear_output(wait=True)
                print("Stage ", s+1, ". Generation ", g+1, "/", generations)
                if show_progress_imgs is True:
                    # plt.imshow(sampling_mask, cmap='gray')
                    plt.imshow(self.myDNA.get_cached_image(), cmap='gray')
                    plt.show()
            self.imgBuffer.append(self.myDNA.get_cached_image())
        return self.myDNA.get_cached_image()
    
    def calcBrushRange(self, stage, total_stages):
        return [self._calcBrushSize(self.brushesRange[0], stage, total_stages), self._calcBrushSize(self.brushesRange[1], stage, total_stages)]
        
    def set_brush_range(self, ranges):
        self.brushesRange = ranges
        
    def set_sampling_mask(self, img_path):
        self.sampling_mask = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2GRAY)
        
    def create_sampling_mask(self, s, stages):
        percent = 0.2
        start_stage = int(stages*percent)
        sampling_mask = None
        if s >= start_stage:
            t = (1.0 - (s-start_stage)/max(stages-start_stage-1,1)) * 0.25 + 0.005
            sampling_mask = self.calc_sampling_mask(t)
        return sampling_mask
        
    '''
    we'd like to "guide" the brushtrokes along the image gradient direction, if such direction has large magnitude
    in places of low magnitude, we allow for more deviation from the direction. 
    this function precalculates angles and their magnitudes for later use inside DNA class
    '''
    def _imgGradient(self, img):
        #convert to 0 to 1 float representation
        img = np.float32(img) / 255.0 
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #normalize magnitudes
        mag /= np.max(mag)
        #lower contrast
        mag = np.power(mag, 0.3)
        return mag, angle
    
    def calc_sampling_mask(self, blur_percent):
        img = np.copy(self.img_grey)
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #calculate blur level
        w = img.shape[0] * blur_percent
        if w > 1:
            mag = cv2.GaussianBlur(mag,(0,0), w, cv2.BORDER_DEFAULT)
        #ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
        scale = 255.0/mag.max()
        return mag*scale

    def _calcBrushSize(self, brange, stage, total_stages):
        bmin = brange[0]
        bmax = brange[1]
        t = stage/max(total_stages-1, 1)
        return (bmax-bmin)*(-t*t+1)+bmin