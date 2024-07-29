from module.camera import Camera 
from module.AI_model import AI_model_factory, Yolo, MobileSAM
class CamServer():
    def __init__(self):
        self.camera = Camera()
        self.yolo = AI_model_factory.create(Yolo)
        self.sam = AI_model_factory.create(MobileSAM)
        self.yolo_results
    def detect_leaf(self):
        image = self.camera.capture_rgb_img()
        return self.yolo.predict(image)
    
    def segment_leaf(self, image):
        results = self.detect_leaf(image)
        return self.sam.predict(results)
        
    def select_leaf(self, yolo_results, index):
        return list(yolo_results[index])
    
    def get_leaf_center(self):
        
        pass
    
    def get_leaf_corners(self):
        
        pass    
        
        
    def get_leaves_center(self):
        pass
    def show_segment_result(self):
        pass
    def show_detection_result(self):
        pass
    