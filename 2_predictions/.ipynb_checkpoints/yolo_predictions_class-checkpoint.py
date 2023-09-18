#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np 
import os 
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
         #Load YAML
        with open(data_yaml, mode = 'r') as f:
             data_yaml = yaml.load(f, Loader=SafeLoader)
        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        #Load YOLO Model onnx
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        #Set using cpu by prefereble backand through OpenCv
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # WE have CPU Envoirment thats y using CPU if yoy have GPU use CUDA  

    def predictions(self, image):
        # claculating the shape here d is RGB Color three sensities of the image 
        row, col, d = image.shape   
        #get the YOLO prediction from the image 
        
        #step-1 convert image into square image(array)
        max_rc = max(row, col) # calculating the max rows and max column 
        input_image = np.zeros((max_rc, max_rc,3), dtype=np.uint8) 
        #overlay actual image into Empty image
        input_image[0:row, 0:col] = image
        
        #Step-2 get prediction from square array 
        INPUT_WIDTH_YOLO = 640 # input width of rows and column 
        INPUT_HEIGHT_YOLO = 640 # input height of rows and column
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH_YOLO, INPUT_HEIGHT_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        #Pass the Yolo to Neural Network
        preds = self.yolo.forward() #detection or prediction from yolo 
        
        
        
        # you can see how actual convert square matrix image with he help of Yolo Model
        # #  Number of Rows 25200 is Bounding Boxes detected by the Yolo model 
        # #  Each and every rows has those information which is column 
        # ![2023-09-12.png](attachment:19406941-2e98-4e86-99a9-aa63d447c412.png)
        
        # Non Maximum Supression which is removing the duplicate Bounding boxes/ detection
        
        #step:-1 filter detection based on confidence (0.4) and probability score (0.25)
        detections = preds[0]
        # detections.shape
        # defining empty boxes, confidences and classes
        boxes = [] 
        confidences = []
        classes = []
        
        # Defining width and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/INPUT_WIDTH_YOLO
        y_factor = image_h/INPUT_HEIGHT_YOLO
        
        for i in range(len(detections)):
            row = detections[i]
            confidence_val = row[4] #confidence of detection on object
            if confidence_val > 0.4:
                class_score = row[5:].max() #maximum probability from 20 Objects
                class_id = row[5:].argmax() #get the index position at which max probabilty occur 
        
                if class_score  > 0.25:
                    cx, cy, w, h = row[0:4]
                    #construct bounding from four values 
                    #left , top, width and height 
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width  = int(w*x_factor)
                    height = int(h*y_factor)
        
                    box_val = np.array([left, top, width, height])
        
                    #append values into the list 
                    confidences.append(confidence_val)
                    boxes.append(box_val)
                    classes.append(class_id)
        
        #clean duplicate values 
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        #classes_np = np.array(classes).tolist()
        
        
        #Non Maximum Supression 
        index =  cv2.dnn.NMSBoxes(boxes_np, confidences_np,0.25,0.45).flatten()
        # above index shows rows we need to consider for detections 
        
        #len(index) # number of face detection length 
        
        
        # Draw The Bounding Box
        for ind in index:
            #print(ind) # print index position 
            #extract bounding 
            x,y,w,h = boxes_np[ind]
            bounding_box_conf = int(confidences_np[ind]*100)
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colors = self.generate_colors(classes_id)
            
            text = f'{class_name}: {bounding_box_conf}%'
            #print(text)
            cv2.rectangle(image, (x,y), (x+w, y+h), colors,2)
            cv2.rectangle(image, (x,y-30), (x+w, y), colors,-1)
        
            cv2.putText(image,text,(x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7,(0,0,0),1)
        
        return image 

# generate color for diffrent objects 
    def generate_colors(self,ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc,3)).tolist()
        return tuple(colors[ID])

























