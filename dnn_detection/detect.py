
import cv2

#importing and using necessary files
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

#Loading Tenserflow pretrained model by using dnn_DetctionModel module
model = cv2.dnn_DetectionModel(frozen_model,config_file)

#Reading Coco dataset
classLabels=[]
filename='models/yolo3.txt'
with open(filename,'rt') as fpt:
  classLabels = fpt.read().rstrip('\n').split('\n')

#input image preproccessing
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#reading image
img = cv2.imread('street.jpg')

#object detection by using dnn_DetectionModel.detect module
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

#plotting boxes
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (0, 255, 0), 3)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 0, 255), thickness=3)

cv2.imshow("object detection", img)
cv2.waitKey()
cv2.destroyAllWindows()
