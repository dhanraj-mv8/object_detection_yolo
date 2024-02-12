import cv2
import argparse
from imutils.video import webcamvideostream
import numpy as np
import requests

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
save ="C:\\Users\\Admin\\Desktop\\object-detection-opencv-master"
url="http://192.168.137.44:8080/shot.jpg"
while True:
    ##ret, frame = cam.read()
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
    frame= cv2.imdecode(img_arr,-1)
    cv2.imshow("test", frame)
    ##if not ret:
        ##break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame.png"
        cv2.imwrite("C:\\Users\\Admin\\Desktop\\object-detection-opencv-master\\opencv_frame.jpg", frame)
        #print("{} written!".format(img_name))
        img_counter += 1
        break

cam.release()

cv2.destroyAllWindows()

image = cv2.imread("C:\\Users\\Admin\Desktop\\object-detection-opencv-master\\opencv_frame.jpg")

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
a1=[]
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    ax=(round(x)+round(x+w))/2
    ay=(round(y)+round(y+h))/2
    a1.append([ax,ay])
    

cv2.imshow("DETECTED OBJECT", image)

cv2.waitKey()

print(a1)
cv2.imwrite("object-detection.jpg", image)
cv2.destroyAllWindows()
