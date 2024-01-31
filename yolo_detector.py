import cv2
import numpy as np

# --------------- READ DNN MODEL ---------------
# Model configuration
config = "model/yolov3.cfg"
# Weights
weights = "model/yolov3.weights"
# Labels
LABELS = open("model/coco.names").read().split("\n")
#print(LABELS, len(LABELS))
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
#print("colors.shape:", colors.shape)

# Load model
net = cv2.dnn.readNetFromDarknet(config, weights)

# --------------- READ THE IMAGE AND PREPROCESSING ---------------
image = cv2.imread("images/laptop.jpg")
height, width, _ = image.shape

# Create a blob
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                              swapRB=True, crop=False)
#print("blob.shape:", blob.shape)

# --------------- DETECTIONS AND PREDICTIONS ---------------
ln = net.getLayerNames()
#print("ln:", ln)

# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] 
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
#print("ln:", ln)

net.setInput(blob)
outputs = net.forward(ln)
#print("outputs:", outputs)

boxes = []
confidences = []
classIDs = []

for output in outputs:
     for detection in output:
          #print("detection:", detection)
          scores = detection[5:]
          classID = np.argmax(scores)
          confidence = scores[classID]

          if confidence > 0.5:
               #print("detection:", detection)
               #print("classID:", classID)
               box = detection[:4] * np.array([width, height, width, height])
               #print("box:", box)
               (x_center, y_center, w, h) = box.astype("int")
               #print((x_center, y_center, w, h))
               x = int(x_center - (w / 2))
               y = int(y_center - (h / 2))
               #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

               boxes.append([x, y, w, h])
               confidences.append(float(confidence))
               classIDs.append(classID)

idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
print("idx:", idx)

if len(idx) > 0:
     for i in idx:
          (x, y) = (boxes[i][0], boxes[i][1])
          (w, h) = (boxes[i][2], boxes[i][3])

          color = colors[classIDs[i]].tolist()
          text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
          cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
          cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                         0.5, color, 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()