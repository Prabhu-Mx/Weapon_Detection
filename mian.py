import cv2
import numpy as np
import argparse

praser = argparse.ArgumentParser()
praser.add_argument("-i","--input_file", required=True, help="path to input video")
praser.add_argument("-o","--output_file", required=True, help="path to output video")
args = vars(praser.parse_args())

def detect():
    # Load input video
    cap = cv2.VideoCapture(args["input file"])

    # Get input video width and height
    width, height = int(cap.get(3)), int(cap.get(4))

    # If you wannna save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('y2output_video.avi', fourcc, 20.0, (width, height))

    # Load Yolo - give the correct path of yolo weights and cfg file
    net = cv2.dnn.readNet("path to weights", "path to config file(.cfg)")

    # Name custom object
    classes = ["weapon"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Start checking frame by frame of input video file
    while cap.isOpened():
        ret, img = cap.read()
        if ret == True:
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        # Object detected
                        print(class_id)
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, label, (x, y + 50), font, 1, (0, 0, 255), 2)

            # writing the video file
            output.write(img)
            cv2.imshow(args["output file"], img)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    detect()
