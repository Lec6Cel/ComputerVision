import cv2

videoPath ="../datasets/video"
imgPath ="D:/meme/roi.png"
logo = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, (100,100))

cap = cv2.VideoCapture(0) #load from camera
#cap = cv2.VideoCapture(imgPath) #load from image
#cap = cv2.VideoCapture("D:\code\Cuong.mkv") #load from Video
font = cv2.FONT_HERSHEY_SIMPLEX
while (cap.isOpened):

    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame,'Le Huy Cuong', (50,50), font, 1, (0,255,255), 2, cv2.LINE_4)
    frame[100:200 ,0:100] = cv2.addWeighted(frame[100:200, 0:100], 0.7, logo, 0.3, 0)

    cv2.imshow('Video with Text',frame)

    if cv2.waitKey(0) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
