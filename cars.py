import cv2


video = cv2.VideoCapture('dashcam.mp4')

#Our pre-trained car classifier
Haar_File = 'cars.xml'

#create car Classifier
car_tracker = cv2.CascadeClassifier(Haar_File)

#Run forever until car stops or something 
while True:
    #Read the current frame
    (read_successful, frame) = video.read()

    #safe coding 
    if read_successful:
        #Must convert to grayscale
        grayscale_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars
    cars = car_tracker.detectMultiScale(grayscale_frame)

    #Draw rectangles around the cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 128*255), 2 )
    

    #Display the image with the faces spotted
    cv2.imshow('Car detector', frame )

    #Autoclose disabled with the help of wait key
    key= cv2.waitKey(1)

    print("Code Completed")

    if key==75 or key==107:
        break



    
