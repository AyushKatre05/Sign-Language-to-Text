import cv2
import numpy as np
import os

# Creating and Collecting Training Data
mode = 'trainingData'
directory = 'dataSet/' + mode + '/'
minValue = 70

capture = cv2.VideoCapture(0)
interrupt = -1

while True:
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)  # Simulating mirror Image

    # Getting count of existing images
    count = {}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        count[letter] = len(os.listdir(directory + letter.upper()))
    count['zero'] = len(os.listdir(directory + "0"))

    # Printing the count of each set on the screen
    cv2.putText(frame, "ZERO : " + str(count['zero']), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    for letter, value in count.items():
        if letter != 'zero':
            cv2.putText(frame, f"{letter.upper()} : {value}", (10, 60 + 10 * (ord(letter) - ord('a'))), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])

    # Drawing the ROI
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    cv2.imshow("Frame", frame)

    # Image Processing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Output Image after the Image Processing that is used for data collection
    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("test", test_image)

    # Data Collection
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
    if 97 <= interrupt <= 122:  # Check if a valid key ('a' to 'z') is pressed
        letter = chr(interrupt)
        directory_letter = letter.upper()
        if not os.path.exists(directory + directory_letter):
            os.makedirs(directory + directory_letter)
        cv2.imwrite(directory + directory_letter + '/' + str(count[letter]) + '.jpg', test_image)
    elif interrupt & 0xFF == ord('0'):  # Check for the '0' key
        if not os.path.exists(directory + "0"):
            os.makedirs(directory + "0")
        cv2.imwrite(directory + "0/" + str(count['zero']) + '.jpg', test_image)

capture.release()
cv2.destroyAllWindows()
