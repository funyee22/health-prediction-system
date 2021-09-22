from tkinter import *
import tkinter.font as font
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import numpy as np
import pickle
from lbp import get_pixel, lbp_calculated_pixel
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


class GUI:

    def __init__(self, master):
        master.title("Health Prediction System")
        master.minsize(1300, 700)
        master.configure(bg="slate gray")  # set background color

        # value initializations
        self.hidden1 = 0
        self.hidden2 = 0
        self.uploadBtnPressed = 0

        # set reusable fonts
        self.font1 = font.Font(family='Karmatic Arcade', size=28)
        self.font2 = font.Font(family='Terminal', size=18)
        self.font3 = font.Font(family='OCR A Extended', size=16)
        self.font4 = font.Font(family='OCR A Extended', size=14)
        self.font5 = font.Font(family='Mom´sTypewriter', size=15)
        self.font6 = font.Font(family='Terminal', size=22)

        self.frame1(master)

    def createFrame(self, _frame):
        frame = Frame(_frame, bg='slate gray')  # create reusable frame
        return frame

    def frame1(self, master):
        # Page 1

        global healthy
        global fever
        global r_nose
        global s_throat

        self.frame_1 = self.createFrame(master)

        mainTitle = Label(self.frame_1, text="Health  Prediction  System", bg='slate gray', font=self.font1, pady=80)
        mainTitle.pack()

        img_frame = self.createFrame(self.frame_1)

        image1 = Image.open('healthy.png')
        image1 = image1.resize((150, 170))
        healthy = ImageTk.PhotoImage(image1)
        label = Label(img_frame, image=healthy)
        label.grid(row=0, column=1, padx=(0, 20))

        image2 = Image.open('fever.png')
        image2 = image2.resize((150, 170))
        fever = ImageTk.PhotoImage(image2)
        label = Label(img_frame, image=fever)
        label.grid(row=0, column=2, padx=(0, 20))

        image3 = Image.open('sore throat.png')
        image3 = image3.resize((150, 170))
        s_throat = ImageTk.PhotoImage(image3)
        label = Label(img_frame, image=s_throat)
        label.grid(row=0, column=3, padx=(0, 20))

        image4 = Image.open('running nose.png')
        image4 = image4.resize((150, 170))
        r_nose = ImageTk.PhotoImage(image4)
        label = Label(img_frame, image=r_nose)
        label.grid(row=0, column=4, padx=(0, 20))

        img_frame.pack()

        description = Label(self.frame_1, text="Let me predict your health by using an image of you!", bg='slate gray',
                            font=self.font2, pady=20)
        description.pack()

        start = Button(self.frame_1, text="Start", font=self.font3, fg='white', bg='black', padx=10,
                       relief=RAISED, bd=1)
        start.bind("<Button-1>", self.hide_me)
        start.pack(pady=40)

        self.frame_1.pack(fill=BOTH)

    def frame2(self, master):
        # Page 2

        global camera

        self.frame_2 = self.createFrame(master)

        instruction = Label(self.frame_2,
                            text="Please upload an upright image of your face (uncovered)\nor oral cavity (back of your mouth)",
                            bg='slate gray',
                            font=self.font2)
        instruction.grid(pady=(50, 20))
        cam_img = Image.open('camera.png')
        cam_img = cam_img.resize((25, 25))
        camera = ImageTk.PhotoImage(cam_img)
        uploadBtn = Button(self.frame_2, text=" Upload from computer", font=self.font4, fg='white', bg='black',
                           image=camera,
                           compound=LEFT, padx=8,
                           relief=RAISED,
                           bd=1)
        uploadBtn.bind("<Button-1>", self.openFile)
        uploadBtn.grid(pady=(0, 15))
        self.frame_2.pack()

    def hide_me(self, event):
        # For frame navigation purposes

        if self.hidden1 == 0:
            self.frame_1.destroy()
            self.hidden1 = 1
            if self.hidden1 == 1:
                self.frame2(root)
                print("Navigate to page 2")

        elif self.hidden2 == 0:
            self.frame_2.destroy()
            self.hidden2 = 1
            if self.hidden2 == 1:
                self.frame3(root)
                print("Navigate to page 3")

    def openFile(self, event):
        # To select image to upload and confirm

        if self.uploadBtnPressed == 0:
            self.uploadBtnPressed = 1
        else:
            self.uploadBtnPressed = 2

        if self.uploadBtnPressed is 2 and self.filename is not '':
            print("img destroyed")
            self.showImg.destroy()
            self.confirm.destroy()

        self.filename = filedialog.askopenfilename(initialdir="C:/", title='Select An Image', filetypes=(
            ("All Files", "*.*"), ("PNG files", "*.png"), ("JPG files", "*.jpg")))
        print(self.filename)

        if self.filename is not '':
            self.displayImg(self.filename)
            # print(self.uploadBtnPressed)
            self.confirm = Button(self.frame_2, text="Confirm", font=self.font4, fg='white', bg='black', padx=10,
                                  relief=RAISED, bd=1)
            self.confirm.bind("<Button-1>", self.hide_me)
            self.confirm.grid(pady=(10, 0))

    def displayImg(self, path):
        # To display selected image

        global img

        self.selectedImg = Image.open(path)

        self.w, self.h = self.selectedImg.size

        self.imgSize()  # call this function to adjust the image size for display

        self.resizedImg = self.selectedImg.resize(self.size)
        img = ImageTk.PhotoImage(self.resizedImg)
        self.showImg = Label(self.frame_2, image=img)
        self.showImg.grid(row=3, padx=(0, 0), pady=(15, 10), sticky=N)

    def frame3(self, master):
        # Page 3

        global faceImage

        CLASSES = ["NORMAL", "ILL"]
        SUBCLASSES = ["FEVER", "SORE THROAT", "RUNNING NOSE"]

        self.read_img = cv2.imread(self.filename)
        self.crop_image()
        self.lbp()
        self.classifier()

        pred1 = "Health Prediction: " + CLASSES[self.y_pred1[0]]
        if self.y_pred1 == 1:
            pred2 = "Suspected Symptom: " + SUBCLASSES[self.y_pred2[0]]

        self.frame_3 = self.createFrame(master)

        yourResult = Label(self.frame_3, text="Here is your result:", bg='slate gray',
                           font=self.font6)
        yourResult.grid(row=0, pady=(50, 20))

        faceImage = ImageTk.PhotoImage(self.resizedImg)
        showImage = Label(self.frame_3, image=faceImage)
        showImage.grid(pady=(15, 20))

        resultFrame = self.createFrame(self.frame_3)
        firstPred = Label(self.frame_3, text=pred1, bg=self.labelBgColor(), width=30, pady=2,
                          font=self.font5)
        firstPred.grid()

        if self.y_pred1 == 1:
            secondPred = Label(self.frame_3, text=pred2, bg=self.labelBgColor(), width=30, pady=2,
                               font=self.font5)
            secondPred.grid()
        resultFrame.grid()

        self.frame_3.pack()

    def crop_image(self):
        # Detect & crop face from image

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(self.read_img, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(self.read_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        try:
            cropped_img = self.read_img[y:y + h, x:x + w]
            self.X_test = cv2.resize(cropped_img, (200, 200))
            print(self.X_test.shape)
        except Exception:
            print("Face not detected, original image is used")
            self.X_test = cv2.resize(self.read_img, (200, 200))
            print(self.X_test.shape)

    def lbp(self):
        # Extract LBP features

        height, width, channel = 200, 200, 3

        self.X_test = cv2.cvtColor(self.X_test, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((height, width, 3), np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(self.X_test, i, j)
        self.X_test = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
        self.X_test = np.array(self.X_test).reshape(1, -1)
        print(self.X_test.shape)

    def classifier(self):
        # Load saved feature scaler and model for first and second-level classification

        fname = 'scaler.pickle'
        scaler = pickle.load(open(fname, 'rb'))
        self.X_test = scaler.transform(self.X_test)

        filename = "lbp_mlp1.sav"
        model1 = pickle.load(open(filename, 'rb'))
        self.y_pred1 = model1.predict(self.X_test)
        print(self.y_pred1)

        if self.y_pred1 == 1:
            filename = "lbp_mlp2.sav"
            model2 = pickle.load(open(filename, 'rb'))
            self.y_pred2 = model2.predict(self.X_test)
            print(self.y_pred2)

    def imgSize(self):
        # To adjust displayed image size

        if self.h <= 250:
            self.size = (int(self.w * 2), int(self.h * 2))
        elif 251 <= self.h <= 500:
            self.size = (int(self.w * 0.6), int(self.h * 0.6))
        elif 501 <= self.h <= 1000:
            self.size = (int(self.w * 0.4), int(self.h * 0.4))
        elif 1001 <= self.h <= 2300:
            self.size = (int(self.w * 0.2), int(self.h * 0.2))
        else:
            self.size = (int(self.w * 0.1), int(self.h * 0.1))

    def labelBgColor(self):
        # To set the background color of the result label

        color = 'green2'
        if self.y_pred1 == 1:
            color = 'firebrick1'

        return color


root = Tk()
root.iconphoto(False, PhotoImage(file='stethoscope.png'))
gui = GUI(root)
root.mainloop()