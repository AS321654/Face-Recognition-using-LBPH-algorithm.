import numpy as np
from numpy import ma
import cv2
import os
import math
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt
data = ["Kiran Bedi", "Indira Gandhi"]
def detect_face(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_classifier = cv2.CascadeClassifier(r'C:\Users\Ayushi\miniconda3\pkgs\libopencv-4.0.1-hbb9e17c_0\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
    face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=7)
    if len(face)==0:
        image_gray = ()
        return image_gray,face
    else:
        (x,y,w,h) = face[0]
        return image_gray[y:y+w, x:x+h], face[0]
def prepare_data(data_path):
    folders = os.listdir(data_path)
    labels = []
    faces = []
    for folder in folders:
        label = folder
        training_images_path = data_path + '/' + folder
        for image in os.listdir(training_images_path):
            image_path = training_images_path + '/' + image
            training_image = cv2.imread(image_path)
            face, bounding_box = detect_face(training_image)
            if len(bounding_box)==0 :
                break
            else:
                faces.append(face)
                labels.append(label)
    print ('Training Done')
    return faces, labels
faces, labels = prepare_data(r'C:\Users\Ayushi\Desktop\Training_Dataset')
print ('Total faces = ', len(faces))
print ('Total labels = ', len(labels))
def lbp_calculate(images):
     lbp_radius=1
     eps=1e-7
     lbp_numpoints=8
     len(images)
     lbp_features = []
     for im in images:
         lbp = local_binary_pattern(im, lbp_numpoints, lbp_radius, method='default')
         (hist, hist_len) = np.histogram(lbp.ravel(),bins=np.arange(0, 256))
         hist = hist.astype("float")
         hist /= (hist.sum()+eps)
         lbp_features.append(hist)
         lbp_feat=np.array(lbp_features)
     return lbp_feat
Hist_Train = lbp_calculate(faces)
def return_intersection(hist_2):
    rows, cols = (len(faces), 2) 
    intersection = [[0 for i in range(cols)] for j in range(rows)]   
    count = 0;
    for hist in Hist_Train:
        old_err_state = np.seterr(divide='raise')
        if(old_err_state is not )
        ignored_states = np.seterr(**old_err_state)
        res = ma.filled(log(ma.masked_equal(m, 0)), 0)
        subl = np.divide(hist,hist_2)
        suml = np.multiply(np.log(hist,subl),hist)
        distance = np.sum(suml)
        print(distance)
        intersection[count][0] = distance
        intersection[count][1] = count+1
        count = count+1
    return intersection
def predict_face(test_image):
    eps=1e-7
    img = test_image.copy()
    face, box = detect_face(img)
    if len(box)==0:
        print("couldn't detect face")
        return box
    lbp_radius=1
    lbp_numpoints=8
    lbp = local_binary_pattern(face, lbp_numpoints, lbp_radius, method='default')
    (Hist_test, hist_len) = np.histogram(lbp.ravel(),bins=np.arange(0, 256))
    Hist_test = Hist_test.astype("float")
    Hist_test/= (Hist_test.sum()+eps)
    fig = plt.figure() 
    plt.hist(lbp.ravel(), bins = np.arange(0,256))
    plt.title("Numpy Histogram")    
    # show plot 
    plt.show()
    print('test')
    print(Hist_test)
    Intersection = return_intersection(Hist_test)
    sort_inter = sorted(Intersection,key=lambda x: (x[0],x[1]))
    print(sort_inter[0][1])
    label_text = labels[sort_inter[0][1]-1]
    (x,y,w,h) = box
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, label_text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    return img,Hist_test
training_images_path = r'C:\Users\Ayushi\Desktop\Test images'
for image in os.listdir(training_images_path):
    image_path = training_images_path + '/' + image
    test_image = cv2.imread(image_path)
    predicted_image,histtest = predict_face(test_image)
    if len(predicted_image)==0:
        pass
    else:
        cv2.imshow('face recognition',predicted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
