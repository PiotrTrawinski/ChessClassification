import os
import numpy as np
import tensorflow as tf
import cv2
from glob import glob
import sys
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn import metrics

def resizeImage(image, maxWidth = None, maxHeight = None, inter = cv2.INTER_AREA):
    (imageHeight, imageWidth) = image.shape[:2]
    
    if maxWidth is None:
        maxWidth = imageWidth
    if maxHeight is None:
        maxHeight = imageHeight

    width       = maxWidth
    imageHeight = int(imageHeight * (width / imageWidth))
    height      = min(imageHeight, maxHeight)
    width       = int(width * (height / imageHeight))

    resized = cv2.resize(image, (width, height), interpolation = inter)
    return resized

def main():
    dataset_dir = '../data/MergedDataset/test'
    model_checkpoint_filepath = 'Models/checkpoint_run-20210121-165009-1.h5'
    model = tf.keras.models.load_model(model_checkpoint_filepath)

    labels = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
    class_dirs = glob(os.path.join(dataset_dir, "*"))
    allImages = []
    pred = []
    real = []
    
    # while True:
    #     imgagePath = input("path: ")
    #     img = cv2.imread(imgagePath)
    #     x = np.reshape(cv2.resize(img,(256,512)) / 255.0, [1,512,256,3])
    #     img = resizeImage(img, 300, 300)
    #     prediction = np.argmax(model.predict(x), axis=-1)[0]
    #     cv2.imshow(labels[prediction], img)
    #     cv2.waitKey(0)

    index = 0
    for class_dir in class_dirs:
        class_name = os.path.basename(os.path.normpath(class_dir))
        files = glob(os.path.join(class_dir, "*"))
        images = []
        reshapedImages = []
        for file in files:
            img = cv2.imread(file)
            images.append(resizeImage(img, 100, 100))
            img = cv2.resize(img,(256,512))
            img = img / 255.0
            reshapedImages.append(img)
        x = np.reshape(reshapedImages,[len(reshapedImages),512,256,3])
        classes = np.argmax(model.predict(x), axis=-1)
        for c in classes:
            pred.append(c)
        real += [index] * len(classes)
        index += 1
        
        classImgLabels = [[images[i], labels[classes[i]], class_name] for i in range(len(classes))]
        #for c in classImgLabels:
        #    if c[1] != c[2]:
        #        cv2.imshow(class_name + ' vs ' + c[1], c[0])
        #        cv2.waitKey(0)
        #print(class_name, classes)
        allImages += classImgLabels
    
    print(real)
    print(pred)
    cm = metrics.confusion_matrix(real, pred)
    df_cm = pd.DataFrame(cm, index = labels, columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for m in range(64):
        i = m + 20
        ax = fig.add_subplot(8, 8, m + 1, xticks=[], yticks=[])
        rgb = np.fliplr(allImages[i][0].reshape(-1,3)).reshape(allImages[i][0].shape)
        ax.imshow(rgb, cmap=plt.cm.binary, interpolation='nearest')
        clr = 'red'
        if allImages[i][1] == allImages[i][2]:
            clr = 'green'
        ax.text(10, 15, str(allImages[i][1]), color=clr, size='large')
    plt.figure()
    plt.show()

if __name__ == "__main__":
    main()
