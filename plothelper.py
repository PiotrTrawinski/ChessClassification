import json
import matplotlib.pyplot as plt
import numpy as np


def plot1(train_accuracy, validation_accuracy, train_loss, validation_loss):
    plt.figure(figsize=(4, 2))

    plt.subplot(121)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(range(len(train_loss)), train_loss, label="train")
    plt.plot(range(len(validation_loss)), validation_loss, label="valid")

    plt.subplot(122)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(range(len(train_accuracy)), train_accuracy, label="train")
    plt.plot(range(len(validation_accuracy)), validation_accuracy, label="valid")
    plt.gca().legend(loc="lower right")

    plt.show()


def plot2(tacc,vacc,tloss,vloss):
    Epoch_count=len(tloss)
    Epochs=[]
    for i in range (0,Epoch_count):
        Epochs.append(i+1)
    index_loss=np.argmin(vloss)
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    val_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1)
    vc_label='best epoch= '+ str(index_acc + 1)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1,val_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    #plt.style.use('fivethirtyeight')
    plt.show()


if __name__ == '__main__':
    history = json.load(open("history.json.txt", 'r'))

    plot1(history["accuracy"], history["val_accuracy"], history["loss"], history["val_loss"])
    #plot2(history["accuracy"], history["val_accuracy"], history["loss"], history["val_loss"])