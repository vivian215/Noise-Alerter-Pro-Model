# -*- coding: utf-8 -*-
# Household security-related sound classification

import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

def extract_features(sample_chunk, sample_rate, feature_type, feature_dim):
    if (feature_type == "mfcc_mean"):
        mfccs = librosa.feature.mfcc(y=sample_chunk, sr=sample_rate, n_mfcc=feature_dim, hop_length=1024)
        features = np.mean(mfccs.T,axis=0)
    elif (feature_type == "mfcc"):
        features = librosa.feature.mfcc(y=sample_chunk, sr=sample_rate, n_mfcc=feature_dim, hop_length=1024)
    elif (feature_type == "mel_spectro_mean"):
        mel_spectrogram = librosa.feature.melspectrogram(sample_chunk, sample_rate, n_fft=2048, hop_length=1024, n_mels=feature_dim, fmax=8000)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        features = np.mean(log_mel_spectrogram.T,axis=0) 
    elif (feature_type == "mel_spectro"):
        mel_spectrogram = librosa.feature.melspectrogram(sample_chunk, sample_rate, n_fft=2048, hop_length=1024, n_mels=feature_dim, fmax=8000)
        features = librosa.power_to_db(mel_spectrogram)

    return features

def train(file_path, filenames, model_type, feature_type, feature_hop_length, feature_dim):
    
    extracted_features=[]
    for filename in filenames:
        audio_filename = file_path + filename + '_long.wav'
        audio_samples, sample_rate = librosa.load(audio_filename)
    
        #split samples into chunks
        sample_chunks = [audio_samples[x:x+split_size] for x in range(0, len(audio_samples), feature_hop_length)] #17 #7
        for c in range(1, len(sample_chunks)):
            #print("chunk: %d" % c)
            chunk = sample_chunks[c]
            if (len(chunk) < split_size):
                continue
    
            feature = extract_features(chunk, sample_rate, feature_type, feature_dim)
            
            extracted_features.append([feature, filename])
            
    extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
    print(extracted_features_df.head())
    
    X=np.array(extracted_features_df['feature'].tolist())
    y=np.array(extracted_features_df['class'].tolist())
    
    
    
    label_encoder = LabelEncoder()
    y=to_categorical(label_encoder.fit_transform(y))
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    
    num_labels=y.shape[1]
    
    print(num_labels)
    
    
    if model_type == "DNN" :
        model=Sequential()
        # hidden layer 1
        model.add(Dense(128,input_shape=(feature_dim,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # hidden layer 2
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # hidden layer 3
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        # output layer
        model.add(Dense(num_labels))
        model.add(Activation('softmax'))
        
        
        model.summary()
        
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    
    elif model_type == "CNN" :
    
        #convolutional neural network model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(feature_dim, 21, 1)))
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
        model.add(MaxPool2D((2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))          
        model.add(Dense(num_labels, activation='softmax'))
        
        model.summary()
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
        
    elif model_type == "CNN2" :
    
        #convolutional neural network model with extra pooling layer
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(feature_dim, 21, 1)))
        model.add(MaxPool2D((2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
        model.add(MaxPool2D((2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))          
        model.add(Dense(num_labels, activation='softmax'))
        
        model.summary()
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    
    # train model
    if (model_type == "CNN" or model_type == "CNN2"):
        num_epochs = 20 #16
    else:
        num_epochs = 100 #100
        
    num_batch_size = 32
    
    checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                                   verbose=1, save_best_only=True)
    
    training_history = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
    
    test_accuracy=model.evaluate(X_test,y_test,verbose=0)

    if (plot_training_history == True) :
        print(test_accuracy[1])
        
        # list all data in history
        print(training_history.history.keys())
        
        # plot accuracy
        plt.plot(training_history.history['accuracy'])
        plt.plot(training_history.history['val_accuracy'])
        plt.title('CNN model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.show()
        
        # plot training loss history
        plt.plot(training_history.history['loss'])
        plt.plot(training_history.history['val_loss'])
        plt.title('CNN model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    return model, training_history, label_encoder

def test(model, file_path, filenames, model_type, feature_type, lbl_encoder, feature_dim) :    

    correct = 0
    total_tests = 6
    for i in range(0, total_tests):
        filename = filenames[i%3]
        audio_filename = file_path + filename + '_test.wav'
        audio_samples, sample_rate = librosa.load(audio_filename)
        
        x = 10000*i;
        chunk = audio_samples[x:x+split_size]
        print(len(chunk))

        mfccs = extract_features(chunk, sample_rate, feature_type, feature_dim)
        if (model_type == "CNN" or model_type == "CNN2") :
            feature = mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1)
        else:
            feature = mfccs.reshape(1,-1)
       
        result = model.predict(feature);     
        predicted_label=np.argmax(result, axis=1)
        print(predicted_label)
        prediction_class = lbl_encoder.inverse_transform(predicted_label) 
        print(prediction_class[0] + ' =====>' + filename)
        if (prediction_class[0] == filename):
            correct = correct + 1
            print('PASS: idx: {} pass: {} start: {} samp: {} result: {}'.format(i, correct, x, chunk[0], result))
        else:
            print('FAIL: result: {}'.format(result))


def compare_cnn_dnn():
    
    feature_hop_length = int(20480 / 17)
    
    models = []
    histories = []
    for i in range(0, 2):
        print (i)
        if (i == 1):
            nnet_model_type = "CNN2"
        else:
            nnet_model_type = "DNN"
            
        if (nnet_model_type == "CNN" or nnet_model_type == "CNN2" ) :
            nnet_feature_type = "mfcc"
        elif (nnet_model_type == "DNN") :
            nnet_feature_type = "mfcc_mean"

            
        model_trained, history, label_encoder = train(audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, feature_hop_length, feature_frequency_dim)
        models.append(model_trained)
        histories.append(history)
        
        test(model_trained, audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, label_encoder, feature_frequency_dim)
    
    # plot comparison result for CNN and DNN
    plt.plot(histories[0].history['accuracy'])
    plt.plot(histories[1].history['accuracy'])
    plt.title('CNN vs DNN model training - accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['DNN', 'CNN'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['val_accuracy'])
    plt.plot(histories[1].history['val_accuracy'])
    plt.title('CNN vs DNN model training - validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['DNN', 'CNN'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['loss'])
    plt.plot(histories[1].history['loss'])
    plt.title('CNN vs DNN model training - loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['DNN', 'CNN'], loc='upper right')
    plt.show()
    
    plt.plot(histories[0].history['val_loss'])
    plt.plot(histories[1].history['val_loss'])
    plt.title('CNN vs DNN model training - validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['DNN', 'CNN'], loc='upper right')
    plt.show()
# end of function compare_cnn_dnn()

def compare_mfcc_melspectrogram():
    feature_hop_length = int(20480 / 17)
    
    models = []
    histories = []
    for i in range(0, 2):
        print (i)

        nnet_model_type = "DNN"
        if (i == 0):
            nnet_feature_type = "mfcc_mean"
        else:
            nnet_feature_type = "mel_spectro_mean"
            
        model_trained, history, label_encoder = train(audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, feature_frequency_dim)
        models.append(model_trained)
        histories.append(history)
        
        test(model_trained, audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, label_encoder, feature_frequency_dim)
    
    # plot comparison result for CNN and DNN
    plt.plot(histories[0].history['accuracy'])
    plt.plot(histories[1].history['accuracy'])
    plt.title('MFCCs vs MelSpectrogram feature - accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['MFCC', 'MEL_SPECTRO'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['val_accuracy'])
    plt.plot(histories[1].history['val_accuracy'])
    plt.title('MFCCs vs MelSpectrogram feature - validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['MFCC', 'MEL_SPECTRO'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['loss'])
    plt.plot(histories[1].history['loss'])
    plt.title('MFCCs vs MelSpectrogram feature - loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['MFCC', 'MEL_SPECTRO'], loc='upper right')
    plt.show()
    
    plt.plot(histories[0].history['val_loss'])
    plt.plot(histories[1].history['val_loss'])
    plt.title('MFCCs vs MelSpectrogram feature - validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['MFCC', 'MEL_SPECTRO'], loc='upper right')
    plt.show()
# end of function compare_mfcc_melspectro()

def compare_time_shift():
    feature_hop_length = int(20480 / 17)
    
    models = []
    histories = []
    for i in range(0, 2):
        print (i)

#        nnet_model_type = "DNN"
#        nnet_feature_type = "mfcc_mean"
        nnet_model_type = "CNN"
        nnet_feature_type = "mfcc"
        if (i == 0):
            feature_hop_length = int(20480 / 17)

        else:
            feature_hop_length = 20480 # no time shift
            
        model_trained, history, label_encoder = train(audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, feature_hop_length, feature_frequency_dim)
        models.append(model_trained)
        histories.append(history)
        
        test(model_trained, audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, label_encoder, feature_frequency_dim)
    
    # plot comparison result for CNN and DNN
    plt.plot(histories[0].history['accuracy'])
    plt.plot(histories[1].history['accuracy'])
    plt.title('Time Shift vs No Time Shift - accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Time Shift', 'No Time Shift'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['val_accuracy'])
    plt.plot(histories[1].history['val_accuracy'])
    plt.title('Time Shift vs No Time Shift - validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Time Shift', 'No Time Shift'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['loss'])
    plt.plot(histories[1].history['loss'])
    plt.title('Time Shift vs No Time Shift - loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Time Shift', 'No Time Shift'], loc='upper right')
    plt.show()
    
    plt.plot(histories[0].history['val_loss'])
    plt.plot(histories[1].history['val_loss'])
    plt.title('Time Shift vs No Time Shift - validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Time Shift', 'No Time Shift'], loc='upper right')
    plt.show()


from IPython.display import Image 
from keras.utils import plot_model
def compare_feature_frequency_slot():
    feature_hop_length = int(20480 / 3)  #17
    
    models = []
    histories = []
    for i in range(0, 2):
        print (i)

#        nnet_model_type = "DNN"
#        nnet_feature_type = "mfcc_mean"
        nnet_model_type = "CNN"
        nnet_feature_type = "mfcc"
        if (i == 0):
            feature_frequency_dim = 36
        else:
            feature_frequency_dim = 64
            
        model_trained, history, label_encoder = train(audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, feature_hop_length, feature_frequency_dim)
        models.append(model_trained)
        histories.append(history)
        
        plot_model(model_trained, show_shapes=True, show_layer_names=True, to_file='model.png')
        Image('model.png')

        tf.keras.utils.plot_model(
            model_trained,
            to_file="model.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
            layer_range=None,
            show_layer_activations=True)
        
        
        test(model_trained, audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, label_encoder, feature_frequency_dim)
    
    # plot comparison result for CNN and DNN
    plt.plot(histories[0].history['accuracy'])
    plt.plot(histories[1].history['accuracy'])
    plt.title('frequency slots 32 vs 64 - accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['n_mfcc = 32', 'n_mfcc = 64'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['val_accuracy'])
    plt.plot(histories[1].history['val_accuracy'])
    plt.title('frequency slots 32 vs 64 - validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['n_mfcc = 32', 'n_mfcc = 64'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['loss'])
    plt.plot(histories[1].history['loss'])
    plt.title('frequency slots 32 vs 64 - loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['n_mfcc = 32', 'n_mfcc = 64'], loc='upper right')
    plt.show()
    
    plt.plot(histories[0].history['val_loss'])
    plt.plot(histories[1].history['val_loss'])
    plt.title('frequency slots 32 vs 64 - validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['n_mfcc = 32', 'n_mfcc = 64'], loc='upper right')
    plt.show()

def compare_cnn_cnn2():
    
    feature_hop_length = int(20480 / 17)
    
    models = []
    histories = []
    for i in range(0, 2):
        print (i)
        if (i == 1):
            nnet_model_type = "CNN"
        else:
            nnet_model_type = "CNN2"
            
        if (nnet_model_type == "CNN" or nnet_model_type == "CNN2") :
            nnet_feature_type = "mfcc"
        elif (nnet_model_type == "DNN") :
            nnet_feature_type = "mfcc_mean"

            
        model_trained, history, label_encoder = train(audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, feature_hop_length, feature_frequency_dim)
        models.append(model_trained)
        histories.append(history)
        if (i == 0) :
            keras_file = audio_file_path + 'model_cnn_62.h5'
        else :
            keras_file = audio_file_path + 'model_cnn2_62.h5'
        tf.keras.models.save_model(model_trained, keras_file)
        test(model_trained, audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, label_encoder, feature_frequency_dim)
    
    # plot comparison result for CNN and DNN
    plt.plot(histories[0].history['accuracy'])
    plt.plot(histories[1].history['accuracy'])
    plt.title('CNN vs CNN2 model training - accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['CNN2', 'CNN'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['val_accuracy'])
    plt.plot(histories[1].history['val_accuracy'])
    plt.title('CNN vs DNN model training - validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['CNN2', 'CNN'], loc='lower right')
    plt.show()
    
    plt.plot(histories[0].history['loss'])
    plt.plot(histories[1].history['loss'])
    plt.title('CNN vs DNN model training - loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['CNN2', 'CNN'], loc='upper right')
    plt.show()
    
    plt.plot(histories[0].history['val_loss'])
    plt.plot(histories[1].history['val_loss'])
    plt.title('CNN vs DNN model training - validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['CNN2', 'CNN'], loc='upper right')
    plt.show()
# end of function compare_cnn_dnn()

plot_training_history = False
audio_file_path = 'c:/projects/NoiseAlerterPro/Python/audio/'
audio_filenames = ['smoke_detector', 'knocking', 'walking', 'glass', 'doorbell', 'dog', 'noise'];

split_size = 20480
feature_frequency_dim = 62

feature_hop_length = int(20480 / 17)
    
nnet_model_type = "CNN2"
nnet_feature_type = "mfcc"
model_trained, history, label_encoder = train(audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, feature_hop_length, feature_frequency_dim)
keras_file = audio_file_path + 'model_cnn2_62.h5'
tf.keras.models.save_model(model_trained, keras_file)
test(model_trained, audio_file_path, audio_filenames, nnet_model_type, nnet_feature_type, label_encoder, feature_frequency_dim)

#compare_cnn_cnn2()
#compare_cnn_dnn()
#compare_mfcc_melspectrogram()
#compare_time_shift()
#compare_feature_frequency_slot()
 
sys.exit();

