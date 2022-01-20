import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def construcao():

    rede_neural = Sequential()
    rede_neural.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation='relu'))
    rede_neural.add(MaxPooling2D(pool_size=(2,2)))

    rede_neural.add(Conv2D(32, (3,3), activation='relu'))
    rede_neural.add(MaxPooling2D(pool_size=(2,2)))

    rede_neural.add(Flatten())

    rede_neural.add(Dense(units = 4, activation='relu'))
    rede_neural.add(Dense(units = 4, activation='relu'))
    rede_neural.add(Dense(units = 2, activation='softmax'))

    rede_neural.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    
    return rede_neural
    

def treinamento(rede_neural, base_treinamento, base_teste):

    rede_neural.fit_generator(base_treinamento, epochs=10, validation_data=base_teste)
    return rede_neural


def avaliacao(rede_neural,base_teste):

    prob = rede_neural.predict(base_teste)
    previsoes = np.argmax(prob, axis = 1)

    print("Acurácia: ", accuracy_score(previsoes, base_teste.classes))
    print(confusion_matrix(previsoes, base_teste.classes))


# Construção das bases de treinamento e teste

gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True, zoom_range=0.2)
base_treinamento = gerador_treinamento.flow_from_directory('personagens/training_set', target_size = (64, 64), batch_size = 8, class_mode = 'categorical')


gerador_teste = ImageDataGenerator(rescale=1./255)
base_teste = gerador_teste.flow_from_directory('personagens/test_set', target_size = (64, 64), batch_size = 8, class_mode = 'categorical', shuffle = False)

rede_neural = construcao()
rede_neural_treinada = treinamento(rede_neural, base_treinamento, base_teste)
avaliacao(rede_neural_treinada, base_teste)
