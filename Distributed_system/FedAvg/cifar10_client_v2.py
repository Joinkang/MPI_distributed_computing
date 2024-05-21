import tensorflow as tf
import numpy as np
from mpi4py import MPI
import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D , Flatten, Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

from MPI_functions import *



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.keras.backend.clear_session()



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Train the model
epochs =20
batch_size = 16
class_num = 10



print(f"rank = {rank}")

if rank != 1:


    # CIFAR-10 데이터셋 로드
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    num_class_for_sample = len(train_images)//class_num    
    # 데이터 전처리: 이미지 픽셀 값을 0~1 범위로 조정
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # 레이블을 one-hot 인코딩
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    model, optimizer = generate_model()

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    print(f"client- rank = {rank:2d} start init")


    for epoch in range(epochs):   

        print(f"client- rank = {rank:2d} finish init")
        initial_weights = comm.recv(source=0)

        model.set_weights(initial_weights)
         
        indices = data_chop(num_class_for_sample,class_num)

        #print(f"size of training data set in rank {rank} = {num_class_for_sample//5*class_num}")
        model.fit(train_images[indices],train_labels[indices], batch_size=16, epochs=20,verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print('client- rank ', rank , ', Test accuracy: ', test_acc, ', Test loss: ', test_loss)
        comm.send(model.get_weights(), dest=0)
        print(f"client- rank = {rank:2d} epoch {epoch} send weight")
        
print(f"rank {rank} client done")

MPI.Finalize()
