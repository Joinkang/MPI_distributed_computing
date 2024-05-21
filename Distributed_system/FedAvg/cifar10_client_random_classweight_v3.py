import tensorflow as tf
import numpy as np
from mpi4py import MPI
import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D , Flatten, Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

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


    model , optimizer = generate_model()

   
    for epoch in range(epochs):   

        print(f"client- rank = {rank:2d} finish init")
        initial_weights = comm.recv(source=0)

        model.set_weights(initial_weights)

        arr = arr_maker(num_class_for_sample, rank-2, class_num, True)
        
        sample_weights_arr = class_weight(arr)
        
        print(f'clent- rank {rank:2d} arr {arr}')
        print("output = ",end = '')
        [print(f"{i:.2f} ",end ='') for i in sample_weights_arr]
        print("\n")
        
        model.compile(loss=weighted_crossentropy(sample_weights_arr), optimizer=optimizer, metrics=['accuracy'])
         
        indices = data_chop(num_class_for_sample,arr,class_num)  # real rank = rank + 2 

        #print(f"size of training data set in rank {rank} = {num_class_for_sample//5*class_num}")
        model.fit(train_images[indices],train_labels[indices], batch_size=16, epochs=5,verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print(f'client- rank {rank:2d}, Test accuracy: {test_acc:.3f}, Test loss: {test_loss:2.3f}')
        comm.send(model.get_weights(), dest=0)
        print(f"client- rank = {rank:2d} epoch {epoch} send weight")
        
print(f"rank {rank} client done")

MPI.Finalize()
