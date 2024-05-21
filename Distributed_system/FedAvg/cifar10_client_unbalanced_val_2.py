import tensorflow as tf
import numpy as np
from mpi4py import MPI
import os


from tensorflow.keras.layers import Conv2D, MaxPooling2D , Flatten, Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

from MPI_functions_val import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.keras.backend.clear_session()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Train the model
epochs =200  #1 -> 200/ 5-> 40 / 10-> 20 / 20 -> 10
batch_size = 16
class_num = 10

if rank != 1:

    # CIFAR-10 데이터셋 로드
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    num_class_for_sample = len(train_images)//class_num    

    # train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # # 레이블을 one-hot 인코딩
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    ##val
    val_indices_percentage = 0.8
    val_indices_range = int(num_class_for_sample * val_indices_percentage)
    val_indices = []
    for i in range(class_num):
        val_indices += list(range(i*num_class_for_sample + val_indices_range ,(i+1)*num_class_for_sample))

    model, optimizer = generate_model()

    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    for epoch in range(epochs):   

        initial_weights = comm.recv(source=0)
        print(f"client- rank = {rank:2d} finish init")
        model.set_weights(initial_weights)

        arr = arr_maker(num_class_for_sample, rank-2, class_num, True)
         
        print(f'clent- rank {rank:2d} arr {arr}')
        
        indices = data_chop(num_class_for_sample,arr,class_num) # real rank = rank + 2 

        sum_of_datas = int(np.sum(arr))

        print(f'clent- sum_of_datas {sum_of_datas:05d}')

        aug_images = augment_data(train_images[indices])
        aug_images = aug_images.astype('float32') / 255.0
        dataset = tf.data.Dataset.from_tensor_slices((aug_images,train_labels[indices]))
        dataset = dataset.batch(16)


        #print(f"size of training data set in rank {rank} = {num_class_for_sample//5*class_num}")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001,verbose = 1,restore_best_weights=True)
        model.fit(dataset, epochs=10,verbose=0, validation_data=(train_images[val_indices].astype('float32') / 255.0, train_labels[val_indices]), callbacks=[early_stopping])
        # model.fit(train_images[indices],train_labels[indices], batch_size=16, epochs=5,verbose=0, validation_data=(train_images[val_indices], train_labels[val_indices]))
        # model.fit(train_images[indices],train_labels[indices], batch_size=16, epochs=10,verbose=0, validation_data=(train_images[val_indices], train_labels[val_indices]))
        # model.fit(train_images[indices],train_labels[indices], batch_size=16, epochs=20,verbose=0, validation_data=(train_images[val_indices], train_labels[val_indices]))
        
        
        # model.fit(train_images[indices],train_labels[indices], batch_size=16, epochs=20,verbose=0, validation_data=(train_images[val_indices], train_labels[val_indices]), callbacks=[early_stopping])

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print('client- rank ', rank , ', Test accuracy: ', test_acc, ', Test loss: ', test_loss)
        comm.send(model.get_weights(), dest=0)
        print(f"client- rank = {rank:2d} epoch {epoch} send weight")
        
print(f"rank {rank} client done")

MPI.Finalize()
