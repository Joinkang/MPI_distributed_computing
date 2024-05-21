import tensorflow as tf
import numpy as np
from mpi4py import MPI
import os
import time

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

epochs =200
batch_size = 16
class_num = 10


if rank ==0 :
    print(f"rank = {rank}, size = {size}")

    (train_images, train_labels),  (test_images, test_labels) = cifar10.load_data()

    train_labels = to_categorical(train_labels, num_classes=10)
    train_images = train_images / 255.0
    test_labels = to_categorical(test_labels, num_classes=10)
    test_images = test_images / 255.0


    model, optimizer = generate_model()


    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
         
    model.summary()


    for epoch in range(epochs):
        start_time = time.time()
        for i in range(2, size):

            #print(f"send the init model to rank {i} ")
            initial_weights = model.get_weights()
            comm.send(initial_weights, dest=i) 

        gather_weights = [comm.recv(source=i) for i in range(2, size)]
        model.set_weights(federated_averaging(gather_weights,size))

        print(f"server- epoch {epoch:2d}, running time : {time.time()-start_time:5.3f}")
                        
        print(f"rank = {rank} get weight : epoch {epoch}")
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print('--------------------------')
        print(f'\n! Test accuracy: {test_acc} , Loss : {test_loss}\n')
        print('--------------------------')
        
        if (epoch % 10) == 0 :
            model.save(f"/home/mpiuser/FedAvg_10steps_earlystopping/{rank:02d}_epoch{epoch:03d}.h5")
            # model.save(f"/home/mpiuser/FedAvg_5steps/{rank:02d}_epoch{epoch:03d}.h5")
            # model.save(f"/home/mpiuser/FedAvg_10steps/{rank:02d}_epoch{epoch:03d}.h5")
            # model.save(f"/home/mpiuser/FedAvg_20steps/{rank:02d}_epoch{epoch:03d}.h5")



print(f"rank {rank} server done")

MPI.Finalize()

