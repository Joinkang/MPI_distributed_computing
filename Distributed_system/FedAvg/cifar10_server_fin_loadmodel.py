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
    #print(f"rank = {rank}, size = {size}")

    (train_images, train_labels),  (test_images, test_labels) = cifar10.load_data()

    train_labels = to_categorical(train_labels, num_classes=10)
    train_images = train_images / 255.0
    test_labels = to_categorical(test_labels, num_classes=10)
    test_images = test_images / 255.0


    model, optimizer = generate_model()

    path = "/home/mpiuser/FedAvg_1steps/"
    # path = "/home/mpiuser/class_weight_1steps/"





    file_list = os.listdir(path)
    file_list_h5 = [file for file in file_list if file.endswith(".h5")]




    list_num = [int(i[-6:-3]) for i in file_list_h5]   
    max_list_index = np.argmax(list_num)

    print(f"client- rank {rank:2d} @@@@@@@@@@@@@@@@@@@@@@@ load model {file_list_h5[max_list_index]} @@@@@@@@@@@@@@@@@@@@@@@")    
    
    start_epoch = np.max(list_num) 

    
    print(f"client- rank {rank:2d} @@@@@@@@@@@@@@@@@@@@@@@ start_epoch {start_epoch+1} @@@@@@@@@@@@@@@@@@@@@@@")


    for i in range(2, size):
        comm.send(start_epoch, dest=i) ###0!!!
    

    model = tf.keras.models.load_model(f'/home/mpiuser/class_weight_1steps/'+file_list_h5[max_list_index])


    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
         
    #model.summary()


    for epoch in range((start_epoch+1), epochs):
        start_time = time.time()
        for i in range(2, size):

            #print(f"send the init model to rank {i} ")
            initial_weights = model.get_weights()
            comm.send(initial_weights, dest=i) 

        test = [comm.recv(source=i) for i in range(2, size)]
        #print(f"epoch {epoch:2d} weight {np.array(test).shape} ") 
        model.set_weights(federated_averaging(test,size))

        print(f"server- epoch {epoch:2d}, running time : {time.time()-start_time:5.3f}")
                        
        print(f"rank = {rank} get weight : epoch {epoch}")
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print('--------------------------')
        print(f'\n! Test accuracy: {test_acc} , Loss : {test_loss}\n')
        print('--------------------------')
        
        if (epoch % 10) == 0 :
            model.save(f"/home/mpiuser/FedAvg_1steps/{rank:02d}_epoch{epoch:03d}.h5")
            # model.save(f"/home/mpiuser/FedAvg_5steps/{rank:02d}_epoch{epoch:03d}.h5")
            # model.save(f"/home/mpiuser/FedAvg_10steps/{rank:02d}_epoch{epoch:03d}.h5")
            # model.save(f"/home/mpiuser/FedAvg_20steps/{rank:02d}_epoch{epoch:03d}.h5")


print(f"rank {rank} server done")

MPI.Finalize()

