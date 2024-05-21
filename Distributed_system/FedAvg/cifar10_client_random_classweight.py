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

def data_chop(num_class_for_sample,arr,class_num):
        indices = []
        for i in range(class_num):
            choice_sample = list(range(i*num_class_for_sample,(i+1)*num_class_for_sample))
            indices += list(np.random.choice(choice_sample, arr[i], replace=False))
       
        return indices

def arr_maker(num_class_for_sample, rank, class_num, random = False):
    
    row = rank-2 #rank
    cols = class_num #label

    arr = [1000 for i in range(cols)]
    if random :
            for col in range(cols):
                if  row == (col) : arr[col] = np.random.randint(num_class_for_sample//2,num_class_for_sample)
                else : 
                    if row < (cols+1) : arr[col] = np.random.randint(num_class_for_sample//10,num_class_for_sample//2)
                    else : pass

    #print(np.array(arr))
    #np.sum(arr,axis=1)
    return arr

def class_weight(arr): # each rank node
    sample_weights_arr = []

    #print(arr)
    # for rankarr in arr :
    #         sample_weights_arr.append(rankarr/np.sum(rankarr))
    reverse_arr = [1 / x for x in arr]
    sample_weights_arr = reverse_arr/np.sum(reverse_arr)
    # print(np.sum(sample_weights_arr))
    return sample_weights_arr


def weighted_crossentropy(sample_weights_arr): 
    def loss(y_true, y_pred): 
        weights = sample_weights_arr
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon()) #tf.keras.backend.epsilon() = 1e-07
        weighted_losses = weights * y_true * tf.math.log(y_pred) * len(sample_weights_arr)
        return (-tf.reduce_sum(weighted_losses, axis=-1))
    return loss


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


    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_uniform'),
    Dense(10, activation='softmax')
])


    optimizer = SGD(learning_rate=0.001, momentum=0.9)  


    print(f"client- rank = {rank:2d} start init")


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
        model.fit(train_images[indices],train_labels[indices], batch_size=16, epochs=20,verbose=0)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        print(f'client- rank {rank:2d}, Test accuracy: {test_acc:.3f}, Test loss: {test_loss:2.3f}')
        comm.send(model.get_weights(), dest=0)
        print(f"client- rank = {rank:2d} epoch {epoch} send weight")
        
print(f"rank {rank} client done")

MPI.Finalize()
