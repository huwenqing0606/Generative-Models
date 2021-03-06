"""
用生成对抗网络生成MNIST手写数字样本的样例程序
运行环境: Keras 2.3.1 and Tensorflow 1.14.0
参考文献: Goodfellow, I. et al, Generative Adversarial Nets, NIPS 2014
作者：胡文清
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from scipy.special import comb, perm
import numpy as np
from PIL import Image
import argparse
import math
from class_GAN import GAN
import matplotlib.pyplot as plt

# 工作路径
workpath = "\\."


# 读入数据 X 是图像, y 是标签
def load_data():
    # image_data_format选择"channels_last"或"channels_first"，该选项指定了Keras将要使用的维度顺序。
    # "channels_first"假定2D数据的维度顺序为(channels, rows, cols)，3D数据的维度顺序为(channels, conv_dim1, conv_dim2, conv_dim3)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()    
    # 转换字段类型，并将数据导入变量中
    # MNIST转换为 X_train: (60000, 28, 28, 1), X_test: (10000, 28, 28, 1), y_train: (60000, ), y_test: (10000, )
    X_train = (X_train.astype(np.float32) - 127.5)/127.5 # 每个像素由0到255的整数变为-1到1之间
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    return X_train, y_train, X_test, y_test


# 选择给定标签集合 label_set 的训练图像
def select_data_with_label(X, y, label_set):
    X_selected = []
    for index in range(len(X)):
        if (y[index] in label_set):
            X_selected.append(X[index])
    return X_selected


# 生成网络 G
def generator_model(input_dim):
    # 搭建生成器的架构，首先导入序贯模型（sequential），即多个网络层的线性堆叠
    model = Sequential()
    # 添加一个全连接层，输入为 input_dim 维向量，输出为1024维，激活函数为 tanh
    model.add(Dense(input_dim=input_dim, output_dim=1024))
    model.add(Activation('tanh'))
    # 添加一个全连接层，输出为128×7×7维度
    model.add(Dense(128*7*7))
    # 添加一个批量归一化层，该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # 添加一个Reshape层用来将输入shape转换为特定的shape，将含有128*7*7个元素的向量转化为7×7×128张量
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    # 添加一个2维上采样层，即将数据的行和列分别重复2次
    model.add(UpSampling2D(size=(2, 2)))
    # 添加一个2维卷积层，卷积核大小为5×5，激活函数为tanh，共64个卷积核，并采用padding以保持图像尺寸不变
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    # 添加一个2维上采样层，即将数据的行和列分别重复2次
    model.add(UpSampling2D(size=(2, 2)))
    # 卷积核设为1即输出图像的维度
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


# 判别网络 D
def discriminator_model():
    # 判别器架构采用序贯模型
    model = Sequential()
    # 添加一个 2 维卷积层，卷积核大小为 5×5，激活函数为tanh，填充至与输入尺寸相等
    # 输入shape在‘channels_first’模式下为（samples,channels，rows，cols), 在‘channels_last’模式下为（samples,rows,cols,channels）
    # 输出为 64 维, 即 64 个 5x5 卷积过滤器同时作用
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    # 添加一个最大值池化层，pool_size取（2，2）代表使图片在两个维度上均变为原长的一半
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 添加一个 2 维卷积层，卷积核大小为 5x5，激活函数为tanh
    # 输出为 128 维, 即 128 个 5x5 卷积过滤器同时作用
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    # 添加一个最大值池化层，pool_size取（2，2）代表使图片在两个维度上均变为原长的一半
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 添加一个Flatten层把多维输入一维化，常用在从卷积层到全连接层的过渡
    model.add(Flatten())
    # 添加一个全连接层，激活函数为tanh, 输出为 1024 维
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    # 一个结点进行二值分类，并采用sigmoid函数的输出作为概率
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


# 生成网络 G
def generator_model_fullconnect(input_dim):
    model = Sequential(name='generator_fully_connected')
    model.add(Dense(input_dim=input_dim, output_dim=1024))
    model.add(Activation('relu'))
    model.add(Dense(128*7*7))
    model.add(Activation('relu'))
    model.add(Dense(28*28*1))
    model.add(Activation('relu'))
    model.add(Reshape((28,28,1), input_shape=(28*28*1,)))
    return model


# 判别网络 D
def discriminator_model_fullconnect():
    model = Sequential(name='discriminator_fully_connected')
    model.add(Reshape((28*28*1,), input_shape=(28,28,1)))
    model.add(Dense(28*28*1))
    model.add(Activation('relu'))
    model.add(Dense(128*7*7))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


# 将生成网络和判别网络拼接成一个大的神经网络，用于判别生成的图像
# Goodfellow et al文章中的 D(G(z))
# 此处仅用于代码分析与测试
def generator_containing_discriminator(generator, discriminator):
    #采用序贯模型
    model = Sequential()
    #先添加生成器g架构，再令判别器d不可训练，即固定d
    #因此在给定判别器的情况下训练生成器，即通过将生成的结果投入到判别器进行辨别而优化生成器
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


# 生成图片拼接, 即生成器批量生成图像后按照 BATCH_SIZE 拼接成一张图
# 仅工具程序, 故不作进一步注释
def combine_images(images):
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]), dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image


# 用GAN训练生成器和判别器
# train_inputs = [ (28,28,1) , ... , (28,28,1) ] 为输入的训练图片集合
# 此处仅用于代码分析与测试
def GAN_train(train_inputs, BATCH_SIZE):
    # D 为判别网络, G 为生成网络, D_compose_G 为固定判别器的生成网络，生成后输出判别结果
    D = discriminator_model()
    G = generator_model(input_dim=100)
    D_compose_G = generator_containing_discriminator(generator=G, discriminator=D)
    # 定义生成器模型判别器模型更新所使用的优化算法及超参数
    D_optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
    G_optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
    # 编译三个神经网络并设置损失函数和优化算法，其中损失函数都是用的是二元分类交叉熵函数。编译是用来配置模型学习过程的
    G.compile(loss='binary_crossentropy', optimizer="SGD")
    D_compose_G.compile(loss='binary_crossentropy', optimizer=G_optimizer)
    # 前一个架构在固定判别器的情况下训练了生成器，所以在训练判别器之前先要设定其为可训练。
    D.trainable = True
    D.compile(loss='binary_crossentropy', optimizer=D_optimizer)
    # 训练若干个epoch
    # 计算一个epoch所需要的迭代数量，即训练样本数除batchsize的值取整
    train_size = len(train_inputs)
    num_iteration = int(train_size/BATCH_SIZE)#int(comb(len(train_inputs), BATCH_SIZE))
    for epoch in range(10):
        # 在一个epoch内进行迭代训练
        for index in range(num_iteration):
            ##### 训练判别器 #####            
            # 隐变量 Z 服从100维均匀分布U[-1,1], 输出BATCH_SIZE个样; 即抽取一个批量的随机样本z_1,...,z_{BATCH_SIZE}
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            # 直接抽取一个批量的真实图片 , 大小为BATCH_SIZE
            real_images_batch = train_inputs[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            # 随机抽取一个批量的真实图片 , 大小为BATCH_SIZE
            #batch_labels = np.random.choice(train_size, BATCH_SIZE).astype(int)
            #real_images_batch = np.array(train_inputs)[batch_labels]            
            # 输入隐变量Z, 使用生成器生成一个批量的生成图片; verbose为日志显示, 0为不在标准输出流输出日志信息, 1为输出进度条记录
            generated_images = G.predict(noise, verbose=0)
            # 将真实的图片和生成的图片以多维数组的形式拼接在一起，真实图片在上，生成图片在下
            X = np.concatenate((real_images_batch, generated_images))
            # 生成图片真假标签，即一个包含两倍批量大小的列表；前一个批量大小都是1，代表真实图片，后一个批量大小都是0，代表伪造图片
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE            
            # 根据 X 和 y 训练一次判别器 D，更新参数
            D_loss = D.train_on_batch(X, y)
            print("epoch %d maximum iteration %d batch number %d discriminator_loss : %f" % (epoch, num_iteration, index, D_loss))
            
            ##### 训练生成器 #####
            # 隐变量 Z 服从100维均匀分布U[-1,1]，输出BATCH_SIZE个样本；即抽取一个批量的随机样本z_1,...,z_{BATCH_SIZE}
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            # 固定判别器 D
            D.trainable = False
            # 给定判别器，训练一次生成器 G
            G_loss = D_compose_G.train_on_batch(noise, [1] * BATCH_SIZE)
            # 解锁判别器可训练
            D.trainable = True
            print("epoch %d maximum iteration %d batch number %d generator_loss : %f" % (epoch, num_iteration, index, G_loss))
            
            # 每经过100次迭代输出一张生成的图片
            if index % 100 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(workpath+"\\images\\"+str(epoch)+"_"+str(index)+".png")
            
            # 每100次迭代保存一次生成器和判别器的权重
            if index % 100 == 9:
                G.save_weights('generator', True)
                D.save_weights('discriminator', True)
                

if __name__=='__main__':
    # 选择给定标签的MNIST训练图像
    X_train, y_train, X_test, y_test = load_data()
    X_selected = select_data_with_label(X_train, y_train, label_set=[6])
    # 检查神经网络结构
    check_network_arch = 0
    # 检查是否是正确的图像，或直接训练GAN
    check_train_inputs = 0
    # 从测试代码训练
    run_GAN_testcode = 0
    if check_network_arch:
        model = generator_model(input_dim=10)
        model.summary()
        model = discriminator_model()
        model.summary()
        model = generator_model_fullconnect(input_dim=10)
        model.summary()
        model = discriminator_model_fullconnect()
        model.summary()
    elif check_train_inputs:
        for k in range(int(len(X_selected)/100)):
            image = combine_images(np.array(X_selected)[k:k+100])
            image = image*127.5+127.5
            Image.fromarray(image.astype(np.uint8)).save(workpath+"\\images\\图_MNIST训练图片_"+str(k)+".png")
    elif run_GAN_testcode:
        GAN_train(train_inputs = X_selected, BATCH_SIZE = 12)
    else:
        noise_dim = 2
        BATCH_SIZE = 12
        train_epoch = 1
        # 从封装好的GAN类训练
        # 封装好的GAN类，可以直接调用
        GAN = GAN(generator = generator_model_fullconnect(input_dim=noise_dim),
                  discriminator = discriminator_model_fullconnect(),
                  noise_dim = noise_dim,
                  noise_type = 'uniform')
        # 训练
        D_loss_record, G_loss_record = GAN.train(train_inputs = X_selected,
                                                 G_optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True),
                                                 D_optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True),
                                                 BATCH_SIZE = BATCH_SIZE,
                                                 sampling = 'deterministic_sweep', # 'random_uniform',
                                                 num_epoch = train_epoch,
                                                 num_iter = int(len(X_selected)/BATCH_SIZE) # int(comb(len(train_inputs), BATCH_SIZE))
                                                 )
        # 生成
        generated_samples = GAN.generate(num=100)
        # 输出生成图片
        image = combine_images(generated_samples)
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save(workpath+"\\images\\图_MNIST生成图片.png")
        # 输出训练损失函数，供调参用
        plt.plot(D_loss_record, color='red', label='discriminator')
        plt.plot(G_loss_record, color='blue', label='generator')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend(['discriminator', 'generator'], loc='best')
        plt.savefig(workpath+"\\images\\图_MNIST损失函数.png")
        plt.show()
        
