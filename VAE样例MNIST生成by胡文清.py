"""
用变分自动编码器生成MNIST手写数字样本的样例程序
运行环境: Keras 2.3.1 and Tensorflow 1.14.0
参考文献: Kingma, D.P. and Welling, M., Auto-Encoding Variational Bayes, arXiv:1312.6114, Dec. 2013.
作者：胡文清

单位：明略科技营销事业部综合服务部
给欣雨的提示：将此代码运用于人口属性特征概率向量的生成，只需要
    (1) 修改读入的数据 X 为历史人口属性向量，y 为触达特征的分类标签
    (2) 修改编码器设计层和解码器的神经网络结构
    (3) 测试不同的训练超参数
    (4) 直接调用封装好的 VAE 类 class_VAE 中的 VAE 
"""

from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, Adam
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from class_VAE import VAE

# 工作路径
workpath = "D:\\Temporary Files\\2021_08-12_秒针数据科学\\1_ID缺失监测方法论\\20210919基于生成模型的IDFA缺失监测\\15_变分自编码机\\code"


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
    X_selected = np.array(X_selected)
    return X_selected


# 编码器设计层
# 接受输入数据，产生设计层输出张量
def encoder_design(i):
    # 添加一个2维卷积层，卷积核大小为3×3，激活函数为relu，卷积步长为2，填充至与输入尺寸相等
    # 输入shape在‘channels_first’模式下为（samples,channels，rows，cols, 在‘channels_last’模式下为（samples,rows,cols,channels
    # 输出为 8 维, 即 8 个 3x3 卷积过滤器同时作用
    cx = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(i)
    # 添加一个批量归一化层，该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    cx = BatchNormalization()(cx)
    # 添加一个2维卷积层，卷积核大小为3×3，激活函数为relu，卷积步长为2，填充至与输入尺寸相等
    # 输出为 16 维, 即 16 个 3x3 卷积过滤器同时作用
    cx = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
    # 添加一个批量归一化层，该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    cx = BatchNormalization(name='conv')(cx)
    # 添加一个Flatten层把多维输入一维化，常用在从卷积层到全连接层的过渡
    x = Flatten()(cx)
    # 添加一个全连接层，输出为20维向量
    x = Dense(20, activation='relu')(x)
    # 添加一个批量归一化层，该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
    encoder_design_output = BatchNormalization(name='encoder_design_output')(x)
    # 返回编码器设计层的输出张量 encoder_design_output
    return encoder_design_output


# 解码器
# 输入隐含高斯变量，输出生成的数据
def decoder_model_conv_shape(noise_dim, conv_shape):
    # 解码器输入层
    d_i   = Input(shape=(noise_dim, ), name='decoder_input')
    x     = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
    x     = BatchNormalization()(x)
    x     = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    cx    = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    cx    = BatchNormalization()(cx)
    cx    = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
    cx    = BatchNormalization()(cx)
    o     = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
    # 生成解码器网络模型
    decoder_model = Model(d_i, o, name='decoder')
    return decoder_model


# 解码器
# 输入隐含高斯变量，输出生成的数据
def decoder_model(noise_dim):
    conv_shape = (None, 7, 7, 16)
    # 解码器输入层
    d_i   = Input(shape=(noise_dim, ), name='decoder_input')
    x     = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
    x     = BatchNormalization()(x)
    x     = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    cx    = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    cx    = BatchNormalization()(cx)
    cx    = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
    cx    = BatchNormalization()(cx)
    o     = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
    # 生成解码器网络模型
    decoder_model = Model(d_i, o, name='decoder')
    return decoder_model


# 重参数化层函数，将[mu, sigma]层转换为 Z = mu + eps*exp(sigma/2) 其中 eps ~ N(0, I)
# 此处仅用于代码分析与测试
def reparametrization_fn(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps


# VAE损失函数，因为涉及到编码器输出层，所以 keras 需要用二层函数嵌套技巧
# 如果基于 tensorflow 直接编码就没有这个问题
# 此处仅用于代码分析与测试
def VAE_loss(mu, sigma):
    ### 损失函数必须在整个计算图内部定义
    def kl_reconstruction_loss(true, pred):
        # 重构损失
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * 28 * 28
        # KL 散度损失
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # 总损失 = 50% 重构损失 + 50% KL 散度损失
        return K.mean(reconstruction_loss + kl_loss)
    return kl_reconstruction_loss


# 建立并训练 VAE, 返回网络结构和训练好的超参数
# 此处仅用于代码分析与测试
def VAE_train(train_inputs, noise_dim, optimizer, BATCH_SIZE, num_epochs, num_iteration):
    ### 编码器设计层, 输入数据，输出任意维数的张量
    # 输入层
    i = Input(shape=np.shape(train_inputs[0]), name='encoder_input')
    # 编码器设计层
    x = encoder_design(i)
    ### 编码器[mu, sigma]和重参数化层
    ### 在设计层基础上加载高斯隐变量 Z 的均值mu和方差sigma层[mu, sigma]，
    ### 然后加载重参数化层将[mu, sigma]层转换为 Z = mu + eps*sigma 其中 eps ~ N(0, I) 
    # 添加一个全连接层，输出为隐变量 Z 服从的高斯分布的均值
    mu = Dense(noise_dim, name='latent_mu')(x)
    # 添加一个全连接层，输出为隐变量 Z 服从的高斯分布的方差
    sigma = Dense(noise_dim, name='latent_sigma')(x)
    # 添加重参数化层，将[mu, sigma]层转换为 Z = mu + eps*exp(sigma/2) 其中 eps ~ N(0, I)
    Z = Lambda(reparametrization_fn, output_shape=(noise_dim, ), name='Z')([mu, sigma])
    ### 构建并审阅编码器
    encoder = Model(i, [mu, sigma, Z], name='encoder')
    encoder.summary()    
    ### 解码器，输入高斯隐变量Z，输出生成的数据
    # 获取编码器卷积层的尺寸参数用于解码
    conv_shape = K.int_shape(encoder.get_layer('conv').output)
    ### 构建并审阅解码器
    # 用 conv_shape 未知的 decoder_model
    decoder = decoder_model_conv_shape(noise_dim, conv_shape)
    decoder.summary()
    ### 构建并审阅VAE
    vae_outputs = decoder(encoder(i)[2])
    vae = Model(i, vae_outputs, name='vae')
    vae.summary()
    ### 编译VAE
    vae.compile(optimizer=optimizer, loss=VAE_loss(mu, sigma))
    ### 训练VAE
    # 训练若干个epoch
    loss_seq=[]
    for epoch in range(num_epochs):
        for index in range(num_iteration):
            # 构建批量训练数据点
            train_inputs_batch = train_inputs[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            # 训练
            loss = vae.train_on_batch(train_inputs_batch, train_inputs_batch)
            loss_seq.append(loss)
        # 每epoch保存一次生成器和判别器的权重
        encoder.save_weights('encoder_weights', True)
        decoder.save_weights('decoder_weights', True)
        # 打印loss
        print("epoch %d maximum iteration %d vae_loss : %f" % (epoch, num_iteration, loss))            
        # 输出训练中的生成图片
        generated_images=VAE_generate(decoder, noise_dim, 100)
        image = combine_images(generated_images)
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save(workpath+"\\images\\"+str(epoch)+".png")
    return encoder, decoder, loss_seq
    
       
# VAE 数据生成器
# 此处仅用于代码分析与测试
def VAE_generate(decoder, noise_dim, num_samples):
    decoder.load_weights('decoder_weights')
    generated_data = []
    for index in range(num_samples):
        Z_sample = np.random.normal(0,1,size=noise_dim)
        #Z_sample = Z_sample/np.linalg.norm(Z_sample)
        x_decoded=decoder.predict(np.array([Z_sample]))
        data=x_decoded[0].reshape(28,28,1)
        generated_data.append(data)
    generated_data=np.array(generated_data)
    return generated_data


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


# 测试 VAE_train
# 此处仅用于代码分析与测试
def experiment_VAE(train_inputs):
    # 训练VAE
    batch_size = 128
    num_epochs = 50
    noise_dim = 2
    encoder, decoder, loss_seq = VAE_train(train_inputs=train_inputs, 
                                           noise_dim=noise_dim, 
                                           optimizer=Adam(learning_rate=0.01),
                                           BATCH_SIZE=batch_size, 
                                           num_epochs=num_epochs,  
                                           num_iteration=int(len(train_inputs)/batch_size)
                                          )

    # 输出生成图片样例
    generated_images = VAE_generate(decoder, noise_dim, num_samples=100)

    image = combine_images(generated_images)
    image = image*127.5+127.5    
    Image.fromarray(image.astype(np.uint8)).save(workpath+"\\images\\图_MNIST生成图片_VAE.png")    

    # 输出训练损失函数，供调参用
    plt.plot(loss_seq, color='blue', label='vae_loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(['vae_loss'], loc='best')
    plt.savefig(workpath+"\\images\\图_MNIST损失函数.png")
    plt.show()

    return None


if __name__=='__main__':
    # 选择给定标签的MNIST训练图像
    X_train, y_train, X_test, y_test = load_data()
    X_selected = select_data_with_label(X_train, y_train, label_set=[6])
    # 测试
    if (0):
        # 实验和分析VAE
        experiment_VAE(train_inputs=X_selected)
    else:
        # 调用现成的 class_VAE
        vae = VAE(train_inputs=X_selected,
                  noise_dim=2,
                  encoder_design=encoder_design,
                  decoder_model=decoder_model
                 )
        
        batch_size = 128
        num_epochs = 1
        noise_dim = 2        
        # 训练
        encoder, decoder, loss_seq = vae.train(optimizer=Adam(learning_rate=0.01),
                                               BATCH_SIZE=batch_size, 
                                               num_epochs=num_epochs, 
                                               num_iteration=int(len(X_selected)/batch_size)
                                              )
        # 输出生成图片样例
        generated_images = vae.generate(decoder, num_samples=100)

        image = combine_images(generated_images)
        image = image*127.5+127.5    
        Image.fromarray(image.astype(np.uint8)).save(workpath+"\\images\\图_MNIST生成图片_VAE.png")    

        # 输出训练损失函数，供调参用
        plt.plot(loss_seq, color='blue', label='vae_loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend(['vae_loss'], loc='best')
        plt.savefig(workpath+"\\images\\图_MNIST损失函数.png")
        plt.show()
        