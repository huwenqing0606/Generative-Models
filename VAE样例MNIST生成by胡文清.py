"""
用变分自动编码器生成MNIST手写数字样本的样例程序
运行环境: Keras and Tensorflow 1.14
参考文献: Kingma, D.P. and Welling, M., Auto-Encoding Variational Bayes, arXiv:1312.6114, Dec. 2013.
作者：胡文清（明略科技）
"""

from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

# 工作路径
workpath = "D:\\Temporary Files\\2021_08-12_明略科技数据科学\\1_ID缺失监测方法论\\20210919基于生成模型的IDFA缺失监测\\15_变分自编码机\\code"


"""
给欣雨的提示：将此代码运用于人口属性特征概率向量的生成，只需要
    (1) 修改读入的数据 X 为历史人口属性向量，y 为触达特征的分类标签
    (2) 修改编码器设计层和解码器的神经网络结构
    (3) 测试不同的训练超参数
    (4*) 直接调用封装好的 VAE 类 class_VAE 中的 VAE 
    (*鉴于VAE损失函数对于网络层的复杂依赖，keras很难灵活处理，最直接的办法是封装tensorflow版本的代码，目前尚在开发中)
"""


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


# 重参数化层函数，将[mu, sigma]层转换为 Z = mu + eps*sigma 其中 eps ~ N(0, I)
# 此处仅用于代码分析与测试
def reparametrization_fn(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps


# 建立并训练 VAE, 返回网络结构和训练好的超参数
# 此处仅用于代码分析与测试
def VAE_train(train_inputs, latent_dim, epochs, batch_size, validation_split):
    """编码器设计层, 输入数据，输出任意维数的张量"""
    # 输入层
    i = Input(shape=(28,28,1), name='encoder_input')
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
    # 返回编码器设计层的输出张量 x
    x = BatchNormalization(name='encoder_design_output')(x)
    """
    编码器[mu, sigma]和重参数化层
    在设计层基础上加载高斯隐变量 Z 的均值mu和方差sigma层[mu, sigma]，
    然后加载重参数化层将[mu, sigma]层转换为 Z = mu + eps*sigma 其中 eps ~ N(0, I)
    """
    # 添加一个全连接层，输出为隐变量 Z 服从的高斯分布的均值
    mu = Dense(latent_dim, name='latent_mu')(x)
    # 添加一个全连接层，输出为隐变量 Z 服从的高斯分布的方差
    sigma = Dense(latent_dim, name='latent_sigma')(x)
    # 添加重参数化层，将[mu, sigma]层转换为 Z = mu + eps*sigma 其中 eps ~ N(0, I)
    Z = Lambda(reparametrization_fn, output_shape=(latent_dim, ), name='Z')([mu, sigma])
    """构建并审阅编码器"""
    encoder = Model(i, [mu, sigma, Z], name='encoder')
    encoder.summary()    
    """解码器，输入高斯隐变量Z，输出生成的数据"""
    # 获取编码器卷积层的尺寸参数用于解码
    conv_shape = K.int_shape(cx)
    # 解码器输入层
    d_i   = Input(shape=(latent_dim, ), name='decoder_input')
    x     = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
    x     = BatchNormalization()(x)
    x     = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    cx    = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    cx    = BatchNormalization()(cx)
    cx    = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
    cx    = BatchNormalization()(cx)
    o     = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
    """构建并审阅解码器"""
    decoder = Model(d_i, o, name='decoder')
    decoder.summary()
    """构建并审阅VAE"""
    vae_outputs = decoder(encoder(i)[2])
    vae = Model(i, vae_outputs, name='vae')
    vae.summary()
    """编译VAE"""
    vae.compile(optimizer='adam', loss=vae_loss(mu, sigma))
    """训练VAE"""
    vae.fit(train_inputs, train_inputs, epochs = num_epochs, batch_size = batch_size, validation_split = validation_split)
    """返回训练好的VAE编码器和解码器"""
    encoder.save_weights('encoder_weights', True)
    decoder.save_weights('decoder_weights', True)
    return encoder, decoder


# VAE损失函数，因为涉及到编码器输出层，所以需要用二层函数嵌套技巧
# 此处仅用于代码分析与测试
def vae_loss(mu, sigma):
    """损失函数必须在整个计算图内部定义"""
    def kl_reconstruction_loss(true, pred):
        # 重构损失
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * 28 * 28
        # KL 散度损失
        kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # 总损失=50% 重构损失 + 50% KL 散度损失
        return K.mean(reconstruction_loss + kl_loss)
    return kl_reconstruction_loss

        
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


# VAE 图片生成器
def vae_generate(decoder, latent_dim):
    decoder.load_weights('decoder_weights')
    num_samples = 100
    generated_images = []
    for index in range(num_samples):
        Z_sample = np.random.normal(0,1,size=latent_dim)
        x_decoded=decoder.predict(np.array([Z_sample]))
        digit=x_decoded[0].reshape(28,28,1)
        generated_images.append(digit)
    generated_images=np.array(generated_images)
    image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(workpath+"\\images\\图_MNIST生成图片_VAE.png")    
    return None


if __name__=='__main__':
    # 选择给定标签的MNIST训练图像
    X_train, y_train, X_test, y_test = load_data()
    X_selected = select_data_with_label(X_train, y_train, label_set=[6])
 
    # 训练VAE
    batch_size = 128
    num_epochs = 500
    validation_split = 0.2
    latent_dim = 2
    encoder, decoder=VAE_train(train_inputs=X_selected, latent_dim=latent_dim, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)

    # 输出生成图片样例
    vae_generate(decoder, latent_dim)