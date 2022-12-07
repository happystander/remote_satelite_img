from my_generator import *
from model_set import *
from image_processing import *
from keras import callbacks 

'''
数据集相关参数
'''
#  训练数据图像路径
train_image_path = "./temp_datasets/img_train"
#  训练数据标签路径
train_label_path = "./temp_datasets/label_train"
#  验证数据图像路径
validation_image_path = "./temp_datasets/img_test"
#  验证数据标签路径
validation_label_path = "./temp_datasets/label_test"

'''
模型相关参数
'''  
#  批大小
batch_size = 2
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (256, 256, 3)
#  训练模型的迭代总轮数
epochs = 20
#  初始学习率
learning_rate = 1e-3
#  预训练模型地址
premodel_path = None
#  训练模型保存地址
model_path = "./unet_model.hdf5"

#  训练数据数目
train_num = len(os.listdir(train_image_path))
#  验证数据数目
validation_num = len(os.listdir(validation_image_path))
#  训练集每个epoch有多少个batch_size
steps_per_epoch = train_num / batch_size
#  验证集每个epoch有多少个batch_size
validation_steps = validation_num / batch_size
#  标签的颜色字典,用于onehot编码
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                train_image_path, 
                                train_label_path,
                                classNum ,
                                colorDict_GRAY,
                                input_size)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,     
                                validation_image_path,
                                validation_label_path,
                                classNum,
                                colorDict_GRAY,
                                input_size)
#  定义模型
model = unet(pretrained_weights = premodel_path, 
            input_size = input_size, 
            classNum = classNum, 
            learning_rate = learning_rate)
#  打印模型结构
model.summary()
#  回调函数
model_checkpoint = callbacks.ModelCheckpoint(model_path,
                                monitor = 'loss',
                                verbose = 1,# 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                save_best_only = True)

#  获取当前时间

#  模型训练
history = model.fit_generator(train_Generator,
                    steps_per_epoch = steps_per_epoch,
                    epochs = epochs,
                    callbacks = [model_checkpoint],
                    validation_data = validation_data,
                    validation_steps = validation_steps)
model.save(model_path)