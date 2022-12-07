from my_generator import *
from model_set import *
from image_processing import *
from keras.models import load_model

#  测试数据生成器
#  test_iamge_path 测试数据路径
#  resize_shape resize大小
def testGenerator(test_iamge_path, resize_shape = None):
    imageList = os.listdir(test_iamge_path)
    for i in range(len(imageList)):
        img = readTif(test_iamge_path + "\\" + imageList[i])
        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        #  归一化
        img = img / 255.0
        if(resize_shape != None):
            #  改变图像尺寸至特定尺寸
            img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
        #  将测试图片扩展一个维度,与训练时的输入[batch_size,img.shape]保持一致
        img = np.reshape(img, (1, ) + img.shape)
        yield img
#  保存结果
#  test_iamge_path 测试数据图像路径
#  test_predict_path 测试数据图像预测结果路径
#  model_predict 模型的预测结果
#  color_dict 颜色词典
def saveResult(test_image_path, test_predict_path, model_predict, color_dict, output_size):
    imageList = os.listdir(test_image_path)
    for i, img in enumerate(model_predict):
        channel_max = np.argmax(img, axis = -1)
        img_out = np.uint8(color_dict[channel_max.astype(np.uint8)])
        #  修改差值方式为最邻近差值
        img_out = cv2.resize(img_out, (output_size[0], output_size[1]), interpolation = cv2.INTER_NEAREST)
        #  保存为无损压缩png
        cv2.imwrite(test_predict_path + "\\" + imageList[i][:-4] + ".png", img_out)


#  训练模型保存地址
model_path = "./unet_model.hdf5"
#  测试数据路径
test_iamge_path = "./temp_datasets/img_test"
#  结果保存路径
save_path = "./predict"
#  测试数据数目
test_num = len(os.listdir(test_iamge_path))
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (256, 256, 3)
#  生成图像大小
output_size = (256, 256)
#  训练数据标签路径
train_label_path = "./temp_datasets/label_train"
#  标签的颜色字典
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

model = load_model(model_path)

testGene = testGenerator(test_iamge_path, input_size)

#  预测值的Numpy数组
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)

#  保存结果
saveResult(test_iamge_path, save_path, results, colorDict_GRAY, output_size)