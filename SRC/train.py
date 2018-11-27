from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential

i=1
train_dir='C:\\Users\\ййй\\PycharmProjects\\untitled\\train'
val_dir='C:\\Users\\ййй\\PycharmProjects\\untitled\\val'
test_dir='C:\\Users\\ййй\\PycharmProjects\\untitled\\test'

model=load_model(sys.argv[1])

testgen = ImageDataGenerator(rescale=1. / 255)
now_generator = testgen.flow_from_directory(sys.argv[3], target_size=(150, 150), batch_size=10, class_mode='categorical')
a = int(1-(model.predict_generator(now_generator, 1//1)))
print(a)