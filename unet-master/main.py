from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,    # 旋转
                    width_shift_range=0.05,                 # 左右平移
                    height_shift_range=0.05,                # 上下平移
                    shear_range=0.05,                           # 
                    zoom_range=0.05,                            # 随机放大缩小
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir = 'djxc')

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

# testGene = testGenerator("data/membrane/test")

# results = model.predict_generator(testGene, 30, verbose=1)
# saveResult("data/membrane/test", results)