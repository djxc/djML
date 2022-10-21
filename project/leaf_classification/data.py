import os, shutil
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

rootPath = r"E:\Data\MLData\classify\classify-leaves"

categories = ['maclura_pomifera', 'ulmus_rubra', 'broussonettia_papyrifera', 'prunus_virginiana', 'acer_rubrum', 'cryptomeria_japonica', 'staphylea_trifolia', 'asimina_triloba', 'diospyros_virginiana', 'tilia_cordata', 'ulmus_pumila', 'quercus_muehlenbergii', 'juglans_cinerea', 'cercis_canadensis', 'ptelea_trifoliata', 'acer_palmatum', 'catalpa_speciosa', 'abies_concolor', 'eucommia_ulmoides', 'quercus_montana', 'koelreuteria_paniculata', 'liriodendron_tulipifera', 'styrax_japonica', 'malus_pumila', 'prunus_sargentii', 'cornus_mas', 'magnolia_virginiana', 'ostrya_virginiana', 'magnolia_acuminata', 'ilex_opaca', 'acer_negundo', 'fraxinus_nigra', 'pyrus_calleryana', 'picea_abies', 'chionanthus_virginicus', 'carpinus_caroliniana', 'zelkova_serrata', 'aesculus_pavi', 'taxodium_distichum', 'carya_tomentosa', 'picea_pungens', 'carya_glabra', 'quercus_macrocarpa', 'carya_cordiformis', 'catalpa_bignonioides', 'tsuga_canadensis', 'populus_tremuloides', 'magnolia_denudata', 'crataegus_viridis', 'populus_deltoides', 'ulmus_americana', 'pinus_bungeana', 'cornus_florida', 'pinus_densiflora', 'morus_alba', 'quercus_velutina', 'pinus_parviflora', 'salix_caroliniana', 'platanus_occidentalis', 'acer_saccharum', 'pinus_flexilis', 'gleditsia_triacanthos', 'quercus_alba', 'prunus_subhirtella', 'pseudolarix_amabilis', 'stewartia_pseudocamellia', 'quercus_stellata', 'pinus_rigida', 'salix_nigra', 'quercus_acutissima', 'pinus_virginiana', 'chamaecyparis_pisifera', 'quercus_michauxii', 'prunus_pensylvanica', 'amelanchier_canadensis', 'liquidambar_styraciflua', 'pinus_cembra', 'malus_hupehensis', 'castanea_dentata', 'magnolia_stellata', 'chionanthus_retusus', 'carya_ovata', 'quercus_marilandica', 'tilia_americana', 'cedrus_atlantica', 'ulmus_parvifolia', 'nyssa_sylvatica', 'quercus_virginiana', 'acer_saccharinum', 'magnolia_macrophylla', 'crataegus_pruinosa', 'pinus_nigra', 'abies_nordmanniana', 'pinus_taeda', 'ficus_carica', 'pinus_peucea', 'populus_grandidentata', 'acer_platanoides', 'pinus_resinosa', 'salix_matsudana', 'pinus_sylvestris', 'albizia_julibrissin', 'salix_babylonica', 'pinus_echinata', 'magnolia_tripetala', 'larix_decidua', 'pinus_strobus', 'aesculus_glabra', 'ginkgo_biloba', 'quercus_cerris', 'metasequoia_glyptostroboides', 'fagus_grandifolia', 'quercus_nigra', 'juglans_nigra', 'pinus_koraiensis', 'oxydendrum_arboreum', 'morus_rubra', 'crataegus_phaenopyrum', 'pinus_wallichiana', 'tilia_europaea', 'betula_jacqemontii', 'chamaecyparis_thyoides', 'acer_ginnala', 'acer_campestre', 'pinus_pungens', 'malus_floribunda', 'picea_orientalis', 'amelanchier_laevis', 'celtis_tenuifolia', 'gymnocladus_dioicus', 'quercus_bicolor', 'malus_coronaria', 'cercidiphyllum_japonicum', 'cedrus_libani', 'betula_nigra', 'acer_pensylvanicum', 'platanus_acerifolia', 'robinia_pseudo-acacia', 'ulmus_glabra', 'crataegus_laevigata', 'quercus_coccinea', 'prunus_serotina', 'tilia_tomentosa', 'quercus_imbricaria', 'cladrastis_lutea', 'fraxinus_pennsylvanica', 'phellodendron_amurense', 'betula_lenta', 'quercus_robur', 'aesculus_flava', 'paulownia_tomentosa', 'amelanchier_arborea', 'quercus_shumardii', 'magnolia_grandiflora', 'cornus_kousa', 'betula_alleghaniensis', 'carpinus_betulus', 'aesculus_hippocastamon', 'malus_baccata', 'acer_pseudoplatanus', 'betula_populifolia', 'prunus_yedoensis', 'halesia_tetraptera', 'quercus_palustris', 'evodia_daniellii', 'ulmus_procera', 'prunus_serrulata', 'quercus_phellos', 'cedrus_deodara', 'celtis_occidentalis', 'sassafras_albidum', 'acer_griseum', 'ailanthus_altissima', 'pinus_thunbergii', 'crataegus_crus-galli', 'juniperus_virginiana']


def statisticsCategories():
    '''统计不同类的分布情况'''
    categoriesStatis = {}
    categories = []
    with open(rootPath + r"\train.csv") as trainFile:
        lines = trainFile.readlines()
        i = 0
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            line_split = line.replace("\n", "").split(",")
            category = line_split[1]
            print(category)
            if category not in categoriesStatis:
                categoriesStatis[category] = 1
                categories.append(category)
            else:
                categoriesStatis[category] = categoriesStatis[category] + 1
            i = i + 1    
    print(categoriesStatis)
    print(categories)

def createCategoriesFloders():
    '''创建文件夹'''
    root_verify = rootPath + r"\verifyData"
    root_train = rootPath + r"\leafClasses"
    categories = ['maclura_pomifera', 'ulmus_rubra', 'broussonettia_papyrifera', 'prunus_virginiana', 'acer_rubrum', 'cryptomeria_japonica', 'staphylea_trifolia', 'asimina_triloba', 'diospyros_virginiana', 'tilia_cordata', 'ulmus_pumila', 'quercus_muehlenbergii', 'juglans_cinerea', 'cercis_canadensis', 'ptelea_trifoliata', 'acer_palmatum', 'catalpa_speciosa', 'abies_concolor', 'eucommia_ulmoides', 'quercus_montana', 'koelreuteria_paniculata', 'liriodendron_tulipifera', 'styrax_japonica', 'malus_pumila', 'prunus_sargentii', 'cornus_mas', 'magnolia_virginiana', 'ostrya_virginiana', 'magnolia_acuminata', 'ilex_opaca', 'acer_negundo', 'fraxinus_nigra', 'pyrus_calleryana', 'picea_abies', 'chionanthus_virginicus', 'carpinus_caroliniana', 'zelkova_serrata', 'aesculus_pavi', 'taxodium_distichum', 'carya_tomentosa', 'picea_pungens', 'carya_glabra', 'quercus_macrocarpa', 'carya_cordiformis', 'catalpa_bignonioides', 'tsuga_canadensis', 'populus_tremuloides', 'magnolia_denudata', 'crataegus_viridis', 'populus_deltoides', 'ulmus_americana', 'pinus_bungeana', 'cornus_florida', 'pinus_densiflora', 'morus_alba', 'quercus_velutina', 'pinus_parviflora', 'salix_caroliniana', 'platanus_occidentalis', 'acer_saccharum', 'pinus_flexilis', 'gleditsia_triacanthos', 'quercus_alba', 'prunus_subhirtella', 'pseudolarix_amabilis', 'stewartia_pseudocamellia', 'quercus_stellata', 'pinus_rigida', 'salix_nigra', 'quercus_acutissima', 'pinus_virginiana', 'chamaecyparis_pisifera', 'quercus_michauxii', 'prunus_pensylvanica', 'amelanchier_canadensis', 'liquidambar_styraciflua', 'pinus_cembra', 'malus_hupehensis', 'castanea_dentata', 'magnolia_stellata', 'chionanthus_retusus', 'carya_ovata', 'quercus_marilandica', 'tilia_americana', 'cedrus_atlantica', 'ulmus_parvifolia', 'nyssa_sylvatica', 'quercus_virginiana', 'acer_saccharinum', 'magnolia_macrophylla', 'crataegus_pruinosa', 'pinus_nigra', 'abies_nordmanniana', 'pinus_taeda', 'ficus_carica', 'pinus_peucea', 'populus_grandidentata', 'acer_platanoides', 'pinus_resinosa', 'salix_matsudana', 'pinus_sylvestris', 'albizia_julibrissin', 'salix_babylonica', 'pinus_echinata', 'magnolia_tripetala', 'larix_decidua', 'pinus_strobus', 'aesculus_glabra', 'ginkgo_biloba', 'quercus_cerris', 'metasequoia_glyptostroboides', 'fagus_grandifolia', 'quercus_nigra', 'juglans_nigra', 'pinus_koraiensis', 'oxydendrum_arboreum', 'morus_rubra', 'crataegus_phaenopyrum', 'pinus_wallichiana', 'tilia_europaea', 'betula_jacqemontii', 'chamaecyparis_thyoides', 'acer_ginnala', 'acer_campestre', 'pinus_pungens', 'malus_floribunda', 'picea_orientalis', 'amelanchier_laevis', 'celtis_tenuifolia', 'gymnocladus_dioicus', 'quercus_bicolor', 'malus_coronaria', 'cercidiphyllum_japonicum', 'cedrus_libani', 'betula_nigra', 'acer_pensylvanicum', 'platanus_acerifolia', 'robinia_pseudo-acacia', 'ulmus_glabra', 'crataegus_laevigata', 'quercus_coccinea', 'prunus_serotina', 'tilia_tomentosa', 'quercus_imbricaria', 'cladrastis_lutea', 'fraxinus_pennsylvanica', 'phellodendron_amurense', 'betula_lenta', 'quercus_robur', 'aesculus_flava', 'paulownia_tomentosa', 'amelanchier_arborea', 'quercus_shumardii', 'magnolia_grandiflora', 'cornus_kousa', 'betula_alleghaniensis', 'carpinus_betulus', 'aesculus_hippocastamon', 'malus_baccata', 'acer_pseudoplatanus', 'betula_populifolia', 'prunus_yedoensis', 'halesia_tetraptera', 'quercus_palustris', 'evodia_daniellii', 'ulmus_procera', 'prunus_serrulata', 'quercus_phellos', 'cedrus_deodara', 'celtis_occidentalis', 'sassafras_albidum', 'acer_griseum', 'ailanthus_altissima', 'pinus_thunbergii', 'crataegus_crus-galli', 'juniperus_virginiana']
    # 创建文件夹
    for category in categories:
        os.mkdir(os.path.join(root_verify, category))
        os.mkdir(os.path.join(root_train, category))


def moveDataToCategoriesFloder():
    '''将同类的数据放在一个文件夹下'''
    dataPath = rootPath
    root_path = rootPath + "\leafClasses"
    
    with open(rootPath + r"\train.csv") as trainFile:
        lines = trainFile.readlines()
        i = 0
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            line_split = line.replace("\n", "").split(",")
            imageName = line_split[0].replace("/", "\\")
            category = line_split[1]
            # print(category, os.path.join(dataPath, imageName), os.path.join(root_path, category))
            shutil.move(os.path.join(dataPath, imageName), os.path.join(root_path, category))


def splitTrainAndVerify():
    '''将数据分为训练集以及验证集'''
    root_path = rootPath + "\leafClasses"
    verifyPath = rootPath + r"\verifyData"
    for category in categories:
        currentImageFloder = os.path.join(root_path, category)
        files = os.listdir(currentImageFloder)
        np.random.shuffle(files)
        imageNum = len(files)
        limit_value = imageNum * 0.3
        for i, image in enumerate(files):
            if i <= limit_value:
                shutil.move(os.path.join(currentImageFloder, image), os.path.join(verifyPath, category))
        print(files)


def readDataToCSV(imageRoot):
    '''读取数据，生成csv文件'''
    imageFiles = []
    dirPaths = os.listdir(imageRoot)
    for dir in dirPaths:
        currentImageFloder = os.path.join(imageRoot, dir)
        print(currentImageFloder)
        images = os.listdir(currentImageFloder)
        print(images)
        for image in images:
            imagePath = os.path.join(currentImageFloder, image)
            imageFiles.append(imagePath + ", " + dir + "\n")
    
    with open(rootPath + r"\verify_data.csv", "w") as trainData:
        trainData.writelines(imageFiles)
    

class LeafDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集。"""
    def __init__(self, imageFile, mode="train"):
        self.imageDatas = []
        self.mode = mode
        self.ones = torch.sparse.torch.eye(len(categories))
        with open(imageFile) as image_file:
            self.imageDatas = image_file.readlines()

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        print("read {} {} examples".format(len(self.imageDatas), mode))        

    def __getitem__(self, idx):
        if self.mode == "test":
            imagePath = self.imageDatas[idx].replace("\n", "")
            image = Image.open(os.path.join(rootPath, imagePath))
            return self.transform(image), imagePath
        else:
            imagePath, label = self.imageDatas[idx].replace("\n", "").split(", ")        
            label = self.ones.index_select(0, torch.tensor(categories.index(label))) # categories.index(label)
            image = Image.open(os.path.join(rootPath, imagePath))
            image = self.transform(image)
            return (image, label)

    def __len__(self):
        return len(self.imageDatas)


def load_leaf_data(batch_size):
    ''' 加载ITCVD数据集
    '''
    num_workers = 4
    print("load train data, batch_size", batch_size)
    train_iter = torch.utils.data.DataLoader(
        LeafDataset(rootPath + r"\train_data.csv", "train"), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)

    test_iter = torch.utils.data.DataLoader(
        LeafDataset(rootPath + r"\verify_data.csv", "valid"), batch_size, drop_last=True,
        num_workers=num_workers)
    return train_iter, test_iter

def load_test_leaf_data(batch_size, num_workers = 4):
    """加载测试数据集"""
    print("load test data, batch_size", batch_size)
    test_iter = torch.utils.data.DataLoader(
        LeafDataset(rootPath + r"\test.csv", "test"), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)
   
    return test_iter

            
# 图像增强
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """ Mixup 数据增强 -> 随机叠加两张图像 """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # β分布
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam     


def color(x, y, alpha=1.0, use_cuda=True):
    '''图像颜色、亮度、色调等'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    new = transforms.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    new_x = new(x)
    y_a, y_b = y, y[index]
    return new_x, y_a, y_b, lam


def rotate_data(x, y, alpha=1.0, use_cuda=True):
    '''图像按一定角度旋转'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    new = transforms.RandomRotation(degrees=(90, 180))
    new_x = new(x)
    y_a, y_b = y, y[index]
    return new_x, y_a, y_b, lam

def rand_bbox(size, lam):
    '''随机裁剪'''
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    # cx = np.random.randint(W)
    # cy = np.random.randint(H)

    cx = np.int(W / 2)
    cy = np.int(H / 2)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    """ Cutmix 数据增强 -> 随机对主图像进行裁剪, 加上噪点图像
    W: 添加裁剪图像宽
    H: 添加裁剪图像高
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


if __name__ == "__main__":
    # statisticsCategories()
    # createCategoriesFloders()
    # moveDataToCategoriesFloder()
    # splitTrainAndVerify()
    readDataToCSV(rootPath + r"\verifyData")