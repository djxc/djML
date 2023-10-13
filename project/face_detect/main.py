import torch
import torch.utils.data.dataloader as dataloader
from dataset import FaceDataset

def load_face_data(data_root, batch_size):
    ''' 加载ITCVD数据集
    '''
    num_workers = 4
    print("load train data, batch_size", batch_size)
    train_iter = dataloader.DataLoader(
        FaceDataset(True, data_root), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)

    test_iter = dataloader.DataLoader(
        FaceDataset(False, "/2020/clothes_person_test/"), batch_size, drop_last=True,
        num_workers=num_workers)
    return train_iter, test_iter



    
if __name__ == "__main__":
    DATA_ROOT = r"D:\Data\MLData\facedata"
    batch_size = 4
    train_iter, _ = load_face_data(DATA_ROOT, batch_size)  
    num_epochs = 1 
    for i, (features, labels) in enumerate(train_iter):
        _, h, w = features[0].shape   