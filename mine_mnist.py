import cv2
from sklearn.utils import shuffle
import torch
import numpy as np
import os
import astra

from mine import Mine
from torchvision import datasets, transforms

class Mine_Minst(Mine):
    def __init__(self,args):
        super().__init__(args)

        transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.dataset_mnist = datasets.MNIST('data/', train=True, download=True,transform=transform)
        self.data_loader = torch.utils.data.DataLoader(self.dataset_mnist,batch_size=self.batch_size,drop_last =True,shuffle=True)
        self.data_loader_handle = iter(self.data_loader)

    def get_batch_data(self, *args, **kwargs):
        '''
        return:
            input_sinograms: (batch_size, projection_num, img_height,img_width)
            reconstruct_slice: (batch_size, *volume_shape)
        '''
        try:
            cur_batch = next(self.data_loader_handle)[0]
        except Exception as e:
            self.data_loader_handle = iter(self.data_loader)
            cur_batch = next(self.data_loader_handle)[0]

        return cur_batch
        
        

if __name__ == "__main__":
    torch.random.manual_seed(9929)
    np.random.seed(9929)

    import open3d as o3d
    import time
    import torchvision
    import threading
    import queue


    log_path = "garbages/"
    os.makedirs(log_path,exist_ok=True)

    data_gpu_id = 0
    data_torch_device = torch.device("cuda:{}".format(data_gpu_id))
    data_queue = queue.Queue(10)

    mine_args = {
        "batch_size": 2,
        "data_torch_device": data_torch_device,
        "data_gpu_id": data_gpu_id,
        "data_queue":data_queue
    }

    test_primitive_mine = Mine_Minst(mine_args)
    test_primitive_mine.daemon = True
    test_primitive_mine.start()
    
    
    for which_batch in range(100):
        print("{}".format(which_batch))
        tmp_folder = log_path#"garbages/{}/".format(which_batch)
        # os.makedirs(tmp_folder,exist_ok=True)
     
        a_batch_data = data_queue.get()
        sinogram_3d_img = a_batch_data["sinogram_3d_img"]
        
        for which_sample in range(sinogram_3d_img.shape[0]):
            tmp_sample = sinogram_3d_img[which_sample].cpu().numpy().reshape((28,28,1))
            cv2.imwrite(tmp_folder+"{}.png".format(which_sample),tmp_sample*255.0)
        time.sleep(1)