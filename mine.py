'''
This is an abstract class for all data generator
Do NOT use this class!
'''
import threading
# import astra
import torch
import numpy as np

class Mine(threading.Thread):
    def __init__(self,args):
        threading.Thread.__init__(self)
        self.batch_size = args["batch_size"]
        self.latent_code_len = args["latent_code_len"]
        self.training_torch_device = args["data_torch_device"]
        if "data_queue" in args:
            self.data_queue = args["data_queue"]

    def get_batch_data(self, *args, **kwargs):
        '''
        return training data
        '''
        print("Implement me!")
        return None
    
    def run(self):
        print("Starting Mine...")
        # volume_data,sinogram_3d,sinogram_2d,sinogram_3d_img,rendered_results = self.get_batch_data()
        while True:
            # print("generating data....")
            sinogram_3d_img = self.get_batch_data()

            init_z = np.random.normal(0.0,1.0,(self.batch_size,self.latent_code_len)).astype(np.float32)
            init_z = torch.from_numpy(init_z).to(self.training_torch_device)

            # print("done.")
            data_map = {
                "init_z":init_z.to(self.training_torch_device),
                "sinogram_3d_img":sinogram_3d_img.to(self.training_torch_device)
            }
            # if 'primitive' in self.mine_name:
            #     print("[{}]queue num:{}".format(self.mine_name,self.data_queue.qsize()))
            self.data_queue.put(data_map)