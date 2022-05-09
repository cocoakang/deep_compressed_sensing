import torch
import torch.optim as optim
import numpy as np
import cv2
import argparse
from torch.utils.tensorboard import SummaryWriter 
import os
import queue
from datetime import datetime
import time
import shutil
import sys
import threading
from mine_mnist import Mine_Minst
from TRAINING_NET import TRAINING_NET


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--training_gpu_id",type=int,default=0)
    parser.add_argument("--data_gpu_id",type=int,default=0)
    parser.add_argument("--log_path",default="./training_log/")
    parser.add_argument("--task_name",default="auto_encoder")
    parser.add_argument("--sub_task_name",default="nonlinear_nonlinear_novgg")
    parser.add_argument("--need_clean_log_folder",action="store_true")
    parser.add_argument("--pattern_num",type=int,default=25)
    parser.add_argument("--latent_code_len",type=int,default=100)
    parser.add_argument("--max_inner_setp",type=int,default=3)
    parser.add_argument("--max_train_itr",type=int,default=5000000)
    parser.add_argument("--z_step_size",type=float,default=0.01)
    
    args = parser.parse_args()
    args.sub_task_name = args.sub_task_name+"_{}".format(args.pattern_num)

    ##----preparation
    np.random.seed(2333)
    torch.random.manual_seed(2333)
    trainig_torch_device = torch.device("cuda:{}".format(args.training_gpu_id))
    data_torch_device = torch.device("cuda:{}".format(args.data_gpu_id))

    ##-----build mine
    data_primitive_queue = queue.Queue(10)
    mine_args = {
        "batch_size": args.batch_size,
        "data_torch_device": data_torch_device,
        "data_gpu_id": args.data_gpu_id,
        "data_queue":data_primitive_queue,
        "latent_code_len":args.latent_code_len,
    }

    test_primitive_mine = Mine_Minst(mine_args)
    test_primitive_mine.daemon = True
    test_primitive_mine.start()

    ##-----build net
    loss_lambda = {
        "sinogram":1.0,
    }
    training_args = {
        "pattern_num" : args.pattern_num,
        "training_torch_device":trainig_torch_device,
        "loss_lambda" : loss_lambda,
        "src_num":784,
        "det_num":784,
        "latent_code_len":args.latent_code_len,
        "project_method":"norm",
        "max_inner_setp":args.max_inner_setp,
        "z_step_size":args.z_step_size
    }
    trainer_net = TRAINING_NET(training_args)
    trainer_net.to(trainig_torch_device)

    ##-----build optimizer
    lr = 1e-4
    optimizer = optim.Adam(trainer_net.parameters(), lr=lr)

    ##----build logger
    os.makedirs(args.log_path,exist_ok=True)
    log_folder_task = args.log_path+"{}/{}/".format(args.task_name,args.sub_task_name)
    log_folder_task = log_folder_task.strip("/")+datetime.now().strftime('_%m_%d_%H_%M/')
    if os.path.exists(log_folder_task) and args.need_clean_log_folder:
        shutil.rmtree(log_folder_task)
    os.makedirs(log_folder_task,exist_ok=True)

    log_folder_this_time = log_folder_task+"summary"+'_PID{}'.format(os.getpid())+'/'
    tb_writer = SummaryWriter(log_dir=log_folder_this_time)
    trainnig_file_back_folder = log_folder_this_time+"training_files/"
    os.makedirs(trainnig_file_back_folder,exist_ok=True)
    mv_order_name = "cp" if sys.platform.startswith('linux') else "xcopy"
    trainnig_file_back_folder_tmp = trainnig_file_back_folder if sys.platform.startswith('linux') else trainnig_file_back_folder.replace("/","\\")
    print("{} *.py ".format(mv_order_name)+trainnig_file_back_folder_tmp)
    os.system("{} *.py ".format(mv_order_name)+trainnig_file_back_folder_tmp)
    os.system("{} *.sh ".format(mv_order_name)+trainnig_file_back_folder_tmp)
    os.system("{} *.txt ".format(mv_order_name)+trainnig_file_back_folder_tmp)
    model_backup_folder = log_folder_this_time+"models/"
    os.makedirs(model_backup_folder,exist_ok=True)

    ##----------------------------------------------------training
    def logging_func(data_pkg,result_pkg,tb_writer,obj_name):
        sinogram_3d_img_nn,sinogram_3d_img = result_pkg["sinogram_3d_img_nn"],result_pkg["sinogram_3d_img"]

        batch_size,channel_num,img_height,img_width = sinogram_3d_img_nn.shape

        sinogram_3d_img_cat = torch.stack([sinogram_3d_img,sinogram_3d_img_nn],dim=3)
        sinogram_3d_img_cat = torch.reshape(sinogram_3d_img_cat,(batch_size,channel_num,img_height,img_width*2))
        tb_writer.add_images("recon_{}".format(obj_name),sinogram_3d_img_cat, global_step=global_step, dataformats='NCHW')

        sinogram_3d_img_cat_np = sinogram_3d_img_cat.detach().cpu().numpy()
        for which_batch in range(batch_size):
            tmp_img = sinogram_3d_img_cat_np[which_batch]
            tmp_img = np.transpose(tmp_img,(1,2,0))
            cv2.imwrite("garbages/{}.png".format(which_batch),tmp_img*255.0)

    for global_step in range(args.max_train_itr):
        if global_step % 100 == 0:
            print(global_step)
            trainer_net.eval()
            data_pkg = data_primitive_queue.get()
            loss_pkg,result_pkg = trainer_net(data_pkg,return_recon=True)
            logging_func(data_pkg,result_pkg,tb_writer,"mnist")

        ##-----train
        trainer_net.train()
        optimizer.zero_grad()
        # print("getting data...")
        data_pkg = data_primitive_queue.get()
        # print("forwarding...")
        loss_pkg = trainer_net(data_pkg)
        
        loss_pkg["total_loss"].backward()
        optimizer.step()
        
        #logging
        if loss_pkg["total_loss"] is not None:
            tb_writer.add_scalar('loss_total_train', loss_pkg["total_loss"].item(), global_step)
