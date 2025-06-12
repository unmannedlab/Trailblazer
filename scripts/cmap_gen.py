import torch
from torchvision import transforms as T
import model_overhead as model
#from camel_path import *
import numpy as np
from rospkg import RosPack
import os
import sys
import matplotlib.pyplot as plt

rospack = RosPack()
package_path = rospack.get_path('camel')


torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_map(array):
    current_cmap = plt.cm.Blues
    current_cmap.set_bad(color='red')
    fig, ax = plt.subplots(figsize=(40,28)) #costmap
    ax.matshow(array,cmap=plt.cm.Blues, vmin=np.min(array), vmax = np.max(array)) 
    plt.show()

def costmap_gen(data,shape):
    models = model.MSFCN()
    models.load_state_dict(torch.load(os.path.join(package_path,'scripts/weights/camel/model.pth'))['model_state_dict'])
    models.to(device)
    models.eval()
    trans = T.Compose([T.ToTensor()])
    train_data = trans(data).unsqueeze(0).cuda()
    out1 = models(train_data)
    cmap = np.reshape(np.round(np.ravel(np.asarray(out1.tolist())),4),shape)
    cmap = np.round((cmap-cmap.min())/(cmap.max()-cmap.min()),3)
    plot_map(cmap)
    return cmap

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <npy_file>")
        sys.exit(1)
    npy_file = sys.argv[1]
    data = np.load(npy_file,allow_pickle=True)
    shape = (data.shape[0],data.shape[1])
    cmap = costmap_gen(data,shape)
    #np.save('./assets/cavasos_costmap_final.npy',cmap,allow_pickle=True)
    plot_map(cmap)
    print('Costmap Generated')