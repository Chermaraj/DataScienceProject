
import os
import sys
import random
import subprocess
import cv2
import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from PIL import Image


def learning_model(filePath,patient_name):
    random_seed= 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic=True
    batch_size = 64
    validation_split = .34
    shuffle_dataset = True



    def run_cmd(cmd, stderr=subprocess.STDOUT):
      out = None
      try:
        out = subprocess.check_output(
          [cmd], 
          shell=True,
          stderr=subprocess.STDOUT, 
          universal_newlines=True,
        )
      except subprocess.CalledProcessError as e:
        print(f'ERROR {e.returncode}: {cmd}\n\t{e.output}', flush=True, file=sys.stderr)
        raise e
      return out
  
  
    def clone_data(data_root):
      
      clone_uri = 'https://github.com/ieee8023/covid-chestxray-dataset.git'
      if os.path.exists(data_root):
          assert os.path.isdir(data_root), \
            f'{data_root} should be cloned from {clone_uri}'
      else:
          os.mkdir('data')
          mgpath=f'data/images',
          csvpath=f'data/metadata.csv',
          print(
            'Cloning the covid chestxray dataset. It may take a while\n...\n', 
            flush=True
            )
          run_cmd(f'git clone {clone_uri} {data_root}')
      
      
      
    #data_root = os.mkdir('data')
    #print(data_root)
    mgpath=f'data/images',
    #csvpath=f'{data_root}/metadata.csv',
    csvpath=f'data/metadata.csv',


    #clone_data('data')

    meta = pd.read_csv('data/metadata.csv')

    # we, are going to train our model against the only "PA" view of the lungs X-ray images.

    meta['view'].value_counts(dropna=False)

    for x in meta['filename']:
      if x.split('.')[-1]=='gz':
        meta.drop(meta.index[meta['filename']==x], 
                  inplace=True)
              
              
    meta = meta[(meta['finding']!='Chlamydophila')&(meta['finding']!='Legionella')&(meta['finding']!='Klebsiella')
                &(meta['finding']!='todo')&(meta['finding']!='Lymphocytic Interstitial Pneumonia')
                &(meta['finding']!='Bacterial')&(meta['finding']!='Multilobar Pneumonia')&(meta['finding']!='Round pneumonia')
                &(meta['finding']!='Allergic bronchopulmonary aspergillosis')&(meta['finding']!='Influenza')&(meta['finding']!='Swine-Origin Influenza A (H1N1) Viral Pneumonia')
                &(meta['finding']!='Accelerated Phase Usual Interstitial Pneumonia')&(meta['finding']!='Unusual Interstitial Pneumonia')
                &(meta['finding']!='Chronic eosinophilic pneumonia')&(meta['finding']!='Eosinophilic Pneumonia')&(meta['finding']!='Allergic bronchopulmonary aspergillosis ')
                &(meta['finding']!='Cryptogenic Organizing Pneumonia')&(meta['finding']!='Pneumonia')&(meta['finding']!='Tuberculosis')&(meta['finding']!='Lobar Pneumonia')
                &(meta['finding']!='Lipoid')&(meta['finding']!='Varicella')&(meta['finding']!='Mycoplasma Bacterial Pneumonia')&(meta['finding']!='Nocardia')
                &(meta['finding']!='Eosinophilic pneumonia')&(meta['finding']!='ARDS')
                &(meta['finding']!='MERS-CoV')]
    meta = meta[meta['view']=='PA']


    meta['finding'].value_counts(dropna=False)

    # Handling the unbalace in dataset
    # As you could clearly see that COVID affected x-ray images are ten times more in number than other classes. 
    # So in the next cell, we are going to bring down the number of images under COVID.
    # The excess images of COVID that has been removed from the dataset would be used 
    # to detect the region of the lungs that has been affected by COVID-19 through heatmap visualization

    X_train_val, X_test = train_test_split( meta[meta['finding']=='COVID-19'], test_size=0.85, random_state=random_seed)
    meta.drop(X_test.index, inplace=True)
    meta.reset_index(drop=True, inplace=True)


    meta['finding'].value_counts(dropna=False)


    dataset_size = len(meta)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]


    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)


    Labels = np.array(meta['finding']).reshape(len(meta['finding']),1)
    encode = OneHotEncoder()
    encode.fit(Labels)
    labels_enc = encode.transform(Labels).toarray()


    transform=transforms.Compose([
                                  transforms.ToPILImage(),
                                  transforms.RandomCrop(224),                              
                                  transforms.ToTensor(),                                                  
                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])
                                        
    class_names = [
        'COVID-19', 
        'Pneumocystis',
        'COVID-19, ARDS',
        'No Finding',       
        'Streptococcus',      
        'SARS ',     
    ]



    class ChestXrayDataSet(Dataset):
        def __init__(self,csvpath,mgpath,labels_enc,transform=None):
          self.meta_data = pd.read_csv(csvpath)
          self.root_dir = mgpath
          self.labels = self.meta_data['finding']
          self.transform = transform
          for x in self.meta_data['filename']:
            if x.split('.')[-1]=='gz':
              self.meta_data.drop(self.meta_data.index[self.meta_data['filename']==x],
                                  inplace=True)
    
          self.meta_data = self.meta_data[(self.meta_data['finding']!='Chlamydophila')
                                         &(self.meta_data['finding']!='Legionella')
                                         &(self.meta_data['finding']!='Klebsiella')
                                         &(self.meta_data['finding']!='todo')&(self.meta_data['finding']!='Lymphocytic Interstitial Pneumonia')
                                         &(self.meta_data['finding']!='Bacterial')&(self.meta_data['finding']!='Multilobar Pneumonia')&(self.meta_data['finding']!='Round pneumonia')
                                         &(self.meta_data['finding']!='Allergic bronchopulmonary aspergillosis')&(self.meta_data['finding']!='Influenza')&(self.meta_data['finding']!='Swine-Origin Influenza A (H1N1) Viral Pneumonia')
                                         &(self.meta_data['finding']!='Accelerated Phase Usual Interstitial Pneumonia')&(self.meta_data['finding']!='Unusual Interstitial Pneumonia')
                                         &(self.meta_data['finding']!='Chronic eosinophilic pneumonia')&(self.meta_data['finding']!='Eosinophilic Pneumonia')&(self.meta_data['finding']!='Allergic bronchopulmonary aspergillosis ')
                                         &(self.meta_data['finding']!='Cryptogenic Organizing Pneumonia')&(self.meta_data['finding']!='Pneumonia')&(self.meta_data['finding']!='Tuberculosis')&(self.meta_data['finding']!='Lobar Pneumonia')
                                         &(self.meta_data['finding']!='Lipoid')&(self.meta_data['finding']!='Varicella')&(self.meta_data['finding']!='Mycoplasma Bacterial Pneumonia')&(self.meta_data['finding']!='Nocardia')
                                         &(self.meta_data['finding']!='Eosinophilic pneumonia')&(self.meta_data['finding']!='ARDS')
                                         &(self.meta_data['finding']!='MERS-CoV')]
     
          self.meta_data = self.meta_data[self.meta_data['view']=='PA']
          self.meta_data.drop(X_test.index, inplace=True)
          self.meta_data.reset_index(drop=True, inplace=True)
    
        def __len__(self):
          return len(self.meta_data)

        def __getitem__(self, idx):
          if torch.is_tensor(idx):
            idx = idx.tolist()
          img_name = os.path.join(self.root_dir,
                                    self.meta_data.loc[idx,'filename'])
          image = Image.open(img_name).convert('RGB')
          image = np.array(image.resize((256,256)))
          image = image[:,:,0]
          image = np.uint8(((np.array(image)/255).reshape(256,256,1))*255*255)
          image = np.tile(image,3) 
          label = labels_enc[idx]
          if self.transform is not None:
            image = self.transform(image)
          return image, label, idx
      
      
    dataset = ChestXrayDataSet(csvpath[0],mgpath[0],labels_enc,transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
                                                
                                                
    def img_display(img):
        img = img*0.229+0.485   # unnormalize (inp = inp*std + mean)
        npimg = img.numpy()[0]
  
        return npimg
    

    # get some random training images
    dataiter = iter(train_loader)
    images, labels, id_ = dataiter.next()
    # Viewing data examples used for training
    fig, axis = plt.subplots(2, 4, figsize=(15, 10))
    for i, ax in enumerate(axis.flat):
        with torch.no_grad():
            image, label, _ = images[i], labels[i], id_[i]
            ax.imshow(img_display(image),cmap='gray') # add image
            ax.set(title = f"{meta['finding'][_.item()]}") # add label
        
        
        
    # construct model
    class DenseNet121(nn.Module):
      def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
              nn.Linear(num_ftrs, out_size),
              nn.Sigmoid()
          ).cuda()
      def forward(self, x):
        x = self.densenet121(x)
        return x
    
    
    cudnn.benchmark = True
    N_CLASSES = 6


    def compute_AUCs(gt, pred):
        AUROCs = []
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        for i in range(N_CLASSES):
          AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        return AUROCs
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)


    # initialize and load the model
    model = DenseNet121(N_CLASSES)
    model = model.cuda(device)


    optimizer = optim.Adam(model.parameters(),lr=0.0007,weight_decay=0.0000001)
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.cuda(device)


    # loading the model
    model.load_state_dict(torch.load('COVIDdetection/learningmodel/Covid_detection.pt'))


    ## GradCAM( Implementation + Visualization)

    class ChestXrayDataSet_plot(Dataset):
	    def __init__(self, input_X, transform=None):
		    self.data = input_X#np.uint8(test_X*255)
		    self.transform = transform
		    self.root_dir = mgpath[0]
		    self.transform = transform

	    def __getitem__(self, index):
	     if torch.is_tensor(index):
		     index = index.tolist()
	     img_name = os.path.join(self.root_dir,self.data.loc[index,'filename'])
	     image = Image.open(img_name).convert('RGB')
	     image = np.array(image.resize((256,256)))
	     #image = image[:,:,0]
	     image = np.uint8(image*255)
	     #image = np.tile(image,3)
	     image = self.transform(image)
	     return image

	    def __len__(self):
		    return len(self.data)
		

    X_test.reset_index(drop=True, inplace=True)

    test_dataset = ChestXrayDataSet_plot(input_X = X_test,transform=transforms.Compose([
                                            transforms.ToPILImage(),                                                                
                                            transforms.ToTensor(),               
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ]))
                                        

    print("generate heatmap ..........")
    # ======= Grad CAM Function =========
    class PropagationBase(object):
      def __init__(self, model, cuda=True):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

      def _set_hook_func(self):
        raise NotImplementedError

      def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

      def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        #self.probs = F.softmax(self.preds, dim=1)[0]
        #self.prob, self.idx = self.preds[0].data.sort(0, True)

        return self.preds.cpu().data.numpy()

      def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)
    

    class GradCAM(PropagationBase):
      def _set_hook_func(self):
        def func_f(module, input, output):
          self.all_fmaps[id(module)] = output.data.cpu()
    
        def func_b(module, grad_in, grad_out):
          self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
          module[1].register_forward_hook(func_f)
          module[1].register_backward_hook(func_b)

      def _find(self, outputs, target_layer):
        for key, value in outputs.items():
          for module in self.model.named_modules():
            if id(module[1]) == key:
              if module[0] == target_layer:
                return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

      def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item() 
  
      def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

      def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
          gcam += fmap * weight.data
    
        gcam = F.relu(Variable(gcam))
        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))    
        return gcam
    
      def FinalImage(self, gcam, raw_image):
        raw_image = raw_image*0.229+0.485
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = np.float32(gcam) / (600)
        gcam = gcam.astype(np.float) + raw_image.numpy()[0].astype(np.float).reshape(256,256,1)
        gcam = gcam / gcam.max() 
  
        return np.uint8(gcam * 255.0)
    
    
    # Testing with an image
    print(filePath)
    #rel_path = os.path.join("media/patient/BeforeAnalysis/", patient_name)
    #data_path = os.path.join(rel_path,'*g')
    #files = glob.glob(data_path) 
    #for f1 in files: 
     # img = cv2.imread(f1) 
    #rel_path
   # ext = ['png', 'jpg', 'gif'] 
    #rel_path = os.path.join(rel_path, '*.'+ e )
    #rel_path = [rel_path.extend(glob.glob(rel_path + '*.' + e)) for e in ext]
    #images = [cv2.imread(file) for file in files]
    #print(rel_path)
    data_path = filePath
    print(data_path)
    image = Image.open(data_path).convert('RGB')
    image = np.array(image.resize((256,256)))
    image = np.uint8(image*255)
    transform=transforms.Compose([
                                        transforms.ToPILImage(),                                                                
                                        transforms.ToTensor(),               
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])
    image = transform(image)


    heatmap_output = []
    image_id = []
    output_class = []

    thresholds = 0.0
    dataiter = iter(validation_loader)
    images = dataiter.next()
    gcam = GradCAM(model=model, cuda=True)
    for index in range(0,1):
      input_img = Variable((image).unsqueeze(0).cuda(), requires_grad=True)
      probs = gcam.forward(input_img)
      activate_classes = np.where((probs > thresholds)[0]==True)[0] # get the activated class
      for activate_class in activate_classes:
        gcam.backward(idx=activate_class)
        output = gcam.generate(target_layer="densenet121.features.denseblock4.denselayer16.conv2")
        #### this output is heatmap ####
        if np.sum(np.isnan(output)) > 0:
          print("fxxx nan")
        img = gcam.FinalImage(output, image)
        heatmap_output.append(img)
        image_id.append(index)
        output_class.append(activate_class)
    

    relative_probs = probs/sum(probs[0])
    relative_probs


    max_prob_ind = -1
    max_prob = -1
    for i in range(0,6):
      if(relative_probs[0][i]>max_prob):
        max_prob = relative_probs[0][i]
        max_prob_ind = i

    img = Image.fromarray(heatmap_output[max_prob_ind], 'RGB')
    quality_val=95 
    # Saving heatmap image in this folder
    save_path = os.path.join("media/patient/AfterAnalysis/", patient_name)
    new_path =  os.mkdir(save_path) 
    name = "ResultImage.JPEG"
    file_path = os.path.join(save_path, name)
    ResultImage = img.save(file_path, quality=quality_val)
    #img.save('{final_path}/Save.png')
    #img.save('/{final_path}/Result.jpg', 'JPEG')

    if(max_prob_ind==0):
      print("Covid Positive")
      result = "Covid Positive"
    else:
      print("Covid Negative")
      result = "Covid Negative"

      return result,file_path







