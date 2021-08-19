from utils.training import *



def get_dataloaders(dataset, batch_size):
    if dataset == "chexpert":
        return get_chexpert_dataloaders(batch_size)
    if dataset == "mura":
        return mura_dataloaders(batch_size)
    if dataset == "rsna":
        return rsna_dataloaders(batch_size)


def prepare_rsna_datasamples():
    path = Path('/mnt/wamri/rsna/rsna_retro')
    paths = (path/'nocrop_jpg256').iterdir()
    paths = [str(path) for path in paths]
    image_paths = [path.split('/')[-1].strip('.jpg') for path in paths]
    
    df = pd.read_csv(path/'stage_2_train.csv')
    df = df.loc[df.ID.str.contains('any')]
    df['ID'] = df['ID'].map(lambda x: x.split('_')[:2])
    df['ID'] = df['ID'].map(lambda x: ('_'.join(p for p in x)))
    df = df.loc[df.ID.isin(image_paths)]
    
    normal_list = list(df.loc[df.Label==0].ID)
    abnormal_list = list(df.loc[df.Label==1].ID)

    normal_samples = resample(normal_list, n_samples=40000, replace=False, random_state=1)
    abnormal_samples = resample(abnormal_list, n_samples=20000, replace=False, random_state=1)

    train_normal, test_normal = normal_samples[:20000], normal_samples[20000:]
    train_abnormal, test_abnormal = abnormal_samples[:10000], abnormal_samples[10000:]


    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    train_set['ID'] = train_normal + train_abnormal
    train_set['Label'] = [0]*20000 + [1]*10000
    test_set['ID'] = test_normal + test_abnormal
    test_set['Label'] = [0]*20000 + [1]*10000
    
    return train_set, test_set
    
class rsna_dataset(Dataset):
    def __init__(self, train_set, transform=True):
        
        self.paths = ['/mnt/wamri/rsna/rsna_retro/nocrop_jpg256/'+str(p)+'.jpg' for p in train_set['ID']]
        self.labels = [p for p in train_set['Label']]
        self.len = len(self.paths)
        
        # Transformations
        if transform:
            self.tfms = transforms.Compose([transforms.RandomResizedCrop((256, 256), 
                                                                         scale= (0.8, 1), ratio=(0.7,1.3)),
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.RandomRotation(30),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                           ])
            
        else:
            self.tfms = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                           ])
        
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        path = self.paths[idx] 
        x = Image.open(path)
        x = self.tfms(x)
        
        y = self.labels[idx]

        return x, y
    

def rsna_dataloaders(batch_size):
    train_set, test_set = prepare_rsna_datasamples()
    train_ds = rsna_dataset(train_set, transform=True)
    valid_ds = rsna_dataset(test_set, transform=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,num_workers=4)
    
    return train_loader, valid_loader, valid_ds


def crop(im, r, c, target_r, target_c): return im[r:r+target_r, c:c+target_c]

def random_crop(x):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    r, c,*_ = x.shape
    r_pix = 8
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x):
    r, c,*_ = x.shape
    r_pix = 8
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)


def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, 
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def norm_for_imageNet(img):
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (img - imagenet_stats[0])/imagenet_stats[1]

## Mura dataset
def mura_dataloaders(batch_size):
    class MURAXrayDataSet(Dataset):
        """
        Basic Images DataSet
        Args:
        dataframe with data: image_id, label
        image_path
        """

        def __init__(self, df, image_path, transform=False):
            self.image_id = df["image_id"].values
            self.labels = df["label"].astype(int).values
            self.image_path = image_path
            self.transform = transform

        def __getitem__(self, index):
            image_id = self.image_id[index]
            path = self.image_path/"image-{}.png".format(image_id)
            x = cv2.imread(str(path)).astype(np.float32)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255
            if self.transform:
                rdeg = (np.random.random()-.50)*20
                x = rotate_cv(x, rdeg)
                x = random_crop(x)
                if np.random.random() > 0.5: x = np.fliplr(x).copy()
            else:
                x = center_crop(x)
            x = norm_for_imageNet(x)

            y = self.labels[index]
            y = np.expand_dims(y, axis=-1)
            return np.rollaxis(x, 2), y

        def __len__(self):
            return len(self.image_id)

    PATH = Path('/home/yinterian/yinterian.data2/mura')
    train = pd.read_csv(PATH/"MURA-v1.1/train_path_labels.csv")
    valid = pd.read_csv(PATH/"MURA-v1.1/valid_path_labels.csv")
    train_ds = MURAXrayDataSet(train, PATH/"train_350_270", transform=True)
    valid_ds = MURAXrayDataSet(valid, PATH/"valid_350_270")
   
    train_dl = DataLoader(train_ds, batch_size=batch_size,  shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    
    return train_dl, valid_dl, valid_ds


#Chexpert Dataset
class chexpert_dataset(Dataset):
    def __init__(self, df, image_path, transform=None):
        self.image_files = df["Path"].values
        self.labels = df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']].values
        self.labels[self.labels==-1] = 0
        self.image_path = image_path
        if transform:
            self.tfms = transforms.Compose([
                transforms.RandomResizedCrop(
                (256, 256), scale= (0.8, 1), ratio=(0.7,1.3)),
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            
        else:
            self.tfms = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                           ])
        self.len = len(self.image_files)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        path = self.image_path/self.image_files[index]
        x = Image.open(path)
        x = self.tfms(x)
        y = self.labels[index]
        return x, y

def get_chexpert_dataloaders(batch_size):
    PATH = Path('/mnt/wamri/gilmer/chexpert/ChesXPert-250')
    train_df = pd.read_csv('/home/rimmanni/experiments/chexpert/train_df_chexpert_resized.csv')
    valid_df = pd.read_csv('/home/rimmanni/experiments/chexpert/valid_df_chexpert_resized.csv')
    
    train_ds = chexpert_dataset(train_df, image_path=PATH, transform=True)
    valid_ds = chexpert_dataset(valid_df, image_path=PATH, transform=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=4)
    
    return train_loader, valid_loader, valid_ds
