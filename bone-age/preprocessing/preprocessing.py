k = 100
size = 500
image_filenames = glob.glob(train_dataset_path+'*.png')
random_images = random.sample(population = image_filenames,k = k)


means = []
stds = []

for filename in random_images:
    image = cv2.imread(filename,0)
    image = cv2.resize(image,(size,size))
    mean,std = cv2.meanStdDev(image)
#    mean /= 255
#    std /= 255
    
    means.append(mean[0][0])
    stds.append(std[0][0])

avg_mean = np.mean(means) 
avg_std = np.mean(stds)

print('Approx. Mean of Images in Dataset: ',avg_mean)
print('Approx. Standard Deviation of Images in Dataset: ',avg_std)




# To reproduce results use below values
#avg_mean = 52.96
#avg_std = 26.19
#%%

# Split Train Validation Test
# Train - 10000 images
# Val   -  1611 images
# Test  -  1000 images

dataset_size = len(image_filenames)
val_size = dataset_size + 1611



bones_df = pd.read_csv(csv_path)
bones_df.iloc[:,1:3] = bones_df.iloc[:,1:3].astype(np.float)


train_df = bones_df.iloc[:dataset_size,:]
val_df = bones_df.iloc[dataset_size:val_size,:]
test_df = bones_df.iloc[val_size:,:]


age_max = np.max(bones_df['boneage'])
age_min = np.min(bones_df['boneage'])
#%%
class BonesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):

        self.dataframe = dataframe

        
        self.image_dir = image_dir
        self.transform = transform
        

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_name = self.image_dir + str(self.dataframe.iloc[idx,0]) + '.png'
        image = cv2.imread(img_name,0)
        image = image.astype(np.float64)
        gender = np.atleast_1d(self.dataframe.iloc[idx,2])
        bone_age = np.atleast_1d(self.dataframe.iloc[idx,1])
        
        

        sample = {'image': image, 'gender': gender, 'bone_age':bone_age}

        if self.transform:
            sample = self.transform(sample)

        return sample

#%% 
# Custom Transforms for Image and numerical data
        
# Resize and Convert numpy array to tensor
class ToTensor(object):
    

    def __call__(self, sample):
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']


        image = cv2.resize(image,(size,size))
        image = np.expand_dims(image,axis = 0)
        
#       we need to convert  cuda.longtensors to cuda.floatTensor data type
        return {'image': torch.from_numpy(image).float(),
                'gender': torch.from_numpy(gender).float(),
                'bone_age':torch.from_numpy(bone_age).float()}        

# Normalize images and bone age
class Normalize(object):
    
    def __init__(self,img_mean,img_std,age_min,age_max):
        self.mean = mean
        self.std = std
        
        self.age_min = age_min
        self.age_max = age_max
        
    
    
    def __call__(self,sample):
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']
        
        image -= self.mean
        image /= self.std
        
        bone_age = (bone_age - self.age_min)/ (self.age_max - self.age_min)
        
        
        
        return {'image': image,
                'gender': gender,
                'bone_age':bone_age} 
        


data_transform = transforms.Compose([
   Normalize(avg_mean,avg_std,age_min,age_max),
   ToTensor()
   
   ])     
    


#%%
train_dataset = BonesDataset(dataframe = train_df,image_dir=train_dataset_path,transform = data_transform)
val_dataset = BonesDataset(dataframe = val_df,image_dir = val_dataset_path,transform = data_transform)
test_dataset = BonesDataset(dataframe = test_df,image_dir=test_dataset_path,transform = data_transform)

# Sanity Check
print(train_dataset[0])

     
train_data_loader = DataLoader(train_dataset,batch_size=4,shuffle=False,num_workers = 4)
val_data_loader = DataLoader(val_dataset,batch_size=4,shuffle=False,num_workers = 4)
test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers = 4)
