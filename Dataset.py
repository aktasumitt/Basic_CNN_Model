from torch.utils.data import DataLoader,random_split
from torchvision import transforms
import glob
from PIL import Image


def Transformer(reshape_img_size):
    transformer=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
                                transforms.Resize(size=(reshape_img_size,reshape_img_size)),
                                transforms.Grayscale(num_output_channels=1)
                                ])
    return transformer



def Load_From_Path(TRAIN_PATH):
    
    class_dict={}
    full_img_list=[]
    full_image_dict={}
    for i,file in enumerate(glob.glob(TRAIN_PATH+"/*")):
        full_img=[]
        class_dict[i]=file.split("\\")[-1]
        for img in glob.glob(file+"/*"):
            full_img_list.append(img)
            full_img.append(img)
        
        full_image_dict[i]=full_img
    
    return class_dict,full_image_dict,full_img_list
    


class DATASET():
    def __init__(self,class_dict,full_img_list,full_image_dict,transformer):
        self.class_dict=class_dict
        self.full_img_list=full_img_list
        self.transformer=transformer
        self.full_image_dict=full_image_dict
    
    def __get_class_name__(self,idx):
        return self.class_dict[idx] 
    
    def __get_label_size__(self):
        return len(self.class_dict) 
    
    def __len__(self):
        return len(self.full_img_list)
    
    def __getitem__(self,idx):
        image_path = self.full_img_list[idx]
        image=Image.open(image_path)
        image=self.transformer(image)
        
        class_item = -1
        for label,path in self.full_image_dict.items() :
           if image_path in path:
                class_item=label
                break
        
        return (image,class_item)



def Random_split_fn(VALID_SPLIT,TEST_SPLIT,train_dataset):
    valid_len=int(len(train_dataset)*VALID_SPLIT)
    test_len=int(len(train_dataset)*TEST_SPLIT)
    train_len=len(train_dataset)-(valid_len+test_len)

    train_dataset,valid_dataset,test_dataset=random_split(dataset=train_dataset,lengths=[train_len,valid_len,test_len])
    
    return train_dataset,valid_dataset,test_dataset



def Create_Dataloader(train_dataset,test_dataset,valid_dataset,BATCH_SIZE):
    train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    valid_dataloader=DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False)
    test_dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    return train_dataloader,valid_dataloader,test_dataloader