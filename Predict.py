import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob

def predict(prediction_folder_path,model,idx_class_dict:dict,visualize_img:bool,img_size:int):
        
    img_path_list=glob.glob(prediction_folder_path+"/*")
    
    with torch.no_grad():
        img_list=[]
        model.to("cpu")
        
        for i,img_path in enumerate(img_path_list):
            img=Image.open(img_path)
            img_transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5,),(0.5,)),
                                              transforms.Resize((img_size,img_size)),
                                               transforms.Grayscale(1)])(img)
            
            img_list.append(img_transform)            
        
        out=model(torch.stack(img_list))
        _,predict=torch.max(out,1)
        
        for i in range(len(img_list)):
            
            print(f"Predicted label {i+1}.image: ",idx_class_dict[predict[i].item()])
            
            if visualize_img==True:
                plt.subplot(int(len(img_list)/2),2,i+1)
                plt.imshow(torch.permute(img_list[i],(1,2,0)),cmap="gray")
                plt.xlabel(idx_class_dict[predict[i].item()])
                plt.xticks([])
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
                
            
        plt.show()
            
            
    