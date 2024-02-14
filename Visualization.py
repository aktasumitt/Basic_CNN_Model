import matplotlib.pyplot as plt
import torch


def Visualization_Data(train_dataset,class_dict,visualize_size=20):
    image_index=torch.randint(0,10000,(visualize_size,))
    
    for i,index in enumerate(image_index):
     
        plt.subplot(int(visualize_size/5),5,i+1)
        plt.imshow(torch.transpose(torch.transpose(train_dataset[index.item()][0],0,2),0,1))
        plt.xticks([])
        plt.xlabel(f"{train_dataset[index.item()][1]} : {class_dict[train_dataset[index.item()][1]]}")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.show()





def Visualization_Test(Model,test_data,devices):
      print("Visualization...\n")
      
      with torch.no_grad():
        for batch_test,(img,label) in enumerate(test_data):
            img_test=img.to(devices)
            label_test=label.to(devices)
            out_test=Model(img_test)
            _,predict_test=torch.max(out_test,1)
            if batch_test<25:
                plt.subplot(5,5,batch_test+1)
                plt.imshow(torch.transpose(torch.transpose(img[0].cpu().detach(),0,2),0,1))
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(f"p={predict_test[0]},r={label_test[0]}")
        
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
   
        plt.show()
                    