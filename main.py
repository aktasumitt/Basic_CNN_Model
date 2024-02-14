import Config,Dataset,Callbacks,Model,Testing,Training,Visualization,Predict
import torch,warnings
warnings.filterwarnings("ignore")

# Use GPU If You Have
devices=("cuda" if torch.cuda.is_available() else "cpu")
   
   
# Create Transformer
transformer=Dataset.Transformer(reshape_img_size=Config.RESHAPE_IMG_SIZE)


# LOADING İMAGE FROM PATHS
idx_class_dict,full_image_dict,full_img_list=Dataset.Load_From_Path(TRAIN_PATH=Config.TRAIN_PATH)


# Create Datasets
train_dataset=Dataset.DATASET(class_dict=idx_class_dict,full_img_list=full_img_list,full_image_dict=full_image_dict,transformer=transformer)


# Create these from dataset for Model
label_size=train_dataset.__get_label_size__()
img_shape=train_dataset[3][0].shape



# Random split for validation and test data
train_dataset,valid_dataset,test_dataset=Dataset.Random_split_fn(VALID_SPLIT=Config.VALID_SPLIT,
                                                                 TEST_SPLIT=Config.TEST_SPLIT,
                                                                 train_dataset=train_dataset)


# Create Dataloader for Batch
train_dataloader,valid_dataloader,test_dataloader=Dataset.Create_Dataloader(train_dataset=train_dataset,
                                                                            valid_dataset=valid_dataset,
                                                                            test_dataset=test_dataset,
                                                                            BATCH_SIZE=Config.BATCH_SIZE)


# Visualization image of data
Visualization.Visualization_Data(train_dataset=train_dataset,
                   class_dict=idx_class_dict)




# Create Model, Optimizer and Loss Function
model=Model.CNN_Model(channel_size=Config.CHANNEL_SIZE,label_size=label_size,img_size=Config.RESHAPE_IMG_SIZE).to(devices)
optimizers=torch.optim.Adam(params=model.parameters(),
                                lr=Config.LEARNING_RATE)
    
loss_fn=torch.nn.CrossEntropyLoss()



# Loading Checkpoint If You Have
if Config.LOADING_CHECKPOINT==True:
    
    checkpoint=torch.load(f=Config.CALLBACKS_PATH)
    
    start_epoch=Callbacks.load_checkpoint(Checkpoint=checkpoint,
                                optimizer=optimizers,
                                model=model)
    print("Model is Loaded...")
    
else:
    start_epoch=0

    

# Train Model
if Config.TRAIN_MODEL==True:
    Training.Training_Model(start_epoch=start_epoch,
                   EPOCHS=Config.EPOCHS,
                   train_dataloader=train_dataloader,
                   valid_dataloader=valid_dataloader,
                   loss_fn=loss_fn,
                   optimizers=optimizers,
                   Model=model,
                   save_checkpoint=Callbacks.save_checkpoint,
                   PATH_CHECKPOİNT=Config.CALLBACKS_PATH,
                   devices=devices)



# TEST Model
if Config.TEST_MODEL==True:
    loss_test,acc_test=Testing.Test_Model(test_data=test_dataloader,Model=model,loss_fn=loss_fn,devices=devices)



# Visualization Test Predictions
if Config.VISUALIZATION==True:
    Visualization.Visualization_Test(Model=model,
                       test_data=test_dataloader,
                       devices=devices)
    

# Predictions with Custom Image   
if Config.PREDICT==True:
    Predict.predict(prediction_folder_path=Config.PREDICTION_FOLDER_PATH,
                    model=model,
                    idx_class_dict=idx_class_dict,
                    visualize_img=True,
                    img_size=Config.RESHAPE_IMG_SIZE)