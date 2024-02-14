import torch,tqdm
import time



def Training_Model(start_epoch,EPOCHS,train_dataloader,valid_dataloader,loss_fn,optimizers,Model,save_checkpoint,PATH_CHECKPOİNT,devices):
    
    if start_epoch>0:
        print(f"Training is starting from {start_epoch+1}.epoch ....")  
    
    else:
        print("Training is starting from scratch...\n")
    
    for epoch in range(start_epoch,EPOCHS):
        print(f"{epoch+1}.Epoch is starting...\n")
        
        total_train_value=0.0
        correct_train_value=0.0
        loss_train_value=0.0

        Progress_bar=tqdm.tqdm(range(len(train_dataloader)),"Training_Progress")
        
        for batch_train,(img,label) in enumerate(train_dataloader,0):
            img_train=img.to(devices)
            label_train=label.to(devices)
            
            starting_time=time.time()

            optimizers.zero_grad()
            out_train=Model(img_train)
            loss=loss_fn(out_train,label_train)
            _,prediction_train=torch.max(out_train,1)
            loss_train_value+=loss.item()
            loss.backward()
            optimizers.step()
            
            correct_train_value+=(prediction_train==label_train).sum().item()
            total_train_value+=label_train.size(0)                        
            
            if batch_train==len(train_dataloader)-1:
                
                total_valid_value=0.0
                correct_valid_value=0.0
                loss_valid_value=0.0
                
                
                with torch.no_grad():
                    
                    for batch_valid,(img,label) in enumerate(valid_dataloader,0):
                        
                        img_valid=img.to(devices)
                        label_valid=label.to(devices)

                        out_valid=Model(img_valid)
                        loss_valid=loss_fn(out_valid,label_valid)
                        _,predict_valid=torch.max(out_valid,1)
                        
                        correct_valid_value+=(predict_valid==label_valid).sum().item()

                        loss_valid_value+=loss_valid.item()
                        total_valid_value+=label_valid.size(0)

                    finish_time=time.time()  
                    Progress_bar.set_postfix({"EPOCH": f"{epoch+1}/{EPOCHS}",
                                                "BATCH": f"{batch_train+1}/{len(train_dataloader)}",
                                                "ACCURACY_TRAIN": f"{(100*(correct_train_value/total_train_value)):.3f}",
                                                "LOSS_TRAIN": f"{(loss_train_value/(batch_train+1)):.3f}",
                                                "ACCURACY_VALID": f"{(100*(correct_valid_value/total_valid_value)):.3f}",
                                                "LOSS_VALID": f"{(loss_valid_value/(batch_valid+1)):.3f}",
                                                "Time": f"{(finish_time-starting_time):.3f} Second"})
                                    

            
            Progress_bar.update(1)
        Progress_bar.close()
                                     
        save_checkpoint(epoch=epoch+1,
                        optimizers=optimizers,
                        Model=Model,
                        PATH_CHECKPOİNT=PATH_CHECKPOİNT)                       
        