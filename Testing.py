import torch,tqdm


def Test_Model(test_data,Model,loss_fn,devices):
    
    Prog_BAR=tqdm.tqdm(range(len(test_data)),"Test Progress")
    
    with torch.no_grad():
        correct_test=0.0
        total_test=0.0
        loss_test=0.0

        for batch_test,(img,label) in enumerate(test_data):
            img_test=img.to(devices)
            label_test=label.to(devices)
            out_test=Model(img_test)
            loss=loss_fn(out_test,label_test)
            _,predict_test=torch.max(out_test,1)
            
            correct_test+=(predict_test==label_test).sum().item()
            total_test+=label.size(0)
            loss_test+=loss.item()
            
            Prog_BAR.update(1)
        
        
        acc_test=100*(correct_test/total_test)
        
        
        Prog_BAR.set_postfix_str(f"Loss_Test: {(loss_test/(batch_test+1)):.3f}"
                                f"    Acc_Test: {100*(correct_test/total_test):.3f}")
        
        Prog_BAR.close()
        
        return loss_test,acc_test