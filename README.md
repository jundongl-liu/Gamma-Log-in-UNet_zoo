# Gamma-Log-in-UNet_zoo
# Environment

window10 or Ubuntu, 

cuda10.1 + cudnn 7.6.5 + python3.6,

torch1.7.1 + torchvision 0.8.2,

matplotlib, sklearn, scikit-image, opencv-python

# How to run

1.you need to find the file "dataset.py" in lines 72,73,74
  
  and modify the path to the storage path for your project
  ```
  line72      self.train_root = "/your_path/data/oil-spill-detection-dataset/train"
  
  line73      self.val_root = "/your_path/data/oil-spill-detection-dataset/val"
  
  line74      self.test_root = "/your_path/data/oil-spill-detection-dataset/test"
  ```
2.you need to see the lines 346,347,348 in "main.py"

  If you use linux system to run this project, you can use line346 and line347.
  
  If you use windows system to run this project, you can use line346 and line348.
  
  and then, you can get the predict results.                   
  
  ```
  line346             predict =Image.fromarray(predict).convert('L')         #predict
  
  line347             #predict.save(dir +'/'+mask_path[0].split('/')[-1])    #linux
  
  line348             predict.save(dir +'\\'+mask_path[0].split('\\')[-1])   #win
  ```

3.you need to see the lines 350,351,352 in "main.py"

  If you use linux system to run this project, you can use line350 and line351.
  
  If you use windows system to run this project, you can use line350 and line352.
   
  If you want to get the differences between the predict image and the ground truth,
  
  you can delete # in line350,351 or line350,352 .

  ```
  line350             #i_u_img = Image.fromarray(i_u_img).convert('L')       #difference
   
  line351             #i_u_img.save(dir +'/'+mask_path[0].split('/')[-1])    #linux
   
  line352             #i_u_img.save(dir +'\\'+mask_path[0].split('\\')[-1])  #win
  ```

4.You can run this project by using the command line, just like this.
 
  ```python main.py --action test --arch Attention_UNet --epoch 51 --batch_size 4```
  
  in ```--arch```, you can use UNet, UNet_AGC, unet++, unet++_AGC, Attention_UNet, Attention_UNet_AGC
                         r2unet, r2unet_AGC, fcn8s, fcn8s_AGC, fcn32s, fcn32s_AGC
  
  UNet_AGC means the UNet with Gamma_Log Net.
  
  
# Results
 
 you will get the results in "saved_predict" folder.
 The predict results and the difference images are in the same place.
 
 The specific values of iou and difference are displayed at the interface.
  
  
  
  
 




