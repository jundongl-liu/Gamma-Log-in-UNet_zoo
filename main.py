import argparse
import logging
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from UNet import Unet,resnet34_unet
from attention_unet import AttU_Net
from channel_unet import myChannelUnet
from r2unet import R2U_Net
from segnet import SegNet
from unetpp import NestedUNet
from fcn import get_fcn8s,get_fcn32s
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot
from torchvision.models import vgg16

from UNet_AGC import Unet_AGC,resnet34_unet_AGC,Unet_AGC1
from attention_unet_AGC import AttU_Net_AGC
from unetpp_AGC import NestedUNet_AGC
from segnet_AGC import SegNet_AGC
from r2unet_AGC import R2U_Net_AGC
from fcn_AGC import get_fcn8s_AGC,get_fcn32s_AGC
from channel_unet_AGC import myChannelUnet_AGC

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
    parse.add_argument("--epoch", type=int, default=101)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument('--dataset', default='oil-spill-detection-dataset',  # dsb2018_256 #oil
                       help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung')
    #parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet_AGC':
        model = Unet_AGC(3, 1).to(device)
    if args.arch == 'UNet_AGC1':
        model = Unet_AGC1(3, 1).to(device)
    if args.arch == 'UNet':
        model = Unet(3, 1).to(device)

    if args.arch == 'resnet34_unet_AGC':
        model = resnet34_unet_AGC(1,pretrained=False).to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(1,pretrained=False).to(device)

    if args.arch == 'unet++_AGC':
        args.deepsupervision = True
        model = NestedUNet_AGC(args,3,1).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = True
        model = NestedUNet(args,3,1).to(device)

    if args.arch =='Attention_UNet_AGC':
        model = AttU_Net_AGC(3,1).to(device)
    if args.arch =='Attention_UNet':
        model = AttU_Net(3,1).to(device)

    if args.arch == 'segnet_AGC':
        model = SegNet_AGC(3,1).to(device)
    if args.arch == 'segnet':
        model = SegNet(3,1).to(device)

    if args.arch == 'r2unet_AGC':
        model = R2U_Net_AGC(3,1).to(device)
    if args.arch == 'r2unet':
        model = R2U_Net(3,1).to(device)

    if args.arch == 'fcn32s_AGC':
        model = get_fcn32s_AGC(1).to(device)
    if args.arch == 'fcn32s':
        model = get_fcn32s(1).to(device)

    if args.arch == 'myChannelUnet_AGC':
        model = myChannelUnet_AGC(3,1).to(device)
    if args.arch == 'myChannelUnet':
        model = myChannelUnet(3,1).to(device)


    if args.arch == 'fcn8s_AGC':
        assert args.dataset !='esophagus'
        model = get_fcn8s_AGC(1).to(device)
    if args.arch == 'fcn8s':
        assert args.dataset !='esophagus'
        model = get_fcn8s(1).to(device)

    if args.arch == 'cenet_AGC':
        from cenet_AGC import CE_Net_AGC
        model = CE_Net_AGC().to(device)
    if args.arch == 'cenet':
        from cenet import CE_Net_
        model = CE_Net_().to(device)
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset =='oil':  #E:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = oilDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = oilDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = oilDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
     
    if args.dataset =='oil-spill-detection-dataset':  #E:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = newoilDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
        val_dataset = newoilDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1, num_workers=4)
        test_dataset = newoilDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1, num_workers=4)
   
    
    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model,best_iou,val_dataloaders,epoch):
    model= model.eval().to(device)
    sea_iou, oil_iou, lookalike_iou,ship_iou, land_iou =0,0,0,0,0 #new
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        #print(num)
        for x, _,pic,mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
                #print(img_y.shape)
            """
            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0],img_y)  #获取当前预测图的miou，并加到总miou中
            dice_total += get_dice(mask[0],img_y)
            """
            iou,sea_list,oil_list,lookalike_list,ship_list,land_list = get_multi_iou(mask[0],img_y)  #获取当前预测图的miou，并加到总miou中

            miou_total += iou
            sea_iou += sea_list[4]
            oil_iou += oil_list[4]
            lookalike_iou += lookalike_list[4]
            ship_iou += ship_list[4]
            land_iou += land_list[4]


            if i < num:i+=1   #处理验证集下一张图

        aver_iou = miou_total / num
        aver_sea_iou = sea_iou / num
        aver_oil_iou = oil_iou / num
        aver_lookalike_iou = lookalike_iou / num
        aver_ship_iou =ship_iou / num
        aver_land_iou = land_iou / num

        """
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou,aver_hd,aver_dice))
        """
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')

        return best_iou,aver_iou,aver_sea_iou,aver_oil_iou,aver_lookalike_iou,aver_ship_iou,aver_land_iou       #aver_dice,aver_hd

def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_iou,aver_iou,aver_dice,aver_hd = 0,0,0,0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []

    model.load_state_dict(torch.load(
            r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(
                args.epoch) + '.pth', map_location='cpu'))

    for epoch in range(num_epochs):
        
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y,_,mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            #print(inputs )
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, labels)
                loss /= len(outputs)
            else:
                output = model(inputs)


                loss = criterion(output, torch.squeeze(labels).long())

                """
                #focal_loss
                
                logit = output
                target = labels
                n, c, h, w = logit.size()
                criterion_ce = nn.CrossEntropyLoss()
                criterion_ce = criterion_ce.to(device)                    
                
                logpt = -criterion_ce(logit, target.long())
                pt = torch.exp(logpt)
                alpha=0.25
                if alpha is not None:
                    logpt *= alpha
                loss = -((1 - pt) ** gamma) * logpt
                loss /= n
                
                """

                #print(loss)
            if threshold!=None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        #best_iou,aver_iou,aver_dice,aver_hd = val(model,best_iou,val_dataloader,epoch)
        best_iou, aver_iou, aver_sea_iou, aver_oil_iou, aver_lookalike_iou, aver_ship_iou, aver_land_iou = val(model,best_iou,val_dataloader,epoch)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice',iou_list, dice_list)
    metrics_plot(args,'hd',hd_list)
    return model

def test(val_dataloaders,save_predict=True):
    logging.info('final test........')
    dice_list = []
    iou_list = []
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(r'./saved_model/oil_spill_detection/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    #plt.ion() #开启动态模式
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        iou_up = 0
        dice_up = 0
        iou_standard = 0
        dice_standard = 0
        num = len(val_dataloaders)  #验证集图片的总数
        for pic,_,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic) 
            
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            iou,dif_area,i_u_img = get_iou(mask_path[0],predict)
            print()
            miou_total += iou  #获取当前预测图的miou，并加到总miou中
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0],predict)
            dice_total += dice

            dice_list.append(dice)
            iou_list.append(iou)


            predict = predict*256
            #print(np.max(predict))
            #fig = plt.figure()
            

            if save_predict == True:
                if args.dataset == 'driveEye':
                    saved_predict = dir + '/' + mask_path[0].split('/')[-1]
                    saved_predict = '.'+saved_predict.split('.')[1] + '.tif'
                    plt.savefig(saved_predict)
                else:
                    #print(dir +'/'+ mask_path[0].split('/')[-1])
                    #plt.savefig(dir +'/'+mask_path[0].split('/')[-1])
                    predict =Image.fromarray(predict).convert('L')
                    predict.save(dir +'/'+mask_path[0].split('/')[-1])
                    
                    #i_u_img = Image.fromarray(i_u_img).convert('L')
                    #i_u_img.save(dir +'/'+mask_path[0].split('/')[-1])

            #plt.pause(0.01)
            print('iou={},dice={}'.format(iou,dice))
            if i < num:i+=1   #处理验证集下一张图
        #print('M_accuracy=%f,m_precision=%f,m_recall=%f,m_F1=%f' % (accuracy_list/num, precision_list/num, recall_list/num, F1_list/num ))
        #logging.info('M_accuracy=%f,m_precision=%f,m_recall=%f,m_F1=%f' % (accuracy_list/num, precision_list/num, recall_list/num, F1_list/num ))
        #plt.show()
        for l in range(len(iou_list)):
            iou_single = (iou_list[l] - miou_total / num) ** 2
            iou_up += iou_single
            dice_single = (dice_list[l] - dice_total / num) ** 2
            dice_up += dice_single
        iou_standard = (iou_up / num) ** 0.5
        dice_standard = (dice_up / num) ** 0.5

        print('Miou=%f,aver_hd=%f,dv=%f,Miou_standard=%f, aver_dice_standard=%f' % (miou_total/num,hd_total/num,dice_total/num,iou_standard,dice_standard))
        logging.info('Miou=%f,aver_hd=%f,dv=%f,Miou_standard=%f, aver_dice_standard=%f' % (miou_total/num,hd_total/num,dice_total/num,iou_standard,dice_standard))
        #print('M_dice=%f' % (dice_total / num))

if __name__ =="__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()
    device =torch.device("cuda")# torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)

    
    criterion = torch.nn.BCELoss()
    criterion = criterion.to(device)

    optimizer = optim.Adam( model.parameters())   #默认学习率0.001
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)#save_predict=True
