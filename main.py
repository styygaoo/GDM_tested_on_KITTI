import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from get_kitti import KITTIDataset
from model_loader import load_model
from transforms import ToTensor
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torch.optim as optim
from data_processing import unpack_and_move, inverse_depth_norm, depth_norm
from config_model_TTA import configure_model, collect_params
import oer
import time
from metrics import AverageMeter, Result


def save_results(average):
    file_path = os.path.join("./results", '{}.txt'.format("metrics_each_step_P1"))
    with open(file_path, 'a') as f:
        f.write('RMSE,MAE,REL,Lg10,Delta1,Delta2,Delta3,Epoch\n')
        f.write('{average.rmse:.3f}'
                ',{average.mae:.3f}'
                ',{average.absrel:.3f}'
                ',{average.lg10:.3f}'
                ',{average.delta1:.3f}'
                ',{average.delta2:.3f}'
                ',{average.delta3:.3f}'
                ',{epoch}\n'.format(
                    average=average, epoch=epoch + 1))

crops = {
    'kitti' : [128, 381, 45, 1196],
    'nyu' : [20, 460, 24, 616],
    'nyu_reduced' : [20, 460, 24, 616]}


transformation = T.ToTensor()
trans = T.Compose([T.ToTensor()])
batch_size = 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
maxDepth = 80
My_to_tensor = ToTensor(test=True, maxDepth=maxDepth)

# Load the GDM and the checkpoint
# model_original = load_model('GuideDepth', '/HOMES/yigao/KITTI_2_VKITTI/KITTI_Half_GuideDepth.pth')
model_original = load_model('GuideDepth', '/HOMES/yigao/VKITTI_2_KITTI/Models/vkitti/checkpoint_19.pth')

# Load model parameter to be fine-tuned during test phase
model = configure_model(model_original)
params, param_names = collect_params(model)


# Prepare test dataloader for TTA
testset = KITTIDataset('/HOMES/yigao/Downloads/eval_testset/KITTI_test', (192, 640))
testset_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)       # , drop_last=True


batches = len(testset)/batch_size

# Define loss function and optimizer for fine-tuning
optimizer = optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0)

# oering the given model to make it adaptive for test data
adapted_model = oer.OER(model, optimizer)
adapted_model.cuda()

average_meter = AverageMeter()
loss_values_epoch = []
loss_values_step = []
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(testset_loader):
        t0 = time.time()
        images, gts = data
        for b in range(images.shape[0]):
            # print(b)
            packed_data = {'image': images[b], 'depth': gts[b]}
            data = My_to_tensor(packed_data)
            image, gt = unpack_and_move(data)
            # image, gt = data['image'], data['depth']
            image = image.unsqueeze(0)
            gt = gt.unsqueeze(0)
            # print(gt)
            if b >= 1:
                batched_images = torch.cat((batched_images, batched_image))
                batched_gts = torch.cat((batched_gts, batched_gt))
            else:
                batched_images = image
                batched_gts = gt
            batched_image = image
            batched_gt = gt

            batched_image.detach().cpu()
            batched_gt.detach().cpu()
            image = image.detach().cpu()
            gt = gt.detach().cpu()
        # print(batched_images.shape)
        data_time = time.time() - t0
        t0 = time.time()
        # print("{:.3f}MB allocated".format(torch.cuda.memory_allocated() / 1024 ** 2))
        torch.cuda.empty_cache()  # Releases all unoccupied cached memory currently held by the caching allocator
        inv_prediction, loss = adapted_model(batched_images)
        loss_values_step.append(loss.detach().cpu())
        # inv_prediction = model_original(batched_images).detach().cpu()
        predictions = inverse_depth_norm(inv_prediction)

        batched_images = batched_images.detach().cpu()
        batched_gts = batched_gts.detach().cpu()
        predictions = predictions.detach().cpu()

        gpu_time = time.time() - t0

        result = Result()
        result.evaluate(predictions.data, batched_gts.data)
        average_meter.update(result, gpu_time, data_time, image.size(0))

    running_loss += loss.item() * images.size(0)
    loss_values_epoch.append(running_loss / batches)
    save_results(average_meter.average())

# Report
avg = average_meter.average()
current_time = time.strftime('%H:%M', time.localtime())

print('\n*\n'
      'RMSE={average.rmse:.3f}\n'
      'MAE={average.mae:.3f}\n'
      'Delta1={average.delta1:.3f}\n'
      'Delta2={average.delta2:.3f}\n'
      'Delta3={average.delta3:.3f}\n'
      'REL={average.absrel:.3f}\n'
      'Lg10={average.lg10:.3f}\n'
      't_GPU={time:.3f}\n'.format(
    average=avg, time=avg.gpu_time))


