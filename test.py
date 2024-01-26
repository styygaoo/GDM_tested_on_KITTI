import pickle
import matplotlib.pyplot as plt
import os

# pkl = open('/HOMES/yigao/VKITTI_2_KITTI/Splits/kitti/eigen_test.pickle', 'rb')
# im = pickle.load(pkl)
# print(im)
# plt.imshow(im)


dict = { '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000000.png': 1,
         '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000003.png': 1,
         '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000006.png': 1,
         '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000009.png': 1,
         '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/00000000012.png': 1,
         '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/00000000015.png': 1,
         '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/00000000018.png': 1,
         '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/00000000021.png': 1,
         '2011_09_26/2011_09_26_drive_0002_sync/image_02/data/00000000024.png': 1
         }
filename = "test_pkl.pkl"


if not os.path.isfile(filename):
   with open(filename,'wb') as file:
       pickle.dump(dict, file)
   file.close()