import json
import os

{'cat', 'airplane', 'cow', 'horse', 'train', 
'boat', 'chair', 'person', 'tv', 'sheep', 'bicycle', 
'motorcycle', 'bus', 'dining table', 'dog', 'bird', 
'potted plant', 'car', 'couch', 'bottle'}

{'airplane', 'bicycle', 'bird', 'boat', 'bottle',
'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 
'dog', 'horse', 'motorcycle', 'person', 'potted plant',
'sheep', 'couch', 'train', 'tv'}

{'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
'tvmonitor'}

# with open('/data/dya/dataset/coco/instances_unlabeledtrainval20class0.json', 'r') as f:
#     label = json.load(f)
#     id_c_map = {}
#     for ci in label['categories']:
#         id_c_map[ci['id']] = ci['name']
#     cat_set = set()
#     for ann in label['annotations']:
#         ann['id'] = 1 * 10000000 + ann['id']
#         cat_set.add(id_c_map[ann['category_id']])
#     print(cat_set)

# s = 'instances_unlabeledtrainval20class0.json'

# sss = s.split('.')[0].split('/')[-1]
# print(sss)

# env_dist = os.environ

# os.environ['coco_test'] = '12312'

# print(env_dist['coco_test'])


with open('/data/dya/workspace/memory_head/ann_data/test.json', 'w+') as f:
    testdict = {'asasdasd':32}
    json.dump(testdict, f)