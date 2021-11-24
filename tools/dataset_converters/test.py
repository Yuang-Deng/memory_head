import json

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

with open('/data/dya/dataset/coco/instances_unlabeledtrainval20class0.json', 'r') as f:
    label = json.load(f)
    id_c_map = {}
    for ci in label['categories']:
        id_c_map[ci['id']] = ci['name']
    cat_set = set()
    for ann in label['annotations']:
        cat_set.add(id_c_map[ann['category_id']])
    print(cat_set)