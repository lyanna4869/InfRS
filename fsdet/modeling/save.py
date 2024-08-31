import os
import os.path
import csv
import numpy as np
import torch
from fsdet.structures.instances import Instances

classes = ('airplane','ship','storage tank','baseball diamond','tennis court','ground track field','basketball court','habor','bridge','vehicle','__background__')

save_txt_path = '/cosine_sim_plot/results/'

def save_results(batched_inputs ,results):
    # img_info = batched_inputs[0]['annotations'][0]['category_id']
    # object_id = batched_inputs[0]['annotations'][0]['object_id']
    img_No = batched_inputs[0]['image_id']
    file_name = save_txt_path+img_No+'image'+'.csv'
    os.makedirs(os.path.dirname(file_name),exist_ok=True)

    results_tobesaved = results
    box_feature = results_tobesaved[0]['instances'].box_feature.cpu().numpy()
    pred_cla = int(results_tobesaved[0]['instances'].pred_classes)
   # for idx in range(len(batched_inputs[0]['annotations'])):
    category_id = batched_inputs[0]['annotations'][0]['category_id']
    object_id = batched_inputs[0]['annotations'][0]['object_id']

    things = [category_id,object_id,pred_cla]
    headers = ['class_info','object_id','pred_cla','box_features']



    with open(file_name,"w",encoding='utf-8',newline='') as f:
        write = csv.writer(f)
        write.writerow(headers)
        write.writerow(things)
        # for i in range(len(things)):
        #     write.writerow(things[i])
        write.writerows(box_feature)