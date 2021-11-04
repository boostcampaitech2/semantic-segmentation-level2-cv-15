import torch
import os
import numpy as np


#def weight_averaging():

fold_num=[3,4]
root_dir='/opt/ml/segmentation/semantic-segmentation-level2-cv-15/mmsegmentation/Pretrained_model'
model_path_list = [os.path.join(root_dir,f'swin_plain_Fold_{i}.pth')  for i in fold_num]

model_list = [torch.load(i) for i in model_path_list]





    #print(model_list)
    # print(list(model.keys()))
    # print(model['state_dict'])

print("Model's state_dict:")
#print(model['state_dict'])


for param_tensor in model_list[0]['state_dict']:
    tmp = torch.zeros(model_list[0]['state_dict'][param_tensor].shape)
    for model in model_list:
        tmp += model['state_dict'][param_tensor]
    result=np.true_divide(tmp,len(model_list))
    
    
    model_list[0]['state_dict'][param_tensor]=result

torch.save(model_list[0],'/opt/ml/segmentation/semantic-segmentation-level2-cv-15/mmsegmentation/swa_model/weight_average_model_3.pth')


'''
    print(param_tensor, "\t", model['state_dict'][param_tensor].size())
    answer=model['state_dict'][param_tensor]+ model['state_dict'][param_tensor]
    result=np.true_divide(answer,len(fold_num))
    print('----default----')
    print(model['state_dict'][param_tensor])
    #result=answer//len(fold_num)
    print('----diversion----')
    print(answer)
    break

'''

'''
p=[]
for param_tensor in model_list[0]['state_dict']:
    model=model_list[0]
    
    answer=model['state_dict'][param_tensor]+ model['state_dict'][param_tensor]
    
print(answer)

for model in model_list:
    p.append(model['state_dict'][param_tensor])
print(sum(p))

'''