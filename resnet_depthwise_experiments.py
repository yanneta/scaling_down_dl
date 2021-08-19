#!/usr/bin/env python
# coding: utf-8

# In[2]:


pwd


# In[3]:


import torch
torch.cuda.set_device(4)
print(torch.cuda.current_device())


# In[4]:


get_ipython().run_line_magic('run', '../../prepare_data.py')
get_ipython().run_line_magic('run', '../../architectures.py')


# In[5]:


batch_size = 32


# In[6]:


train_loader, valid_loader, valid_dataset = get_chexpert_dataloaders(batch_size)


# In[7]:


x, y = next(iter(train_loader))


# In[8]:


x.shape, y.shape


# In[9]:


model = resnet18(num_classes=5, block=depthwise_block).cuda()


# In[10]:


sum(p.numel() for p in model.parameters())


# In[9]:


get_ipython().run_cell_magic('time', '', 'lrs, losses = LR_range_finder(model, train_loader, \n                              loss_fn=F.binary_cross_entropy_with_logits, \n                              binary=False, lr_high=0.05)\nplot_lr(lrs, losses)')


# # Training

# In[11]:


widths = [1.0, 0.75, 0.5, 0.25]
depths = [[[[64, 2], [128, 2]], [[256, 2], [512, 1]]],
          [[[64, 2], [128, 2]], [[256, 1], [512, 1]]],
          [[[64, 2], [128, 1]], [[256, 1], [512, 1]]],
          [[[64, 2], [128, 1]], [[256, 2], [512, 1]]],
          [[[64, 1], [128, 1]], [[256, 2], [512, 1]]],
          [[[64, 1], [128, 1]], [[256, 1], [512, 1]]],
         ]


# In[ ]:


data = []

for w in widths:
    for d in depths:
        d_s = sum(j[1] for i in d for j in i)
        print('width multiplier - %.3f depth multiplier - %.3f' % (w, d_s))
        model = resnet18(num_classes=5, block=depthwise_block, width_mult=w, 
                         inverted_residual_setting1=d[0], 
                         inverted_residual_setting2=d[1]).cuda()
        
        p = sum(p.numel() for p in model.parameters())
        optimizer = create_optimizer(model, 0.001)
        score, t = train_triangular_policy(model, optimizer, train_loader, valid_loader, valid_dataset,
                                       loss_fn=F.binary_cross_entropy_with_logits, 
                                       dataset='chexpert', binary=False, max_lr=0.001, epochs=15)
        
        p = "/home/rimmanni/Medical_Images/Scaling_experiments/Chexpert/ResDepth_" + str(w) + '_' + str(depths.index(d))
        save_model(model, p)
        data.append([w, d_s, score, p, t])
        print('')


# In[ ]:


columns = ['width_x', 'depth_x', 'val_score', 'params', 'time_per_epoch']
df = pd.DataFrame(data=data, columns=columns)


# In[ ]:


df.to_csv("chexpert_resnet_depthwise_13.csv", index=False)


# In[ ]:


df_re = pd.read_csv('chexpert_resnet_depthwise_13.csv')


# In[ ]:


df_re.head()


# In[ ]:




