# -*- encoding:utf-8 -*-

import datetime
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import warnings
import time
import pandas as pd
import numpy as np
import os
import gc
from urllib import parse
import re
from  tqdm import tqdm_notebook
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_selection import  chi2
from sklearn.feature_selection import  SelectKBest
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import  StratifiedKFold
import numpy as np
import lightgbm as lgb
from sklearn.metrics import  log_loss
from sklearn.preprocessing import  OneHotEncoder
warnings.filterwarnings("ignore")

path = './Input/'

warnings.filterwarnings("ignore")
train = pd.read_table(path + 'round2_iflyad_train.txt')
test = pd.read_table(path + 'round2_iflyad_test_feature.txt')
data = pd.concat([train, test], axis=0, ignore_index=True)

data = data.fillna(-1)

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
data['label'] = data.click.astype(int)

def jiema(string):
    if string:
        reslut = ""
        string = str(string)
        while(True):
            result = parse.unquote(string)
            if result == string:
                break
            string = result
        fil = re.compile(u'[^0-9a-zA-Z_]+', re.UNICODE)
        result = fil.sub('',result)
        return result.lower()
    else:
        return string
object_feature = [col for col in data.columns if data[col].dtype == 'object' and col not in 'user_tags']
for col in object_feature:
    data[col] = data[col].map(jiema)
data['user_tags']=data['user_tags'].apply(lambda x:str(x).replace(","," "))
data['user_tags_copy']=data['user_tags'].str.split()
user_tags=data['user_tags'].str.split().tolist()
user_tags_list_all=[]
for i in  tqdm_notebook(user_tags):
    user_tags_list_all.extend(i)
user_tags_list=list(set(user_tags_list_all))
print(len(user_tags_list))

ut_df=pd.DataFrame({'ut_tags':user_tags_list_all})
ut_count=ut_df.groupby('ut_tags').agg('size').reset_index().rename(columns={0:'count'})
ut_count.sort_values(by=['count'],inplace=True,ascending=False)
data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])
data['advert_industry_inner_2'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[1])

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner_2', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 
                   'creative_is_jump', 'creative_is_download']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature
adid_nuq=['model','make','os','city','province','user_tags','f_channel','app_id','carrier','nnt', 'devtype',
         'app_cate_id','inner_slot_id']

#广告的曝光率 提升5个w
for fea in tqdm_notebook(adid_nuq):
    gp1=data.groupby('adid')[fea].nunique().reset_index().rename(columns={fea:"adid_%s_nuq_num"%fea})
    gp2=data.groupby(fea)['adid'].nunique().reset_index().rename(columns={'adid':"%s_adid_nuq_num"%fea})
    data=pd.merge(data,gp1,how='left',on=['adid'])
    data=pd.merge(data,gp2,how='left',on=[fea])   
    gc.collect()
    

advert_id_nuq=['model','make','os','city','province','user_tags','f_channel','app_id','carrier','nnt', 'devtype',
         'app_cate_id','inner_slot_id']

#广告主的曝光率
for fea in tqdm_notebook(advert_id_nuq):
    gp1=data.groupby('advert_id')[fea].nunique().reset_index().rename(columns={fea:"advert_id_%s_nuq_num"%fea})
    gp2=data.groupby(fea)['advert_id'].nunique().reset_index().rename(columns={'advert_id':"%s_advert_id_nuq_num"%fea})
    data=pd.merge(data,gp1,how='left',on=['advert_id'])
    data=pd.merge(data,gp2,how='left',on=[fea])   
    gc.collect()
    
app_id_nuq=['model','make','os','city','province','user_tags','f_channel','carrier','nnt', 'devtype',
         'app_cate_id','inner_slot_id']

#app id的曝光率 
for fea in tqdm_notebook(app_id_nuq):
    gp1=data.groupby('app_id')[fea].nunique().reset_index().rename(columns={fea:"app_id_%s_nuq_num"%fea})
    gp2=data.groupby(fea)['app_id'].nunique().reset_index().rename(columns={'app_id':"%s_app_id_nuq_num"%fea})
    data=pd.merge(data,gp1,how='left',on=['app_id'])
    data=pd.merge(data,gp2,how='left',on=[fea])   
    gc.collect()


#行业的曝光率
for fea in tqdm_notebook(adid_nuq):
    gp1=data.groupby('advert_industry_inner')[fea].nunique().reset_index().rename(columns={fea:"advert_industry_inner_%s_nuq_num"%fea})
    gp2=data.groupby(fea)['advert_industry_inner'].nunique().reset_index().rename(columns={'advert_industry_inner':"%s_advert_industry_inner_nuq_num"%fea})
    data=pd.merge(data,gp1,how='left',on=['advert_industry_inner'])
    data=pd.merge(data,gp2,how='left',on=[fea])   
    gc.collect()
  
#一级行业的曝光率
for fea in tqdm_notebook(adid_nuq):
    gp1=data.groupby('advert_industry_inner_1')[fea].nunique().reset_index().rename(columns={fea:"advert_industry_inner_1_%s_nuq_num"%fea})
    gp2=data.groupby(fea)['advert_industry_inner_1'].nunique().reset_index().rename(columns={'advert_industry_inner_1':"%s_advert_industry_inner_1_nuq_num"%fea})
    data=pd.merge(data,gp1,how='left',on=['advert_industry_inner_1'])
    data=pd.merge(data,gp2,how='left',on=[fea])   
    gc.collect()
  
#二级行业的曝光率
for fea in tqdm_notebook(adid_nuq):
    gp1=data.groupby('advert_industry_inner_2')[fea].nunique().reset_index().rename(columns={fea:"advert_industry_inner_2_%s_nuq_num"%fea})
    gp2=data.groupby(fea)['advert_industry_inner_2'].nunique().reset_index().rename(columns={'advert_industry_inner_2':"%s_advert_industry_inner_2_nuq_num"%fea})
    data=pd.merge(data,gp1,how='left',on=['advert_industry_inner_2'])
    data=pd.merge(data,gp2,how='left',on=[fea])   
    gc.collect()

data['user_tag_len_num']=data['user_tags'].map(lambda x: len(x))


user_id=['model','make','os','city','province','user_tags','campaign_id']

data['user_id']=data['model'].astype(str)+data['make'].astype(str)+\
data['city'].astype(str)+data['province'].astype(str)+data['user_tags'].astype(str)


gp1=data.groupby('adid')['user_id'].nunique().reset_index().rename(columns={'user_id':"adid_user_id_nuq_num"})
gp2=data.groupby('user_id')['adid'].nunique().reset_index().rename(columns={'adid':"user_id_adid_nuq_num"})
data=pd.merge(data,gp1,how='left',on=['adid'])
data=pd.merge(data,gp2,how='left',on=['user_id'])   
gc.collect()

count_fea=['adid','advert_id','app_id','advert_industry_inner','make','model','os','city','f_channel']
for fea in  count_fea:
    gp=data.groupby(fea).agg('size').reset_index().rename(columns={0:"%s_count_num"%fea})
    data=pd.merge(data,gp,how='left',on=fea)
    gc.collect()

#province to city
gc.collect()
gp1=data.groupby('province')['city'].nunique().reset_index().rename(columns={'city':"province_city_nuq_num"})
data=pd.merge(data,gp1,how='left',on=['province'])
del gp1
gc.collect()
#一级对二级
gc.collect()
gp1=data.groupby('advert_industry_inner_1')['advert_industry_inner_2'].nunique().reset_index().rename(\
                columns={'advert_industry_inner_2':"dvert_industry_inner_1_2_nuq_num"})
data=pd.merge(data,gp1,how='left',on=['advert_industry_inner_1'])
gc.collect()
for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))

cate_feature = origin_cate_list

num_feature = ['creative_width', 'creative_height', 'hour']
num_feature=num_feature+[i for i in data.columns if '_num' in i]

feature = cate_feature + num_feature

#print(len(feature), feature)

test_data = data[data.label == -1]

train_data = data[data.label != -1]
train_y = data[data.label != -1].label.values


train_data=data[data.click!=-1]
test_data=data[data.click==-1]
train_y=train_data['click'].values


for i,fea  in enumerate(cate_feature):
    ohe=OneHotEncoder()
    ohe.fit(data[fea].values.reshape(-1,1))
    tr_x=ohe.transform(train_data[fea].values.reshape(-1,1))
    te_x=ohe.transform(test_data[fea].values.reshape(-1,1))
    if i==0:
        Train_x=tr_x
        Test_x=te_x
    else:
        Train_x=sparse.hstack((Train_x,tr_x),'csr','bool')
        Test_x=sparse.hstack((Test_x,te_x),'csr','bool')
        
print('onehot feature  %d'%len(cate_feature))        
print(Train_x.shape,Test_x.shape)


oth_train_x=sparse.csr_matrix(train_data[num_feature].values)
oth_test_x=sparse.csr_matrix(test_data[num_feature].values)

train_x=sparse.hstack((Train_x,oth_train_x),'csr').astype('float32')
test_x=sparse.hstack((Test_x,oth_test_x),'csr').astype('float32')

train_x=train_x.tocsr()
test_x=test_x.tocsr()



NFOLD=5
oof_train=np.zeros((train_x.shape[0],1))
oof_test=np.zeros((test_x.shape[0],1))
oof_test_skf=np.zeros((NFOLD,test_x.shape[0],1))

lgb_params={"booster":'goss','objective':'binary','max_depth':-1,'metric':'binary_logloss',
            'lambda_l1':0,'lambda_l2':1,
            
            'num_leave':31,'max_bin':250,'min_data_in_leaf': 200,'learning_rate': 0.02,'feature_fraction': 0.8,
            
            'bagging_fraction': 0.7,'bagging_freq': 1}


kf=StratifiedKFold(NFOLD,shuffle=True,random_state=2018)
log_loss_csv=[]
for  i,(idx_train,idx_test) in enumerate(kf.split(train_x,train_y)):
    
 

    x_tr,x_te=train_x[idx_train],train_x[idx_test]
    y_tr,y_te=train_y[idx_train],train_y[idx_test]
    
    lgb_train=lgb.Dataset(x_tr,y_tr)
    lgb_val=lgb.Dataset(x_te,y_te)
    
    bst=lgb.train(lgb_params,train_set=lgb_train,valid_sets=[lgb_train,lgb_val],
                  num_boost_round=10000,early_stopping_rounds=500)
    
    val=bst.predict(x_te,num_iteration=bst.best_iteration)
    logloss=log_loss(y_te,val)
    print('logloss is %f'%logloss)
    log_loss_csv.append(logloss)
    oof_train[idx_test]=val.reshape(-1,1)
    pre=bst.predict(test_x,num_iteration=bst.best_iteration)
    oof_test_skf[i,:]=pre.reshape(-1,1)
logloss=np.mean(log_loss_csv)
print(logloss)
loss = str(logloss)
oof_test=oof_test_skf.mean(axis=0)
sub=pd.DataFrame({'instance_id':test_data['instance_id']})
sub['predicted_score']=oof_test
sub.to_csv("./final_submit/feature_test_score_%s.csv"%loss,index=False)


oof_test=oof_test_skf.mean(axis=0)
#保存train_prob
train_prob=pd.DataFrame(oof_train)
train_prob.columns=['click']
train_prob.to_csv("./stacking/lgb_train_prob_cv%f_%d_dim.csv"%(logloss,train_x.shape[1]),index=False)

test_prob=pd.DataFrame(oof_test)
test_prob.columns=['click']
test_prob.to_csv("./stacking/lgb_test_prob_cv%f_%d_dim.csv"%(logloss,train_x.shape[1]),index=False)
