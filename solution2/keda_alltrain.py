# -*- encoding:utf-8 -*-
# Author:ZJH

import numpy as np 
import pandas as pd
import time
import datetime
import gc
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# 加载数据
path = "./Input/"
train = pd.read_table(path + 'round2_iflyad_train.txt')
test = pd.read_table(path + 'round2_iflyad_test_feature.txt')
train1 = pd.read_table(path + 'round1_iflyad_train.txt')
train= pd.concat([train, train1], axis=0, ignore_index=True)
train_result_df = train[["instance_id"]] 
# 合并训练集，验证集
# data = pd.concat([train,test],axis=0,ignore_index=True)


# 合并训练集，验证集
data = pd.concat([train,test],axis=0,ignore_index=True)
# 缺失值填充
data['make'] = data['make'].fillna(str(-1))
data['model'] = data['model'].fillna(str(-1))
data['osv'] = data['osv'].fillna(str(-1))
data['app_cate_id'] = data['app_cate_id'].fillna(-1)
data['app_id'] = data['app_id'].fillna(-1)
data['click'] = data['click'].fillna(-1)
data['user_tags'] = data['user_tags'].fillna(str(-1))
data['f_channel'] = data['f_channel'].fillna(str(-1))
# replace
replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink', 'app_paid']
for feat in replace:
    data[feat] = data[feat].replace([False, True], [0, 1])
# labelencoder 转化
encoder = ['city', 'province', 'make', 'model', 'osv', 'os_name', 'adid', 'advert_id', 'orderid',
           'advert_industry_inner', 'campaign_id', 'creative_id', 'app_cate_id',
           'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'creative_tp_dnf']

col_encoder = LabelEncoder()
for feat in encoder:
    col_encoder.fit(data[feat])
    data[feat] = col_encoder.transform(data[feat])

data['day'] = data['time'].apply(lambda x : int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x : int(time.strftime("%H", time.localtime(x))))

# 历史点击率
# 时间转换
data['period'] = data['day']
data['period'][data['period']<27] = data['period'][data['period']<27] + 31

for feat_1 in ['advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    gc.collect()
    res=pd.DataFrame()
    temp=data[[feat_1,'period','click']]
    for period in range(27,35):
        if period == 27:
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
        else: 
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_1')
        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)
        count[feat_1+'_rate'] = round(count[feat_1+'_1'] / count[feat_1+'_all'], 5)
        count['period']=period
        count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    print(feat_1,' over')
    data = pd.merge(data,res, how='left', on=[feat_1,'period'])

# 删除没用的特征
drop = ['click', 'time', 'instance_id', 'user_tags', 
        'app_paid', 'creative_is_js', 'creative_is_voicead']

train = data[:train.shape[0]]
test = data[train.shape[0]:]

y_train = train.loc[:,'click']
res = test.loc[:, ['instance_id']]

train.drop(drop, axis=1, inplace=True)
print('train:',train.shape)
test.drop(drop, axis=1, inplace=True)
print('test:',test.shape)

X_loc_train = train.values
y_loc_train = y_train.values
X_loc_test = test.values

out_of_fold_predictions = np.zeros((train.shape[0], 1))
# 模型部分
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=4000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, useed=1000, n_jobs=10, silent=True)

# 五折交叉训练，构造五个模型
skf=list(StratifiedKFold(y_loc_train, n_folds=5, shuffle=True, random_state=2018))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    lgb_model = model.fit(X_loc_train[train_index], y_loc_train[train_index],
                          eval_names =['train','valid'],
                          eval_metric='logloss',
                          eval_set=[(X_loc_train[train_index], y_loc_train[train_index]), 
                                    (X_loc_train[test_index], y_loc_train[test_index])],early_stopping_rounds=500)
    baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
    loss += lgb_model.best_score_['valid']['binary_logloss']
    test_pred= lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]
    vali_pred = lgb_model.predict_proba(X_loc_train[test_index], num_iteration=lgb_model.best_iteration_)[:, 1]
    out_of_fold_predictions[test_index, 0] = vali_pred
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred
print('logloss:', baseloss, loss/5)

# 加权平均
res['predicted_score'] = 0
for i in range(5):
    res['predicted_score'] += res['prob_%s' % str(i)]
res['predicted_score'] = res['predicted_score']/5

# 提交结果
mean = res['predicted_score'].mean()
print('mean:',mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
res[['instance_id', 'predicted_score']].to_csv("./final_submit/keda_alltrain_%f_%s.csv" % (loss/5, now), index=False)

# train_result_df["predict"] = out_of_fold_predictions[:, 0]
# train_result_df[['instance_id', 'predict']].to_csv("alltrain_keda_%s.csv" % now, index=False)

#保存train_prob
train_prob=pd.DataFrame(out_of_fold_predictions[:, 0])
train_prob.columns=['click']
train_prob.to_csv("./stacking/kedaalltrain_cv%f.csv"%(loss/5),index=False)

test_prob=pd.DataFrame(res['predicted_score'])
test_prob.columns=['click']
test_prob.to_csv("./stacking/kedaalltest_cv%f.csv"%(loss/5),index=False)