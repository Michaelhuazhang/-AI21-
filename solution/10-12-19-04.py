#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/7 9:57 AM
# @Author  : Jame
# @Site    : 
# @File    : bs_cyy.py
# @Software: PyCharm


import datetime
import gc
import warnings

import lightgbm as lgb
from scipy import sparse
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")
import time
import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

import joblib
from sklearn.model_selection import train_test_split
import re

######会改的地方####
###特征选择的比例也可以改
#改了多个参数的话，或者同时修改了两个地方，则base_score也要修改
base_score = 0.4141094904242171
generate = 1
gpu=0
n_splits = 5
msg = "cyy，初赛复赛数据集，osv，64，删除频次，不对频次进行Onehot"
###数据保存禁用了，因为不断试特征
prefix = "cyy_bs"
params={
    'boosting_type':'gbdt',
    'num_leaves':32,
    'reg_alpha':3,
    'reg_lambda':8,
    'max_depth':-1,
    'n_estimators':10000,
    'objective':'binary',
    'subsample':0.7,
    'colsample_bytree':0.8,
    'subsample_freq':1,
    'learning_rate':0.05,
    'random_state':2018,
    'n_jobs':-1
}
print(msg)
if gpu:
    params['device'] = 'gpu'

#######
ad_cate_feature = ['adid', 'advert_id', 'orderid',
                   'advert_industry_inner', 'advert_name','campaign_id', 'creative_id', 'creative_type',
                   'creative_tp_dnf', 'creative_has_deeplink','creative_is_jump', 'creative_is_download']
media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

sin_cat_features = ad_cate_feature + media_cate_feature + content_cate_feature
mul_cat_features = ['user_tags']
num_features = ['creative_width', 'creative_height']
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
if generate:
    path = './Input'
    train = pd.read_table(path + '/round2_iflyad_train.txt')
    round1_train = pd.read_table(path + '/round1_iflyad_train.txt')
    train = pd.concat([train, round1_train], axis=0, ignore_index=True)
    ####删除重复的7361条记录
    # train.drop_duplicates('instance_id', inplace=True)
    del round1_train
    gc.collect()
    test = pd.read_table(path + '/round2_iflyad_test_feature.txt')
    # ###代码测试，取小样本进行测试
    # train = train.sample(n=100000, random_state=2018)
    # test = test.sample(n=10000,random_state=2018)

    data = pd.concat([train, test], axis=0, ignore_index=True)
    print("原始数据读取完成")
    ###无用特征删除
    data.drop('creative_is_voicead',axis=1,inplace = True)
    data.drop('app_paid',axis=1,inplace=True)
    data.drop('creative_is_js',axis=1,inplace=True)
    data.drop('os_name',axis=1,inplace=True)
    print("无用特征删除完成")
    #数据缺失值填充
    data[data.select_dtypes('object').columns.tolist()] = data.select_dtypes('object').fillna("-999")
    data.fillna(-999, inplace=True)

    ###bool类型数据转换
    bool_feature = ['creative_is_jump', 'creative_is_download','creative_has_deeplink']
    for i in bool_feature:
        data[i] = data[i].astype(int)

    #####还没测试，
    ### 测试下1、make和model的清洗,标准化，以及2、model缺失值填充是否有作用
    ###1、标注化，apple，Apple，iPhone7,iPhone8,iPhon9等都改为apple
    ###
    ##字符串特征标准化
    #需在na填充之前，否者会报错
    # for fe in ['model', 'make']:
    #     data[fe] = data[fe].apply(lambda x: str(x).lower() if x !=np.nan else x)
    #
    # ##中间这一段单独测试，然后在填充
    # ##2、model缺失填充
    # index = data[data.make == '-999'].index
    # data.loc[index, 'make'] = data[data.make == '-999'].model
    # index = data[data.make == '-999'].index
    # data.loc[index, 'make'] = data[data.make == '-999'].os.astype(str)
    #
    # data.make = data.make.apply(lambda x: "apple" if x == 1 else x)


    # data.make = data.make.apply(lambda x:
    #                 "apple" if re.search(r'iphone*?', x) else
    #                 "vivo" if re.search(r'vivo*?', x) else
    #                 "oppo" if re.search(r'oppo*?', x) else
    #                 "huawei" if re.search(r'huawei*?', x) else
    #                 "xiaomi" if re.search(r'redmi*?', x) else
    #                 "xiaomi" if re.search(r'xiaomi*?', x) else
    #                 "ipad" if re.search(r'ipad*?', x) else
    #                 "ipad" if re.search(r'ipad*?', x) else
    #                 "xiaomi" if "mi" in x else
    #                 "apple" if x == '1' else x)

    ##还有剩下的不知道要不要修改，修改之后粒度变小了
    # map_dic = {
    #     "m5 note": "meizu",
    #     "mha-al00": "huawei",
    #     "gn5001s": "gionee",
    #     "bln-al10": "huawei"
    # }
    # print("make清洗和填充完成")



    #####特征添加创建
    data['area'] = data['creative_height'] * data['creative_width']
    data['os_osv'] = data['os'].astype(str).values + '_' + data['osv'].astype(str).values
    data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
    data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
    data['label'] = data.click.astype(int)
    del data['click']
    num_features.extend(['day', 'hour', 'area']) #线上具有提升
    sin_cat_features.extend(['os_osv'])
    print('特征day, hour,area添加完成')


    data['advert_industry_inner_first'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[0])
    data['advert_industry_inner_second'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[1])
    sin_cat_features.extend(['advert_industry_inner_first', 'advert_industry_inner_second'])
    print('特征first, second加完成')


    ####cyy特征
    #处理user_tags，添加user_tags_len
    data['user_tags']=data['user_tags'].apply(lambda x: x[1:] if x.startswith(',') else x)
    data['user_tags_len'] = data['user_tags'].apply(lambda x: len(x.split(',')))
    num_features.extend(['user_tags_len'])
    print('添加user_tags_len完成')

    data['model_same']= data.model.apply(lambda x:
                         '+' if '+' in str(x) else\
                         '-' if '-' in str(x) else \
                         '_' if '_' in str(x) else \
                         ',' if ',' in str(x) else \
                         '%2b' if '%2b' in str(x) else \
                         '%20' if '%20' in str(x) else \
                         '%2522' if '%2522' in str(x) else \
                         '%25' if '%25' in str(x) else \
                         'other')
    sin_cat_features.extend(['model_same'])
    print('特征model_same添加完成')


    wordfreq_feature_cate_list = []
    def user_tags_freq(data):
        data['user_tags']=data['user_tags'].apply(lambda x: x.split(','))
        apps=data['user_tags'].apply(lambda x:' '.join(x)).tolist()
        vectorizer=CountVectorizer()
        cntTf = vectorizer.fit_transform(apps)
        word=vectorizer.get_feature_names()
        weight=cntTf.toarray()
        df_weight=pd.DataFrame(weight)
        df_weight.columns=word
        temp_df= pd.DataFrame(df_weight.sum()).sort_values(by=[0],ascending=False).reset_index().\
            rename(columns={0: 'user_tags_freq'})
        freq2usertags = temp_df[temp_df['user_tags_freq'] > 40000]['index']
        for i in tqdm(freq2usertags):
            data['is_'+str(i)]=data['user_tags'].apply(lambda x: 1 if i in x else 0)
            new_feature='is_'+str(i)
            wordfreq_feature_cate_list.append(new_feature)
        return data

    data = user_tags_freq(data)
    print("对user_tags_freq处理完成")
    sin_cat_features.extend(wordfreq_feature_cate_list)


    # # num_features.extend(wordfreq_feature_cnt_list)

    #构造广告id转化数,adid_click_count线上降分，必须在线下验证码及划分之后
    # fe = 'osv'
    # old_fe = fe
    # new_fe = '%s_click_count'%old_fe
    # adid_count = pd.DataFrame(train.groupby([old_fe,'click']).size()).reset_index().rename(columns={0: new_fe})
    # adid_count = adid_count[adid_count.click == 1]
    # adid_count.drop('click', axis=1, inplace=True)
    # data = pd.merge(data, adid_count,on=old_fe, how='left')
    # data[new_fe].fillna(0, inplace=True)
    # num_features.extend([new_fe])

    print("LabelEncoder开始：")
    for i in tqdm(sin_cat_features):
        data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    feature = sin_cat_features + num_features
    print("特征数量：%d, 特征:%s" % (len(feature), feature))
    predict = data[data.label == -999]
    predict_result = predict[['instance_id']]
    predict_result['predicted_score'] = 0
    predict_x = predict.drop('label', axis=1)
    train_x = data[data.label != -999]
    train_y = train_x.label.values

    base_train_csr = sparse.csr_matrix((len(train_x), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))
    enc = OneHotEncoder()
    print("OnehotEncoder开始：")
    for feature in tqdm(sin_cat_features):
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr',
                                         'bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=20)
    print("CV开始：")
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')

    train_csr = sparse.hstack(
        (sparse.csr_matrix(train_x[num_features]), base_train_csr), 'csr').astype(
        'float32')
    predict_csr = sparse.hstack(
        (sparse.csr_matrix(predict_x[num_features]), base_predict_csr), 'csr').astype('float32')
    print(train_csr.shape)

    feature_select = SelectPercentile(chi2, percentile=95)
    feature_select.fit(train_csr, train_y)
    train_csr = feature_select.transform(train_csr)
    predict_csr = feature_select.transform(predict_csr)
    print('卡方选择后大小：%s' % str(train_csr.shape))

    # joblib.dump((train_csr, train_y, predict_csr, predict_result), "./data/feature/cyy_data_%s.pkl" % now)
else:
    subfix = ""
    train_csr, train_y, predict_csr, predict_result, predict_off_csr, predict_off_y = joblib.load("./data/feature/cyy_data_%s.pkl" % subfix )
    print("数据没变有变化，直接读取完成！")

fold_name = "%s_%s" % (prefix, now)
submission_filename = "%s.csv" % fold_name
model_name = "%s.model" % fold_name
print("n_split:%d 开始训练"%n_splits)
best_score = []
start_time = time.time()
lgb_model = lgb.LGBMClassifier(**params)
if n_splits == 1:
    train_x_csr, valid_x_csr, train_y_csr, valid_y_csr = train_test_split(train_csr, train_y, test_size=0.2, random_state=2018)
    lgb_model.fit(train_x_csr, train_y_csr,
                  eval_set=[(train_x_csr, train_y_csr),
                            (valid_x_csr, valid_y_csr)],
                  eval_names=["train", "validation", "validation2"],
                  early_stopping_rounds=100)
    best_score.append(lgb_model.best_score_['validation']['binary_logloss'])
    print("验证集：%s"% str(best_score))
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
else:
    skf = StratifiedKFold(n_splits= n_splits, random_state=2018, shuffle=True)
    out_of_fold_predictions = np.zeros((train_csr.shape[0], 1))
    for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
        print("第%d折开始训练"%(index+1))
        lgb_model.fit(train_csr[train_index], train_y[train_index],
                      eval_set=[(train_csr[train_index], train_y[train_index]),
                                (train_csr[test_index], train_y[test_index])],
                      eval_names=["train", "validation"],
                      early_stopping_rounds=100)
        best_score.append(lgb_model.best_score_['validation']['binary_logloss'])
        print("验证集：%s" % str(best_score))
        test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
	# 保存验证集预测结果
        vali_pred = lgb_model.predict_proba(train_csr[test_index], num_iteration=lgb_model.best_iteration_)[:, 1]
        out_of_fold_predictions[test_index, 0] = vali_pred
        print('test mean:', test_pred.mean())
        predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
os.makedirs('./history/%s' % fold_name)
print(np.mean(best_score))
predict_result['predicted_score'] = predict_result['predicted_score'] / n_splits
mean = predict_result['predicted_score'].mean()
print('mean:', mean)
end_time = time.time()
seconds= end_time - start_time
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print("运行用时:%d:%d:%d"% (h,m,s))
joblib.dump(lgb_model, "./history/%s/%s" % (fold_name, model_name))
predict_result[['instance_id', 'predicted_score']].to_csv("./history/%s/%s" % (fold_name,submission_filename) , index=False)
if np.mean(best_score) > base_score:
    status = "\t下降"
else:
    status = "\t有提升"
msg = msg + status
# log = Log()
# log.log(fold_name, best_score, best_score2, lgb_model.best_iteration_, os.path.basename(sys.argv[0]),
#         submission_filename,msg=msg, log_file='./logs/tracks.log')
train_result_df = pd.DataFrame(out_of_fold_predictions[:, 0])
train_result_df.to_csv("./stacking/team1_train2_%f.csv" % (np.mean(best_score)), index=False)
test_prob=pd.DataFrame(predict_result['predicted_score'])
test_prob.columns=['click']
test_prob.to_csv("./stacking/team12cv%f.csv"%(np.mean(best_score)),index=False)
