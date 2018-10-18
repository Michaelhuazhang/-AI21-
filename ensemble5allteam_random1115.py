# -*- encoding:utf-8  -*-
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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

# 加载数据
path = '../Input/'
train = pd.read_table(path + 'round2_iflyad_train.txt')
test = pd.read_table(path + 'round2_iflyad_test_feature.txt')
train1 = pd.read_csv("eda_base2train_10-09-20-35.csv")
train1_y = pd.read_csv("edabase2_10-09-20-35.csv")

train2 = pd.read_csv("train_keda_10-10-15-38.csv")
train2_y = pd.read_csv("keda_10-10-15-38.csv")

train3 = pd.read_csv("lgb_train_prob_cv0.416329_18071_dim.csv")# click
train3_y = pd.read_csv("feature_test_score.csv")

train4 = pd.read_csv("kdxf_train_10-10-12-10.csv")
train4_y = pd.read_csv("kdxf_10-10-12-10.csv")

train5 = pd.read_csv("ensemble_train_10-11-22-47.csv")
train5_y = pd.read_csv("ensemble_submit_10-11-22-47.csv")

train6 = pd.read_csv("eda_base3alltrain_10-10-15-10.csv")
train6_y = pd.read_csv("edabase3alltrain_10-10-15-10.csv")
len_train = len(train)

train7_y = pd.read_csv("ensemble2_submit_10-12-16-21.csv")
train7 = pd.read_csv("ensemble2_train_10-12-16-21.csv")
filepath = '../stacking/'
train8 = pd.read_csv(filepath + "team1_train_0.414479.csv")
train8_y = pd.read_csv(filepath + "team1_cv0.414479.csv")


train9 = pd.read_csv(filepath + "team1_train2_0.414354.csv")
train9_y = pd.read_csv(filepath + "team12cv0.414354.csv")

train10 = pd.read_csv(filepath + "team1_train3_0.414253.csv")
train10_y = pd.read_csv(filepath + "team13_cv0.414253.csv")


train1["predict_2"] = train2.predict
train1["predict_3"] = train3.click
train1["predict_4"] = train4.predict
train1["predict_5"] = train5.predict
train1["predict_6"] = train6[:len_train].predict
train1["predict_7"] = train7.predict
train1["predict_8"] = train8.iloc[0]
train1["predict_9"] = train9.iloc[0]
train1["predict_10"] = train10.iloc[:, 0]

train1_y["predicted_score2"] = train2_y.predicted_score
train1_y["predicted_score3"] = train3_y.predicted_score
train1_y["predicted_score4"] = train4_y.predicted_score
train1_y["predicted_score5"] = train5_y.predicted_score
train1_y["predicted_score6"] = train6_y[:len_train].predicted_score
train1_y["predicted_score7"] = train7_y.predicted_score
train1_y["predicted_score8"] = train8_y.click
train1_y["predicted_score9"] = train9_y.click
train1_y["predicted_score10"] = train10_y.click
train["p1"] = train1.predict
train["p2"] = train1.predict_2
train["p3"] = train1.predict_3
train["p4"] = train1.predict_4
train["p5"] = train1.predict_5
train["p6"] = train1.predict_6
train["p7"] = train1.predict_7
train["p8"] = train1.predict_8
train["p9"] = train1.predict_9
train["p10"] = train1.predict_10

test["p1"] = train1_y.predicted_score
test["p2"] = train1_y.predicted_score2
test["p3"] = train1_y.predicted_score3
test["p4"] = train1_y.predicted_score4
test["p5"] = train1_y.predicted_score5
test["p6"] = train1_y.predicted_score6
test["p7"] = train1_y.predicted_score7
test["p8"] = train1_y.predicted_score8
test["p9"] = train1_y.predicted_score9
test["p10"] = train1_y.predicted_score10
train_result_df = train[["instance_id"]] 

train_ = train
# 合并训练集，验证集
train = pd.concat([train,test],axis=0,ignore_index=True)


drop_feats = ["creative_is_js", "creative_is_voicead", "app_paid", "f_channel"]
train = train.drop(drop_feats, axis=1)


# 缺失值填充
train['make'] = train['make'].fillna(str(-1))
train['model'] = train['model'].fillna(train.model.mode()[0])
train['osv'] = train['osv'].fillna(train.osv.mode()[0])
train['app_cate_id'] = train['app_cate_id'].fillna(train.app_cate_id.mode()[0])
train['app_id'] = train['app_id'].fillna(train.app_id.mode()[0])

train['user_tags'] = train['user_tags'].fillna(str(-1))

train["creative_area"] = train.creative_height * train.creative_width

train['day'] = train['time'].apply(lambda x : int(time.strftime("%d", time.localtime(x))))
train['hour'] = train['time'].apply(lambda x : int(time.strftime("%H", time.localtime(x))))
train["week"] = train["time"].apply(lambda x: int(time.strftime("%w", time.localtime(x))))
train['period'] = train['day']
train['period'][train['period']<27] = train['period'][train['period']<27] + 31
train.drop(["time", "day"], axis=1, inplace=True)
def maphour(x):
    if ((x>=0) & (x<=6)):
        return 0
    elif ((x>6) & (x<=9)):
        return 1
    elif ((x>9) & (x<=13)):
        return 2
    elif ((x>13) & (x<=16)):
        return 3
    elif ((x > 16) & (x <=19 )):
        return 4
    else:
        return 5

train.hour = train.hour.apply(lambda x: maphour(x))
train.week = train.week.apply(lambda x: 1 if (x == 0) | (x == 6) | (x == 5) else 0)




city_counts = train.city.value_counts()
adid =train.adid.value_counts()
orderid_counts = train.orderid.value_counts()# 10000
creadtive_id_counts = train.creative_id.value_counts()#10000
app_id_counts = train.app_id.value_counts() # 10000
def mapcity(x):
    if city_counts[x] >20000:
        return 3
    elif city_counts[x] >10000:
        return 2
    elif city_counts[x] > 1000:
        return 1
    else:
        return 0

train["city_count_range"] = train.city.apply(lambda x:mapcity(x))
train["adid_count_range"] = train.adid.apply(lambda x:1 if adid[x] > 2000 else 0)

train["orderid_counts_range"] = train.orderid.apply(lambda x:1 if orderid_counts[x] > 10000 else 0)
train["creadtive_id_count_range"] = train.creative_id.apply(lambda x:1 if creadtive_id_counts[x] > 10000 else 0)
train["app_id_counts_range"] = train.app_id.apply(lambda x :1 if app_id_counts[x] > 10000 else 0)
user_tags_counts = train.user_tags.value_counts()
make_counts = train.make.value_counts()
model_counts = train.model.value_counts()
inner_slot_id_counts = train.inner_slot_id.value_counts()
advert_name = train.advert_name.value_counts()
advert_industry_inner_counts = train.advert_industry_inner.value_counts()
osv_counts = train.osv.value_counts()
len_train = len(train)
train["make_freq"] = train.make.apply(lambda x: make_counts[x]/len_train)
train["model_freq"] = train.model.apply(lambda x: model_counts[x] / len_train)
train["inner_slot_id_freq"] = train.inner_slot_id.apply(lambda x : inner_slot_id_counts[x] / len_train)
train["adver_name_freq"] = train.advert_name.apply(lambda x : advert_name[x] / len_train)
train["osv_freq"] = train.osv.apply(lambda x: osv_counts[x] / len_train)
train["advert_industry_inner_freq"] = train.advert_industry_inner.apply(lambda x: advert_industry_inner_counts[x] / len_train)
train["user_tags_freq"] = train.user_tags.apply(lambda x:user_tags_counts[x] / len_train)

train.model = train.model.apply(lambda x: x if model_counts[x]>500 else "other")

def mapuser_tags(x):
    if user_tags_counts[x] >100000:
        return 6
    elif user_tags_counts[x] >10000:
        return 5
    elif user_tags_counts[x] > 6000:
        return 4
    elif user_tags_counts[x] > 2000:
        return 3
    elif user_tags_counts[x] > 1000:
        return 2
    elif user_tags_counts[x] > 500:
        return 1
    else:
        return 0

train["user_tags_range"] = train.user_tags.apply(lambda x: mapuser_tags(x))
train["user_tags_bool"] = train.user_tags.apply(lambda x: 1 if user_tags_counts[x] > 2000 else 0)
train["make_bool"] = train.make.apply(lambda x: 1 if make_counts[x] > 10000 else 0)



train['user_tags'] = train['user_tags'].apply(lambda x:x[1:] if x.startswith(',') else x)
train.user_tags = train.user_tags.apply(lambda x:x.split(','))
def gd_num(x):
    num=0
    for i in x:
        if 'gd' in i:
            num+=1
    return num
def ag_num(x):
    num=0
    for i in x:
        if 'ag' in i:
            num+=1
    return num

def num_2(x):
    num=0
    for i in x:
        if i.startswith("2"):
            num+=1
    return num
def num_3(x):
    num=0
    for i in x:
        if i.startswith("3"):
            num+=1
    return num

train['user_tags_ag_num']=train['user_tags'].apply(ag_num)
train['user_tags_gd_num']=train['user_tags'].apply(gd_num)
train['user_tags_2_num']=train['user_tags'].apply(num_2)
train['user_tags_3_num']=train['user_tags'].apply(num_3)

train['user_tags_ag_gd_num']=train['user_tags_ag_num']+train['user_tags_gd_num']
train['user_tags_2_3_num']=train['user_tags_2_num']+train['user_tags_3_num']
train["user_tags_len"] = train.user_tags.apply(lambda x: len(x))

user_tags = train['user_tags'].apply(lambda x:' '.join(x)).tolist()
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(user_tags))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word = vectorizer.get_feature_names()
#weight=tfidf.toarray()
df_weight=pd.DataFrame(tfidf.toarray())
feature=df_weight.columns
df_weight['sum']=0
for f in tqdm(feature):
    df_weight['sum']+=df_weight[f]
train['tfidf_sum'] = df_weight['sum']

train.osv = train.osv.apply(lambda x: x if osv_counts[x]> 1000 else "other")
osv_counts = train.osv.value_counts()
train["osv_range"] = train.osv.apply(lambda x: 1 if osv_counts[x] > 10000 else 0)

train.inner_slot_id = train.inner_slot_id.apply(lambda x: x if inner_slot_id_counts[x]> 1000 else "other")
inner_slot_id_counts = train.inner_slot_id.value_counts()
train["inner_slot_id_range"] = train.inner_slot_id.apply(lambda x: 1 if inner_slot_id_counts[x] > 100000 else 0)

train["advert_industry_inner_counts_range"] = train.advert_industry_inner.apply(lambda x : 1 if advert_industry_inner_counts[x]>100000 else 0)
train['advert_industry_inner_first'] = train['advert_industry_inner'].apply(lambda x:x.split('_')[0])
train['advert_industry_inner_second'] = train['advert_industry_inner'].apply(lambda x:x.split('_')[1])

columns = ["carrier", "devtype", "nnt", "os", "creative_type", "hour", "city_count_range","user_tags_range", "os_name"]
for col in columns:
    train[col] = train[col].astype("str")
suburb_dummies = pd.get_dummies(train[columns], drop_first=True)
train = train.drop(columns,axis=1).join(suburb_dummies)

cols = ['creative_is_jump', 
       'creative_is_download', 'creative_has_deeplink', 'city', 'province', 'make', 'model', 'osv',
       'adid', 'advert_id', 'orderid', 'advert_industry_inner', 'campaign_id',
       'creative_id', 'creative_tp_dnf', 'app_cate_id', 'app_id',
       'inner_slot_id', 'advert_name', 'advert_industry_inner_first',
       'advert_industry_inner_second']
for c in cols:
    lb = LabelEncoder()
    lb.fit(list(train[c].values))
    train[c] = lb.transform(list(train[c].values))


# 删除没用的特征
drop = ['click',  'instance_id', 'user_tags', 
        ]

train1 = train[:train_.shape[0]]
test1 = train[train_.shape[0]:]

y_train = train1.loc[:,'click']
res = test1.loc[:, ['instance_id']]

train1.drop(drop, axis=1, inplace=True)
print('train:',train1.shape)
test1.drop(drop, axis=1, inplace=True)
print('test:',test1.shape)

X_loc_train = train1.values
y_loc_train = y_train.values
X_loc_test = test1.values

##############################

out_of_fold_predictions = np.zeros((train_.shape[0], 1))

# 模型部分
model = lgb.LGBMClassifier(boosting_type='gbdt',num_leaves=64,reg_alpha=3,reg_lambda=1,max_depth=-1,n_estimators=10000,objective='binary',subsample=1,colsample_bytree=0.8,subsample_freq=1,learning_rate=0.05,random_state=2018,n_jobs=-1)

# 五折交叉训练，构造五个模型
skf=list(StratifiedKFold(y_loc_train, n_folds=5, shuffle=True, random_state=1115))
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
    test_pred = lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]
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
res[['instance_id', 'predicted_score']].to_csv("ensemble4allteamrandom115_%f_%s.csv" %(loss/5, now), index=False)

train_result_df["predict"] = out_of_fold_predictions[:, 0]
train_result_df[['instance_id', 'predict']].to_csv("ensemble4allteamtrainrandom1115_%s.csv" % now, index=False)

