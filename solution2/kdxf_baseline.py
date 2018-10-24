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



warnings.filterwarnings("ignore")

# 加载数据
path = './Input/'

train = pd.read_table(path + 'round2_iflyad_train.txt')
test = pd.read_table(path + 'round2_iflyad_test_feature.txt')
data = pd.concat([train, test], axis=0, ignore_index=True)
train_result_df = train[["instance_id"]] 
data = data.fillna(-1)

data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))
data['label'] = data.click.astype(int)
del data['click']

bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                'creative_has_deeplink', 'app_paid']
for i in bool_feature:
    data[i] = data[i].astype(int)

data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])

ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download']

media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature

for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))

cate_feature = origin_cate_list

num_feature = ['creative_width', 'creative_height', 'hour']

feature = cate_feature + num_feature
print(len(feature), feature)

predict = data[data.label == -1]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0
predict_x = predict.drop('label', axis=1)

train_x = data[data.label != -1]
train_y = data[data.label != -1].label.values

# 默认加载 如果 增加了cate类别特征 请改成false重新生成
if os.path.exists('base_train_csr.npz') and True:
    print('load_csr---------')
    base_train_csr = sparse.load_npz('base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz('base_predict_csr.npz').tocsr().astype('bool')
else:
    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr',
                                         'bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=20)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')

    sparse.save_npz('base_train_csr.npz', base_train_csr)
    sparse.save_npz('base_predict_csr.npz', base_predict_csr)

train_csr = sparse.hstack(
    (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
print(train_csr.shape)
feature_select = SelectPercentile(chi2, percentile=95)
feature_select.fit(train_csr, train_y)
train_csr = feature_select.transform(train_csr)
predict_csr = feature_select.transform(predict_csr)
print('feature select')
print(train_csr.shape)

lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=32, reg_alpha=0, reg_lambda=0.1,
    max_depth=-1, n_estimators=5000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, random_state=2018, n_jobs=-1
)
out_of_fold_predictions = np.zeros((train_csr.shape[0], 1))
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    lgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])], early_stopping_rounds=300)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    vali_pred = lgb_model.predict_proba(train_csr[test_index], num_iteration=lgb_model.best_iteration_)[:, 1]
    out_of_fold_predictions[test_index, 0] = vali_pred
    print('test mean:', test_pred.mean())
    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred


print(np.mean(best_score))
predict_result['predicted_score'] = predict_result['predicted_score'] / 5
mean = predict_result['predicted_score'].mean()
print('mean:', mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['instance_id', 'predicted_score']].to_csv("./final_submit/kdxf_%f_%s.csv" % (np.mean(best_score), now), index=False)


train_result_df["click"] = out_of_fold_predictions[:, 0]
train_result_df[[ 'click']].to_csv("./stacking/kdxf_train_%f_%s.csv" % (np.mean(best_score), now), index=False)
test_prob=pd.DataFrame(predict_result['predicted_score'])
test_prob.columns=['click']
test_prob.to_csv("./stacking/kdxf_cv%f.csv"%(np.mean(best_score)),index=False)
