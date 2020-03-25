# coding=utf-8
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
 
if __name__ == '__main__':
    train_data = pd.read_csv('../data/data_preproced.csv')

    label_map = {'PAYOUT': 0, 'INCOME': 1, 'IGNORE': 2, 'STATUS': 3, 
            'BILLREMIND': 4, 'OVERDUES': 5, 'LOAN': 6, 'STAGING': 7}
    label_map_rev = {v:k for k,v in label_map.items()}

    train_data["label"] =  train_data["label"].map(label_map)
    x_train, x_test, y_train, y_test = train_test_split(train_data['text_splited'], train_data['label'], test_size=0.2)
    cw = lambda x: int(x)
    x_train = x_train
    y_train = np.array(y_train.apply(cw))
    x_test = x_test
    y_test = np.array(y_test.apply(cw))
 
    # 将语料转化为词袋向量，根据词袋向量统计TF-IDF
    vectorizer = CountVectorizer(max_features=5000)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_train))
    x_train_weight = tf_idf.toarray()  # 训练集TF-IDF权重矩阵
    tf_idf = tf_idf_transformer.transform(vectorizer.transform(x_test))
    x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵
 
    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(x_train_weight, y_train)
    lgb_val = lgb.Dataset(x_test_weight, y_test, reference=lgb_train)
 
    # 构建lightGBM模型
    params = {'max_depth': 5, 'min_data_in_leaf': 20, 'num_leaves': 35,
              'learning_rate': 0.1, 'lambda_l1': 0.1, 'lambda_l2': 0.2,
              'objective': 'multiclass', 'num_class': 8, 'verbose': -1}
    # 设置迭代次数，默认为100，通常设置为100+
    num_boost_round = 1000
    # 训练 lightGBM模型
    gbm = lgb.train(params, lgb_train, num_boost_round, verbose_eval=100, valid_sets=lgb_val)
 
    # 保存模型到文件
    # gbm.save_model('data/lightGBM_model')
 
    # 预测数据集
    y_pred = gbm.predict(x_test_weight, num_iteration=gbm.best_iteration)
 
    y_predict = np.argmax(y_pred, axis=1)  # 获得最大概率对应的标签
 
    label_all = list(label_map.keys())
    confusion_mat = metrics.confusion_matrix(y_test, y_predict)
    df = pd.DataFrame(confusion_mat, columns=label_all)
    df.index = label_all
 
    print('准确率：', metrics.accuracy_score(y_test, y_predict))
    print('confusion_matrix:', df)
    print('分类报告:', metrics.classification_report(y_test, y_predict))