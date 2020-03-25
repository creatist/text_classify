# coding=utf-8
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.model_selection import train_test_split
 
if __name__ == '__main__':
    train_data = pd.read_csv('../data/data_preproced.csv')

    label_map = {'PAYOUT': 0, 'INCOME': 1, 'IGNORE': 2, 'STATUS': 3, 
            'BILLREMIND': 4, 'OVERDUES': 5, 'LOAN': 6, 'STAGING': 7}
    label_map_rev = {v:k for k,v in label_map.items()}

    train_data["label"] =  train_data["label"].map(label_map)
    cw = lambda x: int(x)
    train_data['labels']=train_data['label'].apply(cw)
 
    x_train, x_test, y_train, y_test = train_test_split(train_data['text_splited'], train_data['label'], test_size=0.2)
 
    # 将语料转化为词袋向量，根据词袋向量统计TF-IDF
    vectorizer = CountVectorizer(max_features=5000)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_train))
    x_train_weight = tf_idf.toarray()  # 训练集TF-IDF权重矩阵
    tf_idf = tf_idf_transformer.transform(vectorizer.transform(x_test))
    x_test_weight = tf_idf.toarray()  # 测试集TF-IDF权重矩阵


    # model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, silent=True, objective='multi:softmax')
    # model.fit(x_train_weight, y_train)
    # y_predict=model.predict(x_test_weight)

    # 将数据转化为DMatrix类型
    dtrain = xgb.DMatrix(x_train_weight, label=y_train)
    dtest = xgb.DMatrix(x_test_weight, label=y_test)

    param = {'silent': 0, 'eta': 0.3, 'max_depth': 6, 'objective': 'multi:softmax', 'num_class': 8, 'eval_metric': 'merror'}  # 参数
    evallist = [(dtrain, 'train'), (dtest, 'test')]
    num_round = 100  # 循环次数
    xgb_model = xgb.train(param, dtrain, num_round,evallist)

    y_predict = xgb_model.predict(dtest)  # 模型预测
    label_all = label_all = list(label_map.keys())
    confusion_mat = metrics.confusion_matrix(y_test, y_predict)
    df = pd.DataFrame(confusion_mat, columns=label_all)
    df.index = label_all
    print('准确率：', metrics.accuracy_score(y_test, y_predict))
    print('confusion_matrix:', df)
    print('分类报告:', metrics.classification_report(y_test, y_predict))

