from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
import pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error, silhouette_score
import joblib
from sklearn.cluster import KMeans

def get_dataset():
    # 加载数据集
    iris = load_iris()
    print("数据集：", iris)
    print("数据集：", iris["DESCR"])
    print("描述：", iris.DESCR)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

def dict():
    # 字典特征抽取
    data = {"小张": "18","小李": "19", "小王": "20"}
    transfer = DictVectorizer(sparse=False)
    data = transfer.fit_transform(data)
    print(data)
    print("特征名：", transfer.get_feature_names_out())


def text_count():
    # 文本处理
    data = ["This is a test sentence", "This is the second sentence to tset"]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)

    print("分类结果：\n", data_new.toarray())
    print("特征类型：\n", transfer.get_feature_names_out())

    return None


def text_chinese_count():
    # 中文文本处理
    data = ["这是第一句测试语句", "这是第二句测试语句"]

    def cut_word(text):
        new_text = " ".join(list(jieba.cut(text)))
        return new_text

    data_new = []
    for sentence in data:
        data_new.append(cut_word(sentence))

    transfer = CountVectorizer()
    # transfer = CountVectorizer(stop_words = ["这是"])
    data_new = transfer.fit_transform(data_new)

    print("分类结果：\n", data_new.toarray())
    print("特征类型：\n", transfer.get_feature_names_out())

    return None


def tfidf_test():
    #使用TFIDF方法进行文本特征抽取
    data = ["这是第一句测试语句", "这是第二句测试语句"]

    def cut_word(text):
        new_text = " ".join(list(jieba.cut(text)))
        return new_text

    data_new = []
    for sentence in data:
        data_new.append(cut_word(sentence))

    transfer = TfidfVectorizer()
    # transfer = CountVectorizer(stop_words = ["这是"])
    data_new = transfer.fit_transform(data_new)

    print("分类结果：\n", data_new.toarray())
    print("特征类型：\n", transfer.get_feature_names_out())

    return None


def preprocess_test():
    # 数据预处理（归一化）
    data = pandas.read_csv("data.txt")
    transfer = MinMaxScaler(feature_range=[0, 1])
    data_new = transfer.fit_transform(data)
    print(data_new)

    # 数据预处理（标准化）
    # 优点：可以排除某些异常值的影响
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)

    return None


def feature_select():
    # 进行特征选择
    # 过滤低方差的特征
    data = pandas.read_csv("data.txt")
    transfer = VarianceThreshold()
    data_new = transfer.fit_transform(data)


def knn():
    # KNN（K近邻），依据邻居来推断本身的类别
    # 优点：简单易实现，无需训练；缺点：计算量大，K值选择对结果影响大。
    iris = load_iris()
    # 划分数据集并标准化
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)

    # 模型评估
    # 方法1：对比真实值和预测值
    y_predict = knn.predict(x_test)
    print("y_predict：", y_predict)
    print("预测是否正确：", y_test == y_predict)
    # 方法2：计算准确率
    score = knn.score(x_test, y_test)
    print("准确率：", score)

    # 如果要给K值选择增加网格搜索和交叉验证
    knn = KNeighborsClassifier()
    knn = GridSearchCV(knn, param_grid={"n_neighbors": [1, 2, 3, 4, 5]}, cv=10)   # 10重交叉验证
    knn.fit(x_train, y_train)

    print("最佳参数:n", knn.best_params_)
    print("最佳结果:n", knn.best_score_)
    print("最佳估计器:", knn.best_estimator_)
    print("交叉验证结果:", knn.cv_results_)


def decision_tree():
    # 可视化，解释能力强；容易过拟合。
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)
    # 模型评估
    # 方法1:直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: n", y_predict)
    print("直接比对真实值和预测值:", y_test == y_predict)
    # 方法2:计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:n", score)
    # 可视化决策树
    export_graphviz(estimator, out_file="decision_tree.dot", feature_names=iris.feature_names)

    return None


def random_forest():
    # 随机森林：多个决策树组成
    # 随机：训练集随机；特征随机。
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    estimator = RandomForestClassifier()

    # 增加网格搜索和交叉验证
    # n_estimators：森林里树木数量；max_depth：树的最大深度
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param, cv=4)   # 4重交叉验证
    estimator.fit(x_train, y_train)

    # 模型评估
    # 方法1:直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: n", y_predict)
    print("直接比对真实值和预测值:", y_test == y_predict)
    # 方法2:计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为:n", score)

    print("最佳参数:n", estimator.best_params_)
    print("最佳结果:n", estimator.best_score_)
    print("最佳估计器:", estimator.best_estimator_)
    print("交叉验证结果:", estimator.cv_results_)


def gradient_descent():
    iris = load_iris()
    print("特征数量：", iris.data.shape)

    # 划分数据集并标准化
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 估计器；learning_rate：学习率；eta0：学习率；max_iter：迭代次数；
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)

    # 得出摸型
    print("梯度下降权重系数为：", estimator.coef_)
    print("梯度下降偏置为：", estimator.intercept_)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测结果：", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差：", error)


def ridge():
    # 正则化：排除某些异常特征/异常点的影响，简化模型，解决过拟合问题
    # L1正则化：Lasso，删除某些特征的影响
    # L2正则化：Ridge，削弱某些特征的影响，更常用
    # 岭回归 = 线性回归 + L2正则化，是线性回归的改进
    iris = load_iris()
    print("特征数量：", iris.data.shape)

    # 划分数据集并标准化
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = Ridge()
    estimator.fit(x_train, y_train)

    # 得出摸型
    print("岭回归权重系数为：", estimator.coef_)
    print("岭回归偏置为：", estimator.intercept_)

    # 保存模型
    joblib.dump(estimator, "ridge.pkl")

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测结果：", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("均方误差：", error)


def model_save():
    # sklearn模型保存和加载
    estimator = Ridge()
    # 保存
    joblib.dump(estimator, "ridge.pkl")
    # 加载
    estimator = joblib.load("ridge.pkl")


def kmeans():
    # 无监督学习：没有目标值
    # 无监督学习：聚类(Kmeans))/降维(PCA)
    data = 1

    estimator = KMeans(n_clusters=3)    # 初始聚类中心的数量
    estimator.fit(data)

    y_predict = estimator.predict(data)
    error = silhouette_score(data, y_predict)



if __name__ == "__main__":
    gradient_descent()
