from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor


def linear1():
    # 正规方程算法
    # 加载数据
    boston = load_boston()

    # 划分数据集并标准化
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 估计器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 得出摸型
    print("正规方程权重系数为：", estimator.coef_)
    print("正规方程偏置为：", estimator.intercept_)


def linear2():
    # 梯度下降算法
    # 加载数据
    boston = load_boston()

    # 划分数据集并标准化
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 估计器
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)

    # 得出摸型
    print("梯度下降权重系数为：", estimator.coef_)
    print("梯度下降偏置为：", estimator.intercept_)


if __name__ == "__main__":
    linear1()
    linear2()