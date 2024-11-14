from django.shortcuts import render
import csv
from django.shortcuts import render
from django.http import HttpResponse
from django import forms
from django.core.files.storage import FileSystemStorage
import pandas as pd
import io
from sklearn.metrics import accuracy_score,classification_report

class CSVUploadForm(forms.Form):
    csv_file=forms.FileField()


# Create your views here.

def upload(request):
    if request.method=='POST' :
        form=CSVUploadForm(request.POST,request.FILES)
        if form.is_valid():
            csv_file=request.FILES['csv_file']
            fs=FileSystemStorage()
            filename=fs.save(csv_file.name,csv_file)
            file_path=fs.path(filename)

            new_df=handle_csv_file(file_path)
            new_df1=handle_csv_file1(file_path)
            html_table=new_df.to_html(classes='table table-striped')
            html_table1=new_df1.to_html(classes='table table-striped')
            return render(request,'upload.html',{'form':form,'html_table':html_table,'html_table1':html_table1})
    else:
        form=CSVUploadForm()
    return render(request,'upload.html',{'form':form})

def handle_csv_file(csv_file):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    from django.conf import settings
    import os
    static_dir = settings.STATICFILES_DIRS[0]

    # 构造 CSV 文件的完整路径
    csv_file_path = os.path.join(static_dir, '2.csv')

    # 确保文件存在
    if os.path.isfile(csv_file_path):
        # 使用 pandas 读取 CSV 文件
        import pandas as pd
        df = pd.read_csv(csv_file_path)
        # ... 你的数据处理逻辑 ...
    else:
        raise FileNotFoundError("CSV file not found in the static directory")


    # 选择数值特征和分类特征
    numerical_features = [
         '待还本金', '待还利息',

    ]

    categorical_features = [
        '是否首标',
         '学历认证', '征信认证'
    ]

    # 目标变量映射
    # 将"已逾期"映射为1，"已还清"映射为0
    # 这里需要确保只选择状态为2和3的数据
    df['target'] = df['标当前状态'].map({2: 0, 3: 1})
    df_target = df[df['target'].isin([0, 1])]

    # 确保X包含了数值特征和分类特征
    X = df_target[numerical_features + categorical_features]
    y = df_target['target']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)  # sparse=False 从 scikit-learn 1.0 开始默认为False
        ])

    # 创建模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 评估模型
    print("Accuracy on test set:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # 保存模型到文件
    # joblib.dump(model, 'trained_model2.pkl')

    # 加载模型
    # model_loaded = joblib.load('trained_model2.pkl')

    # 预测正常还款中的数据是否逾期
    # 假设df_normal是状态为1的数据
    # df_normal = df[df['标当前状态'] == 1]
    # X_normal = df_normal[numerical_features + categorical_features]
    # y_pred_normal = model_loaded.predict(X_normal)
    #
    # # 打印预测结果
    # print("预测结果 for normal repayment:", y_pred_normal)
    # a = 0
    # b = 0
    # for i in range(len(y_pred_normal)):
    #     if y_pred_normal[i] == 0:
    #         a += 1
    #     else:
    #         b += 1
    # print(a, b, a / b)
    #
    # # ...之前的代码...

    # 加载新数据并预测
    # new_df = pd.read_csv('bian.csv')  # 确保文件路径正确
    # # 选择新数据的数值特征和分类特征
    # X_new = new_df[numerical_features + categorical_features]
    # # 使用加载的模型进行预测
    # y_pred_new = model_loaded.predict(X_new)
    #
    # # 打印新数据的预测结果
    # print("预测结果 for new data:", y_pred_new)
    #
    # # 计算并打印新数据中预测为0和1的数量及其比例
    # a = y_pred_new[y_pred_new == 0].shape[0]
    # b = y_pred_new[y_pred_new == 1].shape[0]
    # print(a, b, a / b if b != 0 else "No逾期 cases predicted")

    # 注意：如果b为0，即没有预测为逾期的案例，将导致除以0的错误。
    # 这里使用了一个条件表达式来避免这种情况。

    # ...之前的代码...
    # 加载新数据
    new_df = pd.read_csv(csv_file)  # 确保文件路径正确

    # 选择新数据的数值特征和分类特征
    X_new = new_df[numerical_features + categorical_features]

    # 使用加载的模型预测新数据的逾期概率
    probabilities = model.predict_proba(X_new)[:, 1]  # 逾期概率

    # 将逾期概率添加到新数据的DataFrame中
    new_df['default_probability'] = probabilities

    # 定义风险等级划分函数
    def assign_risk_level(probability):
        if probability < 0.1:
            return 'A'
        elif 0.1 <= probability < 0.2:
            return 'B'
        elif 0.2 <= probability < 0.4:
            return 'C'
        elif 0.4 <= probability < 0.6:
            return 'D'
        else:
            return 'E'

    # 使用 map 方法根据逾期概率分配风险等级
    new_df['risk_level'] = new_df['default_probability'].map(lambda prob: assign_risk_level(prob))
    selected_columns = ['juid','借款金额','default_probability', 'risk_level']
    result_df = new_df[selected_columns]
    # 打印新数据的逾期概率和风险等级
    print("新数据的逾期概率：")
    print(new_df['default_probability'])
    print("新数据的风险等级：")
    print(new_df['risk_level'])

    # 保存包含逾期概率和风险等级的DataFrame到CSV文件
    # new_df.to_csv('new_data_with_default_probabilities_and_risk_levels.csv', index=False)
    return result_df


def handle_csv_file1(csv_file):
    import pandas
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import joblib
    from sklearn.tree import DecisionTreeRegressor

    # 加载数据
    df = pd.read_csv('2.csv')  # 替换为您的CSV文件路径

    # 选择数值特征和分类特征
    numerical_features = [
        '待还本金', '待还利息',

    ]

    categorical_features = [
       '是否首标',
         '学历认证', '征信认证'
    ]

    # 目标变量
    target = '借款金额'

    # 确保X包含了数值特征和分类特征
    X = df[numerical_features + categorical_features]
    y = df[target]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)  # 从 scikit-learn 1.0 开始默认为False
        ])

    # 创建模型管道
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # 1. 岭回归
    ridge_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    # 2. 套索回归
    lasso_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Lasso())
    ])

    # 3. 弹性网络回归
    elasticnet_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet())
    ])

    # 4. 决策树回归
    tree_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor())
    ])

    # 5. 随机森林回归
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    # 6. 梯度提升树回归
    gb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])

    # 训练模型
    # model.fit(X_train, y_train)

    rf_model.fit(X_train, y_train)

    # 预测测试集
    y_pred = rf_model.predict(X_test)

    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)





    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # 假设您的预处理器和模型已经按照之前的代码定义好了
    new_df = pandas.read_csv(csv_file)

    # 创建新的Da-taFrame
    new_df = pd.DataFrame(new_df)

    # 选择用于预测的特征
    X_new = new_df[numerical_features + categorical_features]

    # 使用训练好的随机森林模型进行预测
    y_new_pred = rf_model.predict(X_new)

    # 将预测结果添加到新数据的DataFrame中
    new_df['预测的借款金额'] = y_new_pred
    selected_columns = ['juid','预测的借款金额']
    result_df = new_df[selected_columns]
    # 输出包含预测结果的新数据
    print(new_df["预测的借款金额"])
    return result_df

# views.py
from django.shortcuts import render
import json
from django.http import HttpResponse

# views.py
from django.shortcuts import render
import json
from django.core.files.storage import default_storage
from django.conf import settings
import os

def loan_data_view(request):
    # 假设 JSON 文件位于 STATICFILES_DIRS 指定的目录之一
    json_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'json.json')

    loan_data = {}
    try:
        # 使用 Python 自带的 open 方法读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            loan_data = json.load(json_file)
    except FileNotFoundError:
        print("JSON 文件未找到")
    except json.JSONDecodeError:
        print("JSON 解码错误")
    except Exception as e:
        print(f"发生错误: {e}")

    context = {
        'loan_data_json': json.dumps(loan_data),
    }
    return render(request, 'charts.html', context)

def loan_data_view1(request):
    # 假设 JSON 文件位于 STATICFILES_DIRS 指定的目录之一
    json_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'json.json')

    loan_data = {}
    try:
        # 使用 Python 自带的 open 方法读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            loan_data = json.load(json_file)
    except FileNotFoundError:
        print("JSON 文件未找到")
    except json.JSONDecodeError:
        print("JSON 解码错误")
    except Exception as e:
        print(f"发生错误: {e}")

    context = {
        'loan_data_json': json.dumps(loan_data),
    }
    return render(request, 'charts1.html', context)


def loan_data_view2(request):
    # 假设 JSON 文件位于 STATICFILES_DIRS 指定的目录之一
    json_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'json.json')

    loan_data = {}
    try:
        # 使用 Python 自带的 open 方法读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            loan_data = json.load(json_file)
    except FileNotFoundError:
        print("JSON 文件未找到")
    except json.JSONDecodeError:
        print("JSON 解码错误")
    except Exception as e:
        print(f"发生错误: {e}")

    context = {
        'loan_data_json': json.dumps(loan_data),
    }
    return render(request, 'charts2.html', context)

def loan_data_view3(request):
    # 假设 JSON 文件位于 STATICFILES_DIRS 指定的目录之一
    json_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'json.json')

    loan_data = {}
    try:
        # 使用 Python 自带的 open 方法读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            loan_data = json.load(json_file)
    except FileNotFoundError:
        print("JSON 文件未找到")
    except json.JSONDecodeError:
        print("JSON 解码错误")
    except Exception as e:
        print(f"发生错误: {e}")

    context = {
        'loan_data_json': json.dumps(loan_data),
    }
    return render(request, 'charts3.html', context)

def loan_data_view4(request):
    # 假设 JSON 文件位于 STATICFILES_DIRS 指定的目录之一
    json_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'json.json')

    loan_data = {}
    try:
        # 使用 Python 自带的 open 方法读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            loan_data = json.load(json_file)
    except FileNotFoundError:
        print("JSON 文件未找到")
    except json.JSONDecodeError:
        print("JSON 解码错误")
    except Exception as e:
        print(f"发生错误: {e}")

    context = {
        'loan_data_json': json.dumps(loan_data),
    }
    return render(request, 'charts4.html', context)