#Первая версия этого кода [https://github.com/Alex2023MPEI/Yandex_ML_training_contest_Task_F_Biometrics/blob/main/p1.py] была написана для задачи (Тренировочный контест: машинное обучение, задача F. Биометрия): https://contest.yandex.ru/contest/28413/problems/F/ дата: 2025 август-сентябрь
#Вторая версия этого кода [https://github.com/Alex2023MPEI/Yandex_ML_training_2023_Final_contest_Regression/blob/main/p1.py] была написана для задачи (Final contest ML тренировки 2023, Финальное соревнование): https://contest.yandex.ru/contest/56809/problems/ дата: 2025 сентябрь-ноябрь
#Это третья версия, написана для получения экспериментальных данных при проведении исследований по теме "Система автоматического опредеения и удержания объекта в фокусе" дата: 2026 март-апрель.
import numpy as np;import pandas as pd;
import datetime,json,math,os,pathlib,pickle,pprint,random,string,time,typing;
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler;
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GroupKFold,StratifiedGroupKFold;
from sklearn.impute import KNNImputer,SimpleImputer;
#Импорты для классификации:
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier,Perceptron,RidgeClassifier,SGDClassifier;
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier;
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier;
from sklearn.naive_bayes import BernoulliNB,ComplementNB,GaussianNB,MultinomialNB;
from sklearn.gaussian_process import GaussianProcessClassifier;
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid,RadiusNeighborsClassifier;
from sklearn.neural_network import MLPClassifier;
from xgboost import XGBClassifier,XGBRFClassifier;from lightgbm import LGBMClassifier,DaskLGBMClassifier;
from sklearn.metrics import accuracy_score,auc,average_precision_score,balanced_accuracy_score,brier_score_loss,roc_curve;
from sklearn.metrics import cohen_kappa_score,dcg_score,f1_score,fbeta_score,hamming_loss,hinge_loss,jaccard_score;
from sklearn.metrics import log_loss,matthews_corrcoef,ndcg_score,precision_score,recall_score,roc_auc_score,zero_one_loss;
#Импорты для регрессии:
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor,ElasticNet,Lasso,LassoLarsIC,ARDRegression,OrthogonalMatchingPursuit;
from sklearn.linear_model import BayesianRidge,MultiTaskElasticNet,MultiTaskLasso,HuberRegressor,QuantileRegressor,RANSACRegressor,Lars;
from sklearn.linear_model import TheilSenRegressor,GammaRegressor,PoissonRegressor,TweedieRegressor,PassiveAggressiveRegressor,LassoLars;
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor;
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor;
from sklearn.gaussian_process import GaussianProcessRegressor;
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor;
from sklearn.neural_network import MLPRegressor;
from xgboost import XGBRegressor,XGBRFRegressor;from lightgbm import LGBMRegressor,DaskLGBMRegressor;
from sklearn.metrics import d2_absolute_error_score,d2_pinball_score,d2_tweedie_score,explained_variance_score,max_error;
from sklearn.metrics import mean_absolute_percentage_error,mean_gamma_deviance,mean_pinball_loss,mean_poisson_deviance;
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_tweedie_deviance,median_absolute_error;
from sklearn.metrics import r2_score,root_mean_squared_error,root_mean_squared_log_error;
#Импорты для feature_selector (классы отбора признаков):
from sklearn.feature_selection import GenericUnivariateSelect,RFE,RFECV,SelectFdr,SelectFpr,SelectFromModel,SelectFwe,SelectKBest;
from sklearn.feature_selection import SelectPercentile,SequentialFeatureSelector,VarianceThreshold;
#Импорты для feature_selector (функции для score_func для SelectFdr,SelectFpr,SelectFwe,SelectKBest,SelectPercentile):
from sklearn.feature_selection import chi2,f_classif,f_regression,mutual_info_classif,mutual_info_regression,r_regression;
from sklearn.svm import LinearSVC,LinearSVR,NuSVC,NuSVR,SVC,SVR;

AnyImputer:typing.TypeAlias=KNNImputer|SimpleImputer;#IterativeImputer is experimental and the API might change without any deprecation cycle.
AnyScaler:typing.TypeAlias=MaxAbsScaler|MinMaxScaler|RobustScaler|StandardScaler;
AnyFSEstimator:typing.TypeAlias=LogisticRegression|PassiveAggressiveClassifier|Perceptron|RidgeClassifier|SGDClassifier|LinearRegression|Ridge|SGDRegressor|ElasticNet|Lars|Lasso|LassoLars|LassoLarsIC|OrthogonalMatchingPursuit|ARDRegression|BayesianRidge|HuberRegressor|QuantileRegressor|RANSACRegressor|TheilSenRegressor|GammaRegressor|PoissonRegressor|TweedieRegressor|PassiveAggressiveRegressor|LinearSVC|NuSVC|SVC|LinearSVR|NuSVR|SVR;
AnyFeatureSelector:typing.TypeAlias=GenericUnivariateSelect|RFE|RFECV|SelectFdr|SelectFpr|SelectFromModel|SelectFwe|SelectKBest|SelectPercentile|SequentialFeatureSelector;
AnyModelClassifier:typing.TypeAlias=LogisticRegression|PassiveAggressiveClassifier|Perceptron|RidgeClassifier|SGDClassifier|AdaBoostClassifier|BaggingClassifier|ExtraTreesClassifier|GradientBoostingClassifier|HistGradientBoostingClassifier|RandomForestClassifier|XGBClassifier|LGBMClassifier|XGBRFClassifier|DaskLGBMClassifier|GaussianProcessClassifier|KNeighborsClassifier|RadiusNeighborsClassifier|MLPClassifier|NearestCentroid|MultinomialNB|ComplementNB|BernoulliNB|GaussianNB;
AnyModelRegressor:typing.TypeAlias=LinearRegression|Ridge|SGDRegressor|ElasticNet|Lasso|LassoLarsIC|ARDRegression|OrthogonalMatchingPursuit|BayesianRidge|MultiTaskElasticNet|MultiTaskLasso|HuberRegressor|QuantileRegressor|RANSACRegressor|Lars|TheilSenRegressor|GammaRegressor|PoissonRegressor|TweedieRegressor|PassiveAggressiveRegressor|LassoLars|AdaBoostRegressor|BaggingRegressor|ExtraTreesRegressor|GradientBoostingRegressor|HistGradientBoostingRegressor|RandomForestRegressor|XGBRegressor|XGBRFRegressor|LGBMRegressor|DaskLGBMRegressor|GaussianProcessRegressor|KNeighborsRegressor|RadiusNeighborsRegressor|MLPRegressor;
AnyModel:typing.TypeAlias=AnyModelClassifier|AnyModelRegressor;

def true_with_prob(p:float=0.5)->bool:
    """Функция возвращает True с вероятностью p и False с вероятностью q=1-p"""
    if isinstance(p,float)==False:p:float=0.5;#Если вдруг по ошибке передано значение не того типа
    return random.random()<min(max(p,0.0),1.0);#Return the next random floating-point number in the range 0.0<=X<1.0

def lnx(x:float,eps:float=1.0e-15)->float:
    """Функция возвращает ln(x). Стандартная функция Python log1p(x)=ln(1+x), поэтому эта для удобства. eps для избежания ошибок вида [ValueError: expected argument value > -1, got -1.0]."""
    #log1p(x)=ln(1+x) -> 1+x=y, x=y-1 => ln(1+x)=ln(y)=log1p(x)=log1p(y-1) => ln(y)=log1p(y-1) => ln(x)=log1p(x-1)
    return math.log1p(x-1.0+eps);

def logit_float(x:float|list[float],prob_eps:float=1.0e-15)->float|list[float]:
    """Функция вычисляет logit(x)=ln(x/[1.0-x]), 0.0<x<1.0 (функции sigmoid и logit взаимно-обратны). Вероятности ограничиваются значениеями от prob_eps до 1.0-prob_eps (так как если вероятность близка к 0 или 1, её логит будет стремиться к -inf или +inf соответственно)."""
    #print(f'x (перед преобразованием во float или list[float]): {x}, type(x): {type(x)}');
    if isinstance(x,(float,np.float16,np.float32,np.float64)):
        x:float=float(x);
        if x<prob_eps:x=prob_eps;
        if x>1.0-prob_eps:x=1.0-prob_eps;
    elif isinstance(x[0],(float,np.float16,np.float32,np.float64)):
        x:list[float]=[float(x[i])for i in range(len(x))];
        for i in range(len(x)):
            if x[i]<prob_eps:x[i]=prob_eps;
            if x[i]>1.0-prob_eps:x[i]=1.0-prob_eps;
    #print(f'x (после преобразования во float или list[float]): {x}, type(x): {type(x)}');
    if isinstance(x,float):
        return lnx(x=x/(1.0-x),eps=lnx_eps);
    if isinstance(x,list):
        return [lnx(x=x[i]/(1.0-x[i]),eps=lnx_eps) for i in range(len(x))];

def sigmoid_float(x:float|list[float])->float|list[float]:
    """Функция вычисляет sigmoid(x)=1/(1+exp[-x]), -inf<x<+inf (функции sigmoid и logit взаимно-обратны)"""
    if isinstance(x,(float,np.float16,np.float32,np.float64)):x=float(x);
    elif isinstance(x[0],(float,np.float16,np.float32,np.float64)):x:list[float]=[float(x[i])for i in range(len(x))];
    if isinstance(x,float):return 1.0/(1.0+math.exp(-x));
    if isinstance(x,list):return [1.0/(1.0+math.exp(-x[i])) for i in range(len(x))];

def str_to_float(s:str='1.0',num_min:float=0.0,num_max:float=1.0,num_default:float=1.0)->float:
    """Функция преобразует строку s в число типа float с корретной обработкой возможных исключений"""
    if isinstance(s,float)==True:return s;#Если s типа float, то сразу возвращаем s
    try:
        num:float=float(s);
    except:
        num:float=num_default;
        print(f'Строку [{s}] невозможно преобразовать к типу float, вместо этого возвращено значение {num}');
    if num<num_min:
        print(f'Число num={num} меньше чем num_min={num_min}, поэтому возвращено число {num_min}');
        num=num_min;
    if num>num_max:
        print(f'Число num={num} больше чем num_max={num_max}, поэтому возвращено число {num_max}');
        num=num_max;
    return num;

def str_or_bool_to_bool(s:str|bool)->bool:
    """Функция преобразует строку s в значение типа bool"""
    if isinstance(s,bool)==True:return s;#Если s изначально имеет тип bool, то вернуть s
    if isinstance(s,str)==False:s=str(s);#Если s не строка (и не bool, так как если bool, то эта строка кода не достижима), то преобразуем s в str
    if len(s)==0:return False;
    elif 't' in s.lower():return True;#True
    elif 'y' in s.lower():return True;#Yes
    elif '1' in s.lower():return True;#1 = True
    elif 'f' in s.lower():return False;#False
    elif 'n' in s.lower():return False;#No
    elif '0' in s.lower():return False;#0 = False
    return False;

def cur_dt_to_str()->str:
    """Функция возвращает текущие дату и время в виде 2026-04-18_16-04-01.147265 (минус вместо двоеточия для возможности использовать в названии файла)"""
    cur_dt:datetime.datetime=datetime.datetime.now();
    return cur_dt.strftime(format=f'%Y-%m-%d_%H-%M-%S.%f');#2026-04-18_16-04-01.147265

def load_config(config_file_name:str)->dict[str:str|int|float|list|dict]:
    '''Функция загружает настройки (значения глобальных переменных) из JSON файла config_file_name'''
    config_file_path:pathlib.Path=pathlib.Path(config_file_name);
    if config_file_path.exists():
        try:
            with open(file=config_file_name,mode='rt',encoding='UTF-8')as f:
                result:dict=json.load(fp=f);
                print(f'Конфигурация загружена из файла {config_file_name}');
                return result;
        except:
            print(f'Ошибка при считывании файла {config_file_name} как JSON, конфигурация не загружена');
            return -1;
    else:
        print(f'Путь файла {config_file_name} не существует, конфигурация не загружена');
        return -1;

def generate_hidden_layer_sizes_tuple(n_layers:int=3,n_in:int=100,n_out:int=5,allow_increase:bool=False,log_scale:bool=False)->tuple[int]:
    """Функция возвращает кортеж из чисел типа int, обозначающих количества нейронов в скрытых слоях Multi-layer Perceptron, n_in>n_out"""
    sizes_lst:list[int]=[];
    if n_layers<1:n_layers=1;
    if n_in<1:n_in=1;
    if n_out<1:n_out=1;
    if n_in<n_out:allow_increase=True;#Иначе сеть не построить
    if allow_increase==True:
        if n_in>=n_out:
            for i in range(n_layers):sizes_lst.append(random.randint(a=n_out,b=n_in));
        else:
            for i in range(n_layers):sizes_lst.append(random.randint(a=n_in,b=n_out));
    else:#allow_increase==False
        if log_scale==False:
            step:int=(n_in-n_out)//(n_layers+1);
            for i in range(n_layers):sizes_lst.append(random.randint(a=n_in-step*(i+2),b=n_in-step*(i+1)));
        else:
            log_step:float=(n_in/n_out)**(1/(n_layers+1));
            for i in range(n_layers):sizes_lst.append(random.randint(a=int(n_in/(log_step**(i+2))),b=int(n_in/(log_step**(i+1)))));
    return tuple(sizes_lst);

#ОЧЕНЬ ПОЛЕЗНАЯ ФУНКЦИЯ (список словарей в Python в некотором смысле аналогичен массиву записей в Delphi)
def read_csv(filename:str,delimiter_values:str=',')->list[dict]:#должна возвращать список словарей
    with open(file=filename,mode='rt',encoding='utf-8')as f:lst_lines=[s.rstrip()for s in f.readlines()];
    lst_of_dicts:list[dict]=[];#Изначально список словарей пустой
    lst_keys:list[str]=lst_lines[0].split(delimiter_values);#Первая строка csv файла - это ключи
    lst_values:list[list[str]]=[lst_lines[line_num].split(delimiter_values)for line_num in range(1,len(lst_lines))];
    for dict_num in range(len(lst_lines)-1):lst_of_dicts.append(dict(tuple(zip(lst_keys,lst_values[dict_num]))));
    return(lst_of_dicts);
#И обратная ей функция (записывает список словарей в csv файл)
def write_csv(filename:str,delimiter_values:str,lst_of_dicts:list[dict])->None:#должна принимать список словарей и записывать в *.csv
    s:str=delimiter_values.join(list(lst_of_dicts[0].keys()))+'\n';#Сначала записываем в строку ключи (названия столбцов)
    for d in lst_of_dicts:s=s+delimiter_values.join(d.values())+'\n';#Затем в цикле добавляем в строку значения
    with open(file=filename,mode='wt',encoding='utf-8')as f:f.write(s);
    return None;

def analize_log_pipelines_csv(log_pipelines_csv_file_name:str='log_pipelines.csv',score_valid_mean_threshold_min:float=None,score_valid_mean_threshold_max:float=None,score_test_threshold_min:float=None,score_test_threshold_max:float=None,n_features_selected_randomly_threshold_min:int=None,n_features_selected_randomly_threshold_max:int=None,pipeline_file_size_theshold_min:int=None,pipeline_file_size_theshold_max:int=None)->None:
    """Анализ csv файла для определения id лучших пайплайнов (для которых выполняются заданные ограничения)"""
    log_pipelines_dicts_list:list[dict]=read_csv(filename=log_pipelines_csv_file_name,delimiter_values=',');
    #Преобразование значений некоторых полей из типа str в типы int или float
    if'n_features_all'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['n_features_all']=int(pipeline_dict['n_features_all']);
    if'n_features_selected_randomly'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['n_features_selected_randomly']=int(pipeline_dict['n_features_selected_randomly']);
    if'score_valid_mean'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['score_valid_mean']=float(pipeline_dict['score_valid_mean']);
    if'score_test'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['score_test']=float(pipeline_dict['score_test']);
    if'pipeline_file_size'in log_pipelines_dicts_list[0].keys():
        for pipeline_dict in log_pipelines_dicts_list:pipeline_dict['pipeline_file_size']=int(pipeline_dict['pipeline_file_size']);
    print(f'log_pipelines_dicts_list: {log_pipelines_dicts_list}');
    print(f'len(log_pipelines_dicts_list): {len(log_pipelines_dicts_list)}');
    numeric_keys:list[str]=['n_features_all','n_features_selected_randomly','score_valid_mean','score_test','pipeline_file_size'];
    numeric_keys_stats:list[dict[str:float]]=[];
    for i in range(len(numeric_keys)):
        numeric_keys_stats.append({});
        name:str=numeric_keys[i];
        numeric_keys_stats[i]['name']=numeric_keys[i];
        numeric_keys_stats[i]['number']=sum([1 for j in range(len(log_pipelines_dicts_list))]);
        numeric_keys_stats[i]['sum']=sum([log_pipelines_dicts_list[j][name]for j in range(len(log_pipelines_dicts_list))]);
        numeric_keys_stats[i]['mean']=numeric_keys_stats[i]['sum']/numeric_keys_stats[i]['number'];
        numeric_keys_stats[i]['min']=min([log_pipelines_dicts_list[j][name]for j in range(len(log_pipelines_dicts_list))]);
        numeric_keys_stats[i]['max']=max([log_pipelines_dicts_list[j][name]for j in range(len(log_pipelines_dicts_list))]);
        numeric_keys_stats[i]['std']=(sum([(log_pipelines_dicts_list[j][name]-numeric_keys_stats[i]['mean'])**2 for j in range(len(log_pipelines_dicts_list))])/numeric_keys_stats[i]['number'])**0.5;
        print(f'i: {i}, numeric_keys_stats[i]: {numeric_keys_stats[i]}');
    for name in ['n_features_selected_randomly','score_valid_mean','score_test']:
        min_sub_list:list[dict]=sorted(log_pipelines_dicts_list,key=lambda d:d[name],reverse=False)[:10];
        print(f'Пайплайны с наименьшими значениями {name}:');
        for i in range(len(min_sub_list)):print(f'{i}) {min_sub_list[i]}');
        max_sub_list:list[dict]=sorted(log_pipelines_dicts_list,key=lambda d:d[name],reverse=True)[:10];
        print(f'Пайплайны с наибольшими значениями {name}:');
        for i in range(len(max_sub_list)):print(f'{i}) {max_sub_list[i]}');
    s_lst:list[dict]=[d for d in log_pipelines_dicts_list];#s_lst - список только тех пайплайнов, у которых числовые значения удовлетворяют ограничениям
    if score_valid_mean_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['score_valid_mean']>=score_valid_mean_threshold_min];
    if score_valid_mean_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['score_valid_mean']<=score_valid_mean_threshold_max];
    if score_test_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['score_test']>=score_test_threshold_min];
    if score_test_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['score_test']<=score_test_threshold_max];
    if n_features_selected_randomly_threshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['n_features_selected_randomly']>=n_features_selected_randomly_threshold_min];
    if n_features_selected_randomly_threshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['n_features_selected_randomly']<=n_features_selected_randomly_threshold_max];
    if pipeline_file_size_theshold_min is not None:s_lst:list[dict]=[d for d in s_lst if d['pipeline_file_size']>=pipeline_file_size_theshold_min];
    if pipeline_file_size_theshold_max is not None:s_lst:list[dict]=[d for d in s_lst if d['pipeline_file_size']<=pipeline_file_size_theshold_max];
    
    print(f'Список отобранных пайплайнов:');
    for d in s_lst:print(d);
    print(f'В списке отобранных пайплайнов {len(s_lst)} пайплайнов из {len(log_pipelines_dicts_list)} ({(100*len(s_lst)/len(log_pipelines_dicts_list)):.4f}%)');
    print(f'Пайплайны отобраны с ограничениями: {locals()}');
    ids_of_selected_pipelines:list[str]=sorted([d['pipeline_id']for d in s_lst]);
    print(f'id выбранных {len(ids_of_selected_pipelines)} пайплайнов: [{" ".join(ids_of_selected_pipelines)}]');

def analize_log_pipelines_txt(log_pipelines_txt_file_name:str='log_pipelines.txt',num_of_most_times_used_features_indexes:int=14)->None:
    """Функция проводит анализ файла log_pipelines.txt и выявляет, какие признаки (по индексам) наиболее часто использовались в лучших
    пайплайнах. Это полезно в том случае, если необходимо построить пайплайн с использованием не более чем некоторого количества признаков
    (например, в задаче [B. Финальное соревнование: задача 2], где в условии сказано: [Вторая модель должна быть линейной, т.е.
    представлять собой линейную комбинацию признаков плюс смещение, модель не должна использовать более 15 параметров (14 весов плюс
    смещение)])\n"""
    with open(file=log_pipelines_txt_file_name,mode='rt',encoding='UTF-8')as f:
        txt_log_lines:list[str]=[line.rstrip('\n') for line in f.readlines()];
    print(f'len(txt_log_lines): {len(txt_log_lines)}');
    print(f'txt_log_lines[4]: {txt_log_lines[4]}');
    randomly_selected_indexes_lines:list[str]=[line for line in txt_log_lines if 'randomly_selected_indexes: 'in line];
    randomly_selected_indexes_lines=[line.replace('randomly_selected_indexes: [','').replace(']',',').replace(' ','')for line in randomly_selected_indexes_lines];
    print(f'len(randomly_selected_indexes_lines): {len(randomly_selected_indexes_lines)}');
    print(f'randomly_selected_indexes_lines[:10]: {randomly_selected_indexes_lines[:10]}');
    n_features_selected_randomly_lines:list[str]=[line for line in txt_log_lines if 'n_features_selected_randomly: 'in line];
    n_features_selected_randomly_lines=[line.replace('n_features_selected_randomly: ','') for line in n_features_selected_randomly_lines];
    n_features_selected_randomly_ints:list[int]=[int(line)for line in n_features_selected_randomly_lines];
    print(f'len(n_features_selected_randomly_ints): {len(n_features_selected_randomly_ints)}');
    print(f'n_features_selected_randomly_ints[:50]: {n_features_selected_randomly_ints[:50]}');
    print(f'sum(n_features_selected_randomly_ints): {sum(n_features_selected_randomly_ints)}');
    randomly_selected_indexes_str:str=''.join(randomly_selected_indexes_lines);
    randomly_selected_indexes_list:list[int]=[int(num)for num in randomly_selected_indexes_str.split(sep=',')if len(num)>0];#Чтобы не пытаться
    #преобразовать пустую строку после последней запятой в число типа int
    print(f'len(randomly_selected_indexes_list): {len(randomly_selected_indexes_list)}');
    print(f'randomly_selected_indexes_list[:50]: {randomly_selected_indexes_list[:50]}');
    if sum(n_features_selected_randomly_ints)==len(randomly_selected_indexes_list):
        print(f'sum(n_features_selected_randomly_ints): {sum(n_features_selected_randomly_ints)}, len(randomly_selected_indexes_list): {len(randomly_selected_indexes_list)}, эти числа равны, проверка работает правильно');
    else:#Сумма n_features_selected_randomly_ints должна быть равна длине randomly_selected_indexes_list (и равна суммарному количеству выбранных признаков во всех пайплайнах, попавших в файл log_pipelines.txt)
        print(f'sum(n_features_selected_randomly_ints): {sum(n_features_selected_randomly_ints)}, len(randomly_selected_indexes_list): {len(randomly_selected_indexes_list)}, эти числа НЕ равны, проверка показывает наличие ошибки');
    selected_non_zero_times_indexes:list[int]=sorted(list(set(randomly_selected_indexes_list)));
    print(f'selected_non_zero_times_indexes: {selected_non_zero_times_indexes}, len(selected_non_zero_times_indexes): {len(selected_non_zero_times_indexes)}');
    #Сохранение в словарь {index:num_of_this_index}
    indexes_times_dict:dict[int:int]={};#Ключи - индексы, значения - их количества
    for ind in selected_non_zero_times_indexes:indexes_times_dict[ind]=0;
    for i in randomly_selected_indexes_list:indexes_times_dict[i]=indexes_times_dict[i]+1;
    print(f'indexes_times_dict: {indexes_times_dict}');
    #Сохранение в список словарей [{'index':,'times':},...,{'index':,'times':}]
    indexes_times_list_dicts:list[dict[str:int]]=[];
    for ind in selected_non_zero_times_indexes:indexes_times_list_dicts.append({'index':ind,'times':0});
    #Это рабочий вариант, но лучше так не делать, так как он рассчитывает на то, что словари для всех выбранных индексов
    #расположены по порядку увеличения этих индексов без пропусков
    #for ind in randomly_selected_indexes_list:indexes_times_list_dicts[ind]['times']=indexes_times_list_dicts[ind]['times']+1;
    #Другой вариант (тоже рабочий, но должен быть более общим):
    for ind in randomly_selected_indexes_list:
        index:int=None;
        for index_key in range(len(indexes_times_list_dicts)):#Для эффективности можно заменить это for на while но число признаков
            if indexes_times_list_dicts[index_key]['index']==ind:#вряд ли будет больше ста тысяч, поэтому можно так оставить
                index=ind;
        print(f'len(indexes_times_list_dicts): {len(indexes_times_list_dicts)}, index: {index}');
        indexes_times_list_dicts[index]['times']=indexes_times_list_dicts[index]['times']+1;
    print(f'indexes_times_list_dicts (before sorting): {indexes_times_list_dicts}');
    indexes_times_list_dicts.sort(key=lambda d:d['times'],reverse=True);#Сортировка по убыванию количеств
    print(f'indexes_times_list_dicts (after sorting): {indexes_times_list_dicts}');
    #[{'index':218,'times':114},{'index':307,'times':113},...,{'index':290,'times':71},{'index':269,'times': 69}]
    print(f'{num_of_most_times_used_features_indexes} наиболее часто использованных признаков (в виде словаря номер:количество): {indexes_times_list_dicts[:num_of_most_times_used_features_indexes]}');
    print(f'{num_of_most_times_used_features_indexes} наиболее часто использованных признаков (в виде списка): {[d["index"] for d in indexes_times_list_dicts[:num_of_most_times_used_features_indexes]]}');
    #14 наиболее часто использованных признаков (в виде словаря номер:количество): [{'index': 218, 'times': 114}, {'index': 307, 'times': 113}, {'index': 56, 'times': 111}, {'index': 266, 'times': 111}, {'index': 63, 'times': 110}, {'index': 67, 'times': 110}, {'index': 77, 'times': 109}, {'index': 336, 'times': 108}, {'index': 84, 'times': 107}, {'index': 105, 'times': 107}, {'index': 376, 'times': 106}, {'index': 59, 'times': 105}, {'index': 73, 'times': 105}, {'index': 257, 'times': 105}]
    #14 наиболее часто использованных признаков (в виде списка): [218, 307, 56, 266, 63, 67, 77, 336, 84, 105, 376, 59, 73, 257]

def analize_one_pkl_file(pkl_file_name:str)->None:
    """Функция выводит информацию о содержимом одного pkl файла"""
    with open(file=pkl_file_name,mode='rb')as pkl_file:#binary mode doesn't take an encoding argument
        print(f'======== Информация об одном pkl файле ========:');
        pkl_file_size:int=os.path.getsize(filename=pkl_file_name);
        print(f'pkl_file_name: {pkl_file_name}, pkl_file_size: {pkl_file_size}');        
        pkl_obj=pickle.load(file=pkl_file);
        print(f'pkl_obj: {pkl_obj}');
        print(f'type(pkl_obj): {type(pkl_obj)}');
        if type(pkl_obj)==list:pass;
        elif type(pkl_obj)==dict:
            pkl_obj_keys=pkl_obj.keys();
            print(f'pkl_obj_keys: {pkl_obj_keys}');
            for key in pkl_obj_keys:
                value=pkl_obj[key];
                __dict__str:str=f'Объект {value} не имеет атрибута __dict__';
                if hasattr(value,'__dict__'):__dict__str=f'value.__dict__: {value.__dict__}';
                print(f'key: {key}, type(key): {type(key)}, value: {value}, type(value): {type(value)}, \n value.__dir__(): {value.__dir__()}, \n {__dict__str}\n');
#Функции - трансформации признаков:
def data_transformation_pairwise_multiplications(feature_matrix:np.ndarray)->np.ndarray:
    """Функция добавляет к матрице признаков все попарные произведения исходных признаков. Функция не использует никакие статистики
    по всей выборке или какой-либо части выборки, поэтому НЕ ПРИВОДИТ к утечке данных."""
    n_samples:int=feature_matrix.shape[0] ;#количество строк таблицы, остаётся постоянным при добавлении новых признаков
    n_features:int=feature_matrix.shape[1];#количество столбцов таблицы, увеличивается при добавлении новых признаков
    n_pairs:int=n_features*(n_features-1)//2;#количество попарных произведений исходных признаков
    new_feature_matrix:np.ndarray=np.zeros((n_samples,n_features+n_pairs));#Добавление признаков (столбцов), заполненных нулями
    new_feature_matrix[:,:n_features]=feature_matrix;#Копирование значений имеющихся столбцов
    current_col:int=n_features;#Номер текущего столбца для записи нового признака (начинается с n_features)
    for i in range(n_features):
        for j in range(i+1,n_features):
            new_feature_matrix[:,current_col]=feature_matrix[:,i]*feature_matrix[:,j];
            current_col=current_col+1;
    return new_feature_matrix;
def data_transformation_add_functions(feature_matrix:np.ndarray)->np.ndarray:
    """Функция добавляет к матрице признаков результаты применения различных функций к исходным признакам. Функция не использует
    никакие статистики по всей выборке или какой-либо части выборки, поэтому НЕ ПРИВОДИТ к утечке данных."""
    n_samples:int=feature_matrix.shape[0] ;#Количество строк таблицы, остаётся постоянным при добавлении новых признаков
    n_features:int=feature_matrix.shape[1];#Количество столбцов таблицы, увеличивается при добавлении новых признаков
    n_added_features:int=n_features*13;#Количество добавленных признаков (по 13 функций на каждый признак)
    new_feature_matrix:np.ndarray=np.zeros((n_samples,n_features+n_added_features));#Добавление признаков (столбцов), заполненных нулями
    new_feature_matrix[:,:n_features]=feature_matrix;#Копирование значений имеющихся столбцов
    current_col:int=n_features;#Начинаем добавлять с номера столбца n_features (номера 0..n_features-1 уже заняты)
    for i in range(n_features):
        x:np.ndarray=feature_matrix[:,i];        
        new_feature_matrix[:,current_col+0]=np.exp(x);              #1. exp(x)=e^x (экспонента)
        new_feature_matrix[:,current_col+1]=1.0/(1.0+np.exp(-x));   #2. sigmoid(x)=1/(1+e^(-x)) (сигмоида)
        new_feature_matrix[:,current_col+2]=x**2;                   #3. sqr(x)=x^2 (возведение в квадрат)
        new_feature_matrix[:,current_col+3]=np.abs(x);              #4. abs(x)=|x| (модуль числа)
        new_feature_matrix[:,current_col+4]=np.maximum(0,x);        #5. ReLU(x)=max(0,x) (функция активации ReLU)
        new_feature_matrix[:,current_col+5]=np.arctan(x);           #6. arctg(x) (арктангенс числа x)        
        new_feature_matrix[:,current_col+6]=np.sinh(x);             #7. sh(x)=(exp(x)-exp(-x))/2 (гиперболический синус)        
        new_feature_matrix[:,current_col+7]=np.cosh(x);             #8. ch(x)=(exp(x)+exp(-x))/2 (гиперболический косинус)        
        new_feature_matrix[:,current_col+8]=np.tanh(x);             #9. th(x)=(exp(2x)-1)/(exp(2x)+1) (гиперболический тангенс)        
        new_feature_matrix[:,current_col+9]=1.0/np.cosh(x);         #10. sch(x)=2/(exp(x)+exp(-x)) (гиперболический секанс)        
        new_feature_matrix[:,current_col+10]=np.log(np.abs(x)+1e-9);#11. ln_abs(x)=ln(abs(x)+1e-9) (натуральный логарифм от модуля)
        new_feature_matrix[:,current_col+11]=np.sqrt(np.abs(x));    #12. sqrt_abs(x)=sqrt(abs(x)) (корень из модуля)
        new_feature_matrix[:,current_col+12]=np.exp(-x**2);         #13. gauss(x)=exp(-x^2)
        current_col=current_col+13;#Увеличение current_col на количество добавленных признаков
    return new_feature_matrix;
def data_transformation_keep_same(feature_matrix:np.ndarray)->np.ndarray:
    """Функция оставляет матрицу признаков без изменений. Функция не использует никакие статистики по всей выборке или какой-либо
    части выборки, поэтому НЕ ПРИВОДИТ к утечке данных."""
    n_samples:int=feature_matrix.shape[0] ;#Количество строк таблицы, остаётся постоянным при добавлении новых признаков
    n_features:int=feature_matrix.shape[1];#Количество столбцов таблицы, увеличивается при добавлении новых признаков (но конкретно в этой функции остаётся постоянным)
    new_feature_matrix:np.ndarray=np.zeros((n_samples,n_features));
    new_feature_matrix[:,:n_features]=feature_matrix;#Копирование значений имеющихся столбцов
    return new_feature_matrix;

def load_data_from_npy(save_opened_and_closed_features_csvs:bool=False,save_opened_parquet:bool=False)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Загрузка данных из *.npy файлов (эти файлы содержат массивы NumPy типа np.ndarray) с возможной записью в csv/parquet и преобразованиями"""
    def print_loaded_data_info()->None:
        '''Функция выводит информацию о массивах opened_data, closed_data и opened_target, не внося никаких изменений.'''
        print(f'n_samples_opened: {n_samples_opened}, n_samples_closed: {n_samples_closed}, n_features_opened: {n_features_opened}, n_features_closed: {n_features_closed}');
        print(f'opened_data_all_features.shape: {opened_data_all_features.shape}, closed_data_all_features.shape: {closed_data_all_features.shape}, opened_target.shape: {opened_target.shape}');
        print(f'opened_data_all_features.dtype: {opened_data_all_features.dtype}, closed_data_all_features.dtype: {closed_data_all_features.dtype}, opened_target.dtype: {opened_target.dtype}');
        if debug_info_depth>=4:
            print(f'Первая строка открытых данных [float(a)for a in opened_data_all_features[0]]: {[float(a)for a in opened_data_all_features[0]]}');print(f'...');
            print(f'Последняя строка открытых данных [float(a)for a in opened_data_all_features[{n_samples_opened-1}]]: {[float(a)for a in opened_data_all_features[n_samples_opened-1]]}');
            print(f'Первая строка закрытых данных [float(a)for a in closed_data_all_features[0]]: {[float(a)for a in closed_data_all_features[0]]}');print(f'...');
            print(f'Последняя строка закрытых данных [float(a)for a in closed_data_all_features[{n_samples_closed-1}]]: {[float(a)for a in closed_data_all_features[n_samples_closed-1]]}');
            print(f'id открытых данных: {[opened_ids[sample_num] for sample_num in range(n_samples_opened)]}');
            print(f'id закрытых данных: {[closed_ids[sample_num] for sample_num in range(n_samples_closed)]}');
    opened_data_npy_filename:str=conf_dict['data_files_names']['opened_data_npy'];
    opened_target_npy_filename:str=conf_dict['data_files_names']['opened_target_npy'];
    closed_data_npy_filename:str=conf_dict['data_files_names']['closed_data_npy'];
    assert os.path.exists(path=opened_data_npy_filename),f'Нет файла с открытыми данными (признаками), в соответствии с конфигурацией он должен называться {opened_data_npy_filename}';
    assert os.path.exists(path=opened_target_npy_filename),f'Нет файла со значениями target открытых данных, в соответствии с конфигурацией он должен называться {opened_target_npy_filename}';
    assert os.path.exists(path=closed_data_npy_filename),f'Нет файла с закрытыми данными (признаками), в соответствии с конфигурацией он должен называться {closed_data_npy_filename}';
    opened_data_all_features:np.ndarray=np.load(file=opened_data_npy_filename,allow_pickle=False);#Загрузка данных
    opened_target:np.ndarray=np.load(file=opened_target_npy_filename,allow_pickle=False);
    closed_data_all_features:np.ndarray=np.load(file=closed_data_npy_filename,allow_pickle=False);
    closed_target:np.ndarray=np.ndarray(shape=(closed_data_all_features.shape[0],),dtype=opened_target.dtype);
    n_samples_opened:int=opened_data_all_features.shape[0];n_features_opened:int=opened_data_all_features.shape[1];n_samples_closed:int=closed_data_all_features.shape[0];n_features_closed:int=closed_data_all_features.shape[1];
    assert n_samples_opened==conf_dict['n_samples_opened'],f'Значение n_samples_opened в файле конфигурации (равное {conf_dict['n_samples_opened']}) отличается от количества строк файла {opened_data_npy_filename} (равного {n_samples_opened}), необходимо исправить ошибку';
    assert n_samples_closed==conf_dict['n_samples_closed'],f'Значение n_samples_closed в файле конфигурации (равное {conf_dict['n_samples_closed']}) отличается от количества строк файла {closed_data_npy_filename} (равного {n_samples_closed}), необходимо исправить ошибку';
    #В папке с этим скриптом должны быть файлы со значениями id для всех открытых и закрытых образцов. Если этих файлов НЕТ, значит
    #перед запуском этого скрипта их нужно создать отдельным скриптом, так как этот скрипт при запуске сразу их считывает.
    #Скрипт с созданием txt файлов с id будет небольшим, поэтому все данные можно задавать прямо в нём, не вынося в config.
    #ОСНОВНАЯ ИДЕЯ ТАКАЯ: в каких бы форматах (txt, csv, json, npy, ...) не были представлены входные данные (открытые, закрытые,
    #метки открытых), сначала отдельный скрипт create_input_files.py представляет все данные в одинаковой форме, а именно:
    #1. файлы txt с id открытых и закрытых данных (2 файла)
    #2. файлы npy с массивами np.ndarray для opened_data, opened_target, closed_data с dtype=np.float64
    #Тогда при использовании этого же скрипта для другой задачи нужно будет внести изменения только в маленький скрипт
    #create_input_files.py, а этот большой скрипт останется без изменений
    max_id_len:int=conf_dict['max_id_len'];#Все id - это строки, max_id_len - это их максимальная длина (длина этих строк может быть и меньше, но не больше)
    opened_ids:np.ndarray=np.ndarray(shape=(opened_data_all_features.shape[0],),dtype=f'U{str(max_id_len)}');#id - это строки, а не числа
    closed_ids:np.ndarray=np.ndarray(shape=(closed_data_all_features.shape[0],),dtype=f'U{str(max_id_len)}');
    opened_ids_txt_filename:str=conf_dict['data_files_names']['opened_ids_txt'];closed_ids_txt_filename:str=conf_dict['data_files_names']['closed_ids_txt'];
    with open(file=opened_ids_txt_filename,mode='rt',encoding='UTF-8')as f:opened_ids_list:list[str]=f.read().split(sep='\n');
    with open(file=closed_ids_txt_filename,mode='rt',encoding='UTF-8')as f:closed_ids_list:list[str]=f.read().split(sep='\n');
    for i in range(n_samples_opened):opened_ids[i]=opened_ids_list[i];
    for i in range(n_samples_closed):closed_ids[i]=closed_ids_list[i];
    print(f'opened_ids.shape: {opened_ids.shape}, opened_ids.dtype: {opened_ids.dtype}, opened_ids[0]: {opened_ids[0]}, opened_ids[{n_samples_opened-1}]: {opened_ids[n_samples_opened-1]}');
    print(f'closed_ids.shape: {closed_ids.shape}, closed_ids.dtype: {closed_ids.dtype}, closed_ids[0]: {closed_ids[0]}, closed_ids[{n_samples_closed-1}]: {closed_ids[n_samples_closed-1]}');
    del_opened_parquet_dataframes:bool=str_or_bool_to_bool(s=conf_dict['del_opened_parquet_dataframes']);
    if save_opened_parquet==True:#Сначала запишем файл до трансформаций (применение функций, попарные произведения, ...), затем после
        feature_cols:list[str]=[f'X{i}'for i in range(n_features_opened)];#Формируем имена колонок для признаков
        opened_ids_data_target_df:pd.DataFrame=pd.DataFrame(data=opened_data_all_features,columns=feature_cols);
        opened_ids_data_target_df.insert(loc=0,column='id',value=opened_ids);
        opened_ids_data_target_df['Y']=opened_target;
        parquet_filename:str=conf_dict['data_files_names']['opened_ids_data_target_before_transforms_parquet'];
        if os.path.exists(path=parquet_filename):
            print(f'Файл parquet (до трансформаций) уже существует: {parquet_filename}, он не был перезаписан');
        else:
            opened_ids_data_target_df.to_parquet(path=parquet_filename,index=False);
            print(f'Файл parquet (до трансформаций) успешно сохранён: {parquet_filename}');
        if del_opened_parquet_dataframes==True:del opened_ids_data_target_df;
    if save_opened_parquet==True:#Теперь запишем только id и значения target (без data)
        opened_ids_target_df:pd.DataFrame=pd.DataFrame({'id':opened_ids,'Y':opened_target});
        parquet_filename:str=conf_dict['data_files_names']['opened_ids_target_parquet'];
        if os.path.exists(path=parquet_filename):
            print(f'Файл parquet (содержащий только id и target) уже существует: {parquet_filename}, он не был перезаписан');
        else:
            opened_ids_target_df.to_parquet(path=parquet_filename,index=False);
            print(f'Файл parquet (содержащий только id и target) успешно сохранён: {parquet_filename}');
        #opened_ids_target_df и так маленький, его можно не очищать
    print(f'ПЕРЕД ТРАНСФОРМАЦИЯМИ ПРИЗНАКОВ:');print_loaded_data_info();#Далее выполняются трансформации данных
    transform_1:str=conf_dict['data_transformation']['transform_1'];#Трансформация 1
    if transform_1=='pairwise_multiplications':
        opened_data_all_features=data_transformation_pairwise_multiplications(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_pairwise_multiplications(feature_matrix=closed_data_all_features);
    elif transform_1=='add_functions':
        opened_data_all_features=data_transformation_add_functions(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_add_functions(feature_matrix=closed_data_all_features);
    elif transform_1=='keep_same':
        opened_data_all_features=data_transformation_keep_same(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_keep_same(feature_matrix=closed_data_all_features);
    n_samples_opened:int=opened_data_all_features.shape[0];n_features_opened:int=opened_data_all_features.shape[1];n_samples_closed:int=closed_data_all_features.shape[0];n_features_closed:int=closed_data_all_features.shape[1];
    print(f'ПОСЛЕ ТРАНСФОРМАЦИИ ПРИЗНАКОВ НОМЕР 1 ({transform_1}):');print_loaded_data_info();
    transform_2:str=conf_dict['data_transformation']['transform_2'];#Трансформация 2
    if transform_2=='pairwise_multiplications':
        opened_data_all_features=data_transformation_pairwise_multiplications(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_pairwise_multiplications(feature_matrix=closed_data_all_features);
    elif transform_2=='add_functions':
        opened_data_all_features=data_transformation_add_functions(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_add_functions(feature_matrix=closed_data_all_features);
    elif transform_2=='keep_same':
        opened_data_all_features=data_transformation_keep_same(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_keep_same(feature_matrix=closed_data_all_features);
    n_samples_opened:int=opened_data_all_features.shape[0];n_features_opened:int=opened_data_all_features.shape[1];n_samples_closed:int=closed_data_all_features.shape[0];n_features_closed:int=closed_data_all_features.shape[1];
    print(f'ПОСЛЕ ТРАНСФОРМАЦИИ ПРИЗНАКОВ НОМЕР 2 ({transform_2}):');print_loaded_data_info();
    transform_3:str=conf_dict['data_transformation']['transform_3'];#Трансформация 3
    if transform_3=='pairwise_multiplications':
        opened_data_all_features=data_transformation_pairwise_multiplications(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_pairwise_multiplications(feature_matrix=closed_data_all_features);
    elif transform_3=='add_functions':
        opened_data_all_features=data_transformation_add_functions(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_add_functions(feature_matrix=closed_data_all_features);
    elif transform_3=='keep_same':
        opened_data_all_features=data_transformation_keep_same(feature_matrix=opened_data_all_features);
        closed_data_all_features=data_transformation_keep_same(feature_matrix=closed_data_all_features);
    n_samples_opened:int=opened_data_all_features.shape[0];n_features_opened:int=opened_data_all_features.shape[1];n_samples_closed:int=closed_data_all_features.shape[0];n_features_closed:int=closed_data_all_features.shape[1];
    print(f'ПОСЛЕ ТРАНСФОРМАЦИИ ПРИЗНАКОВ НОМЕР 3 ({transform_3}):');print_loaded_data_info();
    if save_opened_parquet==True:#Теперь запишем файл после трансформаций (применение функций, попарные произведения, ...)
        feature_cols:list[str]=[f'X{i}'for i in range(n_features_opened)];#Формируем имена колонок для признаков
        opened_ids_data_target_df:pd.DataFrame=pd.DataFrame(data=opened_data_all_features,columns=feature_cols);
        opened_ids_data_target_df.insert(loc=0,column='id',value=opened_ids);
        opened_ids_data_target_df['Y']=opened_target;
        parquet_filename:str=conf_dict['data_files_names']['opened_ids_data_target_after_transforms_parquet'];
        if os.path.exists(path=parquet_filename):
            print(f'Файл parquet (после трансформаций) уже существует: {parquet_filename}, он не был перезаписан');
        else:
            opened_ids_data_target_df.to_parquet(path=parquet_filename,index=False);
            print(f'Файл parquet (после трансформаций) успешно сохранён: {parquet_filename}');
        if del_opened_parquet_dataframes==True:del opened_ids_data_target_df;
    
    if save_opened_and_closed_features_csvs==True:#Сохранение csv файлов со значениями всех признаков для открытых и закрытых данных
        opened_data_all_features_csv_file_name:str=f'all_features_opened_data.csv';
        closed_data_all_features_csv_file_name:str=f'all_features_closed_data.csv';
        opened_buf_str_list:list[str]=[];
        closed_buf_str_list:list[str]=[];
        buf_s:str='id,'+','.join(['X'+str(i)for i in range(n_features_opened)])+',Y'+'\n';#Добавление заголовков csv файлов
        opened_buf_str_list.append(buf_s);
        buf_s:str='id,'+','.join(['X'+str(i)for i in range(n_features_closed)])+',Y'+'\n';
        closed_buf_str_list.append(buf_s);
        for sample_index in range(n_samples_opened):
            buf_s:str=opened_ids[sample_index]+','+','.join([str(float(opened_data_all_features[sample_index][feature_index]))for feature_index in range(n_features_opened)])+','+str(float(opened_target[sample_index]))+'\n';
            opened_buf_str_list.append(buf_s);
        for sample_index in range(n_samples_closed):
            buf_s:str=closed_ids[sample_index]+','+','.join([str(float(closed_data_all_features[sample_index][feature_index]))for feature_index in range(n_features_closed)])+','+str(float(closed_target[sample_index]))+'\n';
            closed_buf_str_list.append(buf_s);
        if os.path.exists(path=opened_data_all_features_csv_file_name):
            print(f'csv файл со всеми признаками opened_data уже существует: {opened_data_all_features_csv_file_name}, он не был перезаписан');
        else:
            with open(file=opened_data_all_features_csv_file_name,mode='wt',encoding='UTF-8')as f_csv:f_csv.writelines(opened_buf_str_list);
            print(f'csv файл со всеми признаками opened_data успешно сохранён: {opened_data_all_features_csv_file_name}');
        if os.path.exists(path=closed_data_all_features_csv_file_name):
            print(f'csv файл со всеми признаками opened_data уже существует: {closed_data_all_features_csv_file_name}, он не был перезаписан');
        else:
            with open(file=closed_data_all_features_csv_file_name,mode='wt',encoding='UTF-8')as f_csv:f_csv.writelines(closed_buf_str_list);
            print(f'csv файл со всеми признаками closed_data успешно сохранён: {closed_data_all_features_csv_file_name}');
    return opened_data_all_features,opened_target,opened_ids,closed_data_all_features,closed_target,closed_ids;

def create_log_files()->None:
    """Функция создаёт log файлы (если они не существуют)"""
    if pathlib.Path('log_pipelines.txt').exists()==False:#Создать файл log_pipelines.txt если его не существует
        with open(file='log_pipelines.txt',mode='wt',encoding='UTF-8')as f_log:pass;
    with open(file='log_pipelines.txt',mode='at',encoding='UTF-8')as f_log:
        print(f'{'='*32}\nРезультаты запуска скрипта с конфигурацией: \n{json.dumps(obj=conf_dict,ensure_ascii=False,indent=4,sort_keys=False)}\n',file=f_log);
    if pathlib.Path('log_pipelines.csv').exists()==False:#Создать файл log_pipelines.csv если его не существует и заполнить его заголовок
        with open(file='log_pipelines.csv',mode='wt',encoding='UTF-8')as f_log:
            header_str:str=f'pipeline_id,n_features_all,n_features_selected_randomly,use_imputer,imputer_type,use_var_thresholder,var_thresholder_type,use_scaler,scaler_type,use_feature_selector,feature_selector_type,fs_score_func_type,fs_estimator_type,model_type,score_type,score_valid_mean,score_valid_std,score_test,dt_pipe_start_str,seconds_processing,pipeline_file_size';
            print(header_str,file=f_log);
    if pathlib.Path('log_results.txt').exists()==False:#Создать файл log_results.txt если его не существует
        with open(file='log_results.txt',mode='wt',encoding='UTF-8')as f_log:pass;

def run_one_pipeline_experiment_v1(num_features_select_from_all_min:int=5,num_features_select_from_all_max:int=50,randomly_selected_indexes:list[int]=None,problem_type:str='regression',task_output:str='mono_output',score_type:str='mean_squared_error',fbeta_score_beta:float=1.0,d2_pinball_score_alpha:float=0.5,d2_tweedie_score_power:float=0.0,mean_pinball_loss_alpha:float=0.5,mean_tweedie_deviance_power:float=0.0,use_imputer_probability:float=0.95,imputer_type:str=None,imputer_hyperparams:dict=None,use_var_thresholder_probability:float=0.95,var_thresholder_type:str=None,var_thresholder_hyperparams:dict=None,use_feature_selector_probability:float=0.7,feature_selector_type:str=None,prefered_feature_selector_types:list[str]=None,feature_selector_hyperparams:dict=None,fs_score_func_type:str=None,fs_estimator_type:str=None,prefered_fs_estimator_types:list[str]=None,fs_estimator_hyperparams:dict=None,use_scaler_probability:float=0.9,scaler_type:str=None,prefered_scaler_types:list[str]=None,scaler_hyperparams:dict=None,model_type:str=None,prefered_model_types:list[str]=None,model_hyperparams:dict=None,num_folds:int=10,score_valid_min_threshold:float=None,score_valid_max_threshold:float=None,non_negative_y_guarantee:bool=False,use_only_linear_models:bool=False,use_only_models_with_predict_proba:bool=True,n_cpu_cores:int=-1)->str:
    """
    Запуск одного эксперимента со случайным выбором пайплайна, его компонентов и их гиперпараметров\n
    problem_type='regression'|'classification_binary'|'classification_multiclass'\n
    task_output='mono_output'|'multi_output'\n
    score_type for classification (from sklearn.metrics import ...):\n
    accuracy_score,auc,average_precision_score,balanced_accuracy_score,brier_score_loss\n
    cohen_kappa_score,dcg_score,f1_score,fbeta_score,hamming_loss,hinge_loss,jaccard_score\n
    log_loss,matthews_corrcoef,ndcg_score,precision_score,recall_score,roc_auc_score,zero_one_loss\n
    score_type for regression (from sklearn.metrics import ...):\n
    d2_absolute_error_score,d2_pinball_score,d2_tweedie_score,explained_variance_score,max_error\n
    mean_absolute_percentage_error,mean_gamma_deviance,mean_pinball_loss,mean_poisson_deviance\n
    mean_squared_error,mean_squared_log_error,mean_tweedie_deviance,median_absolute_error\n
    r2_score,root_mean_squared_error,root_mean_squared_log_error\n\n\n

    Сначала из матриц признаков opened_data_all_features и closed_data_all_features выбираются
    некоторым способом num_features_select_from_all столбцов (признаков), из которых составляются
    матрицы opened_data и closed_data соответственно\n
    Если num_features_select_from_all=0, то используются все признаки.
    Значение num_features_select_from_all определяется через num_features_select_from_all_min
    и num_features_select_from_all_max.\n
    Если в качестве параметра randomly_selected_indexes передаётся список, то отбираются именно эти признаки, а не случайные.\n

    use_only_linear_models:bool=False - если True, то выбираются только те пайплайны, у моделей которых есть атрибуты 'coef_' и 'intercept_'\n
    use_only_models_with_predict_proba:bool=True - если True, то выбираются только те пайплайны, у моделей которых есть атрибут (метод) predict_proba (возвращающий вероятности), используется только для классификации (для регрессии игнорируется)\n

    use_scaler_probability:float=0.9,scaler_type:str=None,scaler_hyperparams:dict=None - эти параметры позволяют использовать
    различные варианты scaler (или не использовать scaler вообще если use_scaler=False)
        
    fs_score_func_type - это тип функции, которая используется для feature_selector, если он равен GenericUnivariateSelect,SelectFdr,SelectFpr,
    SelectFwe,SelectKBest,SelectPercentile. Возможные значения: chi2,f_classif,f_regression,mutual_info_classif,mutual_info_regression,
    r_regression.

    Пайплайн: imputer -> var_thresholder -> scaler -> feature_selector -> model
    """
    dt_pipe_start:datetime.datetime=datetime.datetime.now();#Для лога (когда этот пайплайн запущен)
    dt_pipe_start_str:str=dt_pipe_start.strftime(format='%Y-%m-%d_%H-%M-%S');
    seconds_pipe_start:float=time.time();#Для лога (чтобы вычислить время обработки этого пайплайна)
    print(f'Функция run_one_pipeline_experiment_v1 вызвана с параметрами: {locals()}');
    pipeline_id:str=''.join(random.choices(population=string.ascii_uppercase+string.digits,k=16));
    error_str:str='PIPELINE_ERROR';
    use_imputer:bool=true_with_prob(p=use_imputer_probability);#Использовать ли imputer в этом эксперименте
    use_var_thresholder:bool=true_with_prob(p=use_var_thresholder_probability);#Использовать ли var_thresholder в этом эксперименте
    use_scaler:bool=true_with_prob(p=use_scaler_probability);#Использовать ли scaler в этом эксперименте
    use_feature_selector:bool=true_with_prob(p=use_feature_selector_probability);#Использовать ли feature_selector в этом эксперименте
    copy_data_arrays:bool=conf_dict['copy_data_arrays'];#При выполнении преобразований (imputer, var_thresholder, scaler, feature_selector) каждый раз создавать
    #копию массива или перезаписывать исходный массив (если каждый раз создавать копию, то тратится больше памяти и при некоторых значениях n_samples, n_features
    #её может не хватить, тогда будет примерно так:
    #numpy._core._exceptions._ArrayMemoryError: Unable to allocate 260. GiB for an array with shape (22400, 1556730) and data type float64)
    # 1. Загрузка ВСЕХ данных (открытых и закрытых) и отбор num_features_select_from_all признаков из всех
    #Загрузка выполняется отдельно, так как:
    #1) Если эксперимент повторяется много раз, загружать данные каждый раз неэффективно по времени
    #2) Данные могут быть представлены в разных форматах (csv,json,npy,...), поэтому обработку каждого из этих форматов лучше
    #выполнять отдельно в своей функции (load_data_from_npy, load_data_from_csv, load_data_from_json, ...)
    #Отбор num_features_select_from_all признаков:
    n_samples_opened:int=opened_data_all_features.shape[0];
    n_samples_closed:int=closed_data_all_features.shape[0];
    n_features_all:int=opened_data_all_features.shape[1];#Равно closed_data_all_features.shape[1]    
    if randomly_selected_indexes is None:#Если индексы не заданы, то они выбираются случайным образом
        #ЕСЛИ ИНДЕКСЫ ВЫБИРАЮТСЯ СЛУЧАЙНЫМ ОБРАЗОМ, ТО НИКАКАЯ ИНФОРМАЦИЯ О ТЕСТОВОЙ ВЫБОРКЕ НЕ ИСПОЛЬЗУЕТСЯ => УТЕЧКИ ДАННЫХ НЕТ
        if num_features_select_from_all_min==num_features_select_from_all_max==0:num_features_select_from_all=0;
        else:num_features_select_from_all:int=random.randint(a=num_features_select_from_all_min,b=num_features_select_from_all_max);
        n_features_selected_randomly:int=num_features_select_from_all;
        if num_features_select_from_all==0:#Если num_features_select_from_all=0 и индексы не заданы, то используются все признаки
            all_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
            #ЕСЛИ ВЫБИРАЮТСЯ ВСЕ ИНДЕКСЫ, ТО НИКАКАЯ ИНФОРМАЦИЯ О ТЕСТОВОЙ ВЫБОРКЕ НЕ ИСПОЛЬЗУЕТСЯ => УТЕЧКИ ДАННЫХ НЕТ
            randomly_selected_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
        else:#Если num_features_select_from_all!=0 и индексы не заданы, то выполняется отбор случайного множества признаков из всех        
            all_indexes:list[int]=[i for i in range(n_features_all)];#Все индексы 0..n_features_all-1
            for i in range(10):random.shuffle(x=all_indexes);#Shuffle list x in place, and return None.
            randomly_selected_indexes:list[int]=[all_indexes[i] for i in range(num_features_select_from_all)];#Индексы выбранных признаков
            #(то есть первые num_features_select_from_all индексов из всех n_features_all индексов, перемешанных случайным образом)
            randomly_selected_indexes.sort();#Эта сортировка по идее ни на что не влияет, но так просто удобнее для человека
            #Использование списка randomly_selected_indexes и его сохранение в лог делает эсперименты воспроизводимыми
    else:#Если индексы заданы, то выбираются столбцы именно с этими индексами (ничего случайным образом не выбирается, ВНЕ зависимости
        #от num_features_select_from_all_min и num_features_select_from_all_max)
        n_features_selected_randomly:int=len(randomly_selected_indexes);
        #ЕСЛИ ИНДЕКСЫ ЗАДАНЫ, ТО ЭТО ИСПОЛЬЗУЕТСЯ ДЛЯ ВОСПРОИЗВОДИМОСТИ ЭКСПЕРИМЕНТОВ, ПОЭТОМУ УТЕЧКИ ДАННЫХ ТОЖЕ НЕТ
    #Для создания и заполнения массивов opened_data и closed_data используется один и тот же код вне зависимости от переданных
    #значений randomly_selected_indexes, num_features_select_from_all_min и num_features_select_from_all_max
    opened_data:np.ndarray=np.zeros((n_samples_opened,n_features_selected_randomly));#Создание новых массивов
    closed_data:np.ndarray=np.zeros((n_samples_closed,n_features_selected_randomly));
    for col_idx,feature_idx in enumerate(randomly_selected_indexes):#Копирование столбцов с выбранными индексами из исходных массивов
        opened_data[:,col_idx]=opened_data_all_features[:,feature_idx];
        closed_data[:,col_idx]=closed_data_all_features[:,feature_idx];
    #Все этапы пайплайна (imputer, var_thresholder, scaler, feature_selector, model) применяются к одному и тому же набору признаков (который
    #выбран в соответствии со списком randomly_selected_indexes)

    # 2. Разделение на train+valid (для cross_valid) и final_test (для оценки на holdout)
    split_random_state:int=int(time.time()*(10**9))%(2**32);#Количество наносекунд с начала эпохи Unix -> [0, 4294967295]
    hyperparam_random_state:int=int(random.uniform(a=0.0,b=1e20))%(2**32);
    score_func_random_state:int=int(random.uniform(a=0.0,b=1e20))%(2**32);
    opened_data_len:int=opened_data.shape[0];
    X_train_cv:np.ndarray;X_test_final:np.ndarray;y_train_cv:np.ndarray;y_test_final:np.ndarray;ids_train_cv:np.ndarray;ids_test_final:np.ndarray;
    #Проблема утечки данных в случае добавления аугментаций. Например, bike_82_aug2.jpg в ids_train_cv, bike_82_aug3.jpg в ids_test_final, а эти две картинки
    #между собой очень похожи => утечка данных!!! Такая же проблема внутри кросс-валидации. Если аугментации не добавляются, то проблемы нет.
    #Эту проблему нужно решить.

    #Это тот код, который у меня был (он работает правильно при отсутствии аугментаций, но при наличии аугментаций приводит к утечке данных и завышению метрик на test_final):
    """
    if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
        X_train_cv,X_test_final,y_train_cv,y_test_final,ids_train_cv,ids_test_final=train_test_split(
        opened_data,opened_target,opened_ids,test_size=conf_dict['holdout_size'],random_state=split_random_state,stratify=opened_target);
    elif problem_type=='regression':
        X_train_cv,X_test_final,y_train_cv,y_test_final,ids_train_cv,ids_test_final=train_test_split(
        opened_data,opened_target,opened_ids,test_size=conf_dict['holdout_size'],random_state=split_random_state);
    """
    
    #Сюда вставять новые коды от DeepSeek/QWEN для разделения на train_cv/test_final с учётом групп картинок, полученных из одной с использованием аугментаций:
    #================================Начало кода деления на train_cv/test_final с учётом групп аугментации================================
    #Сначала нужно удостовериться, что группы аугментации точно указаны (то есть в id каждого объекта есть строка "_aug").
    opened_ids_joined_str:str=' | '.join(list(opened_ids));#Использован символ |, так как он НЕ может использоваться в названии файла
    num_opened_ids_with_aug_substr:int=opened_ids_joined_str.count(f'_aug');
    if num_opened_ids_with_aug_substr<opened_ids.shape[0]:
        raise(AssertionError(f'Среди {opened_ids.shape[0]} образцов opened_data есть как минимум {opened_ids.shape[0]-num_opened_ids_with_aug_substr} тех, у которых в id нет строки [_aug]. Если аугментации не используются, то к каждому id из файла [{conf_dict['data_files_names']['opened_ids_txt']}] нужно добавить строку [_aug0] (перед расширением файла)!!!'));
    first_id:str=str(opened_ids[0]);                        # например "bike_1_aug0.jpg"
    base_name:str=first_id.rsplit(sep='_aug',maxsplit=1)[0];# "bike_1"
    num_lines_start_with_base_name:int=0;    # подсчитываем, сколько следующих строк начинаются с base_name
    for i in range(len(opened_ids)):
        if str(opened_ids[i]).startswith(base_name):num_lines_start_with_base_name=num_lines_start_with_base_name+1;
        else:break;#Тут можно делать break, так как все id результатов нескольких аугментаций одного исходного изображения идут подряд
    n_augs:int=num_lines_start_with_base_name;
    n_groups_to_print:int=conf_dict['n_groups_to_print'];#Сколько групп, индексов и id выводить в консоль (если выводить все, то их слишком много)
    print(f'n_augs (количество аугментаций каждого исходного изображения [нулевая аугментация - это копия исходного объекта]): {n_augs}');
    n_groups:int=opened_data.shape[0]//n_augs;#Количество групп (исходных изображений) [все образцы opened_data - это n_groups групп по n_augs образцов в каждой]
    print(f'n_groups (количество групп изображений по n_augs={n_augs} аугментаций в каждом): {n_groups}');
    print(f'Всего образцов (n_augs*n_groups): {n_augs}*{n_groups}={n_augs*n_groups}, количество образцов в opened_data: {opened_data.shape[0]}');
    if n_augs*n_groups!=opened_data.shape[0]:
        raise(AssertionError(f'n_augs={n_augs}, n_groups={n_groups}, n_augs*n_groups={n_augs*n_groups}, количество образцов в opened_data: {opened_data.shape[0]}\nПроизведение n_augs*n_groups ДОЛЖНО БЫТЬ РАВНО количеству образцов в opened_data!!!'));
    group_indices:np.ndarray=np.arange(stop=n_groups);#Индексы групп и соответствующие им метки
    group_targets:np.ndarray=opened_target[0::n_augs];#берём метку первого [технически нулевого] элемента каждой группы (то есть используется срез, аналогично list)
    # Разделение индексов групп на train/valid и final test
    if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
        groups_train_cv,groups_test_final,_,_=train_test_split(group_indices,group_targets,test_size=conf_dict['holdout_size'],random_state=split_random_state,stratify=group_targets);
    elif problem_type=='regression':
        groups_train_cv,groups_test_final=train_test_split(group_indices,test_size=conf_dict['holdout_size'],random_state=split_random_state);
    def group_indices_to_sample_indices(group_indices:np.ndarray,n_augs:int)->list:
        """Функция преобразования индексов групп в индексы образцов"""
        sample_indices:list[int]=[];
        for g in group_indices:
            start:int=g*n_augs;
            sample_indices.extend(range(start,start+n_augs));
        return sample_indices;
    indices_train_cv:np.ndarray=np.array(object=group_indices_to_sample_indices(group_indices=groups_train_cv,n_augs=n_augs),dtype=np.int32);
    indices_test_final:np.ndarray=np.array(object=group_indices_to_sample_indices(group_indices=groups_test_final,n_augs=n_augs),dtype=np.int32);
    X_train_cv:np.ndarray=opened_data[indices_train_cv];
    y_train_cv:np.ndarray=opened_target[indices_train_cv];
    ids_train_cv:np.ndarray=opened_ids[indices_train_cv];
    X_test_final:np.ndarray=opened_data[indices_test_final];
    y_test_final:np.ndarray=opened_target[indices_test_final];    
    ids_test_final:np.ndarray=opened_ids[indices_test_final];
    if conf_dict['save_sample_partition']==True:#Для сохранения информации о том, в какую часть (cv_fold0, cv_fold1, ..., cv_fold9, test_final) попал каждый образец
        max_part_name_len:int=conf_dict['max_part_name_len'];#Максимальная длина метки того, к какой из частей относится образец из opened_data
        part_assignment:np.ndarray=np.full(shape=opened_data_len,fill_value='',dtype=f'U{str(max_part_name_len)}');#Массив для хранения меток
        part_assignment[indices_test_final]='test_final';
    #================================Конец кода деления на train_cv/test_final с учётом групп аугментации================================
    #Вывод информации о результатах разделения остаётся без изменений:
    print(f'Размеры X_train_cv (train+cross_valid): {X_train_cv.shape}, размеры X_test_final (holdout): {X_test_final.shape}');
    print(f'Доля X_train_cv от opened_data: {X_train_cv.shape[0]/opened_data_len}, доля X_test_final от opened_data: {X_test_final.shape[0]/opened_data_len}');#0.8 0.2
    print(f'Размеры y_train_cv (train+cross_valid): {y_train_cv.shape}, размеры y_test_final (holdout): {y_test_final.shape}');
    print(f'Доля y_train_cv от opened_data: {y_train_cv.shape[0]/opened_data_len}, доля y_test_final от opened_data: {y_test_final.shape[0]/opened_data_len}');#0.8 0.2
    if conf_dict['print_ids_lists']==True:
        print(f'len(opened_ids): {len(opened_ids)}, list(opened_ids) [первые {n_groups_to_print} групп]: {[str(id)for id in opened_ids][:n_groups_to_print*n_augs]}');
        print(f'len(ids_train_cv): {len(ids_train_cv)}, list(ids_train_cv) [первые {n_groups_to_print} групп]: {[str(id)for id in ids_train_cv][:n_groups_to_print*n_augs]}');
        print(f'len(indices_train_cv): {len(indices_train_cv)}, indices_train_cv [первые {n_groups_to_print} групп]: {[str(indice)for indice in indices_train_cv][:n_groups_to_print*n_augs]}');
        print(f'len(groups_train_cv): {len(groups_train_cv)}, groups_train_cv [первые {n_groups_to_print} групп]: {[str(group)for group in groups_train_cv][:n_groups_to_print]}');
        print(f'len(ids_test_final): {len(ids_test_final)}, list(ids_test_final) [первые {n_groups_to_print} групп]: {[str(id)for id in ids_test_final][:n_groups_to_print*n_augs]}');
        print(f'len(indices_test_final): {len(indices_test_final)}, indices_test_final [первые {n_groups_to_print} групп]: {[str(indice)for indice in indices_test_final][:n_groups_to_print*n_augs]}');
        print(f'len(groups_test_final): {len(groups_test_final)}, groups_test_final [первые {n_groups_to_print} групп]: {[str(group)for group in groups_test_final][:n_groups_to_print]}');
    if conf_dict['save_ids_lists_to_txt']==True:
        buf_lst_s:str=f"opened_ids: [{','.join([str(id)for id in opened_ids])}]";
        with open(file=f'ids_opened_all.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);
        buf_lst_s:str=f"ids_train_cv: [{','.join([str(id)for id in ids_train_cv])}]";
        with open(file=f'ids_train_cv_list.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);
        buf_lst_s:str=f"indices_train_cv: [{','.join([str(indice)for indice in indices_train_cv])}]";
        with open(file=f'indices_train_cv_list.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);
        buf_lst_s:str=f"groups_train_cv: [{','.join([str(group)for group in groups_train_cv])}]";
        with open(file=f'groups_train_cv_list.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);
        buf_lst_s:str=f"ids_test_final: [{','.join([str(id)for id in ids_test_final])}]";
        with open(file=f'ids_test_final_list.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);
        buf_lst_s:str=f"indices_test_final: [{','.join([str(indice)for indice in indices_test_final])}]";
        with open(file=f'indices_test_final_list.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);
        buf_lst_s:str=f"groups_test_final: [{','.join([str(group)for group in groups_test_final])}]";
        with open(file=f'groups_test_final_list.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);

    #3.1. Установка типа и гиперпараметров imputer
    if use_imputer==True:
        if imputer_type is None:#Выбор случайного imputer_type из списка:
            imputer_types:list[str]=['KNNImputer','SimpleImputer'];
            imputer_type=random.choice(seq=imputer_types);
        if imputer_hyperparams is None:
            imputer_hyperparams:dict[str,int|float|bool|str];
            if imputer_type=='KNNImputer':imputer_hyperparams={'n_neighbors':random.randint(a=1,b=15),'weights':random.choice(seq=['uniform','distance']),'metric':'nan_euclidean','copy':True,'add_indicator':random.choice(seq=[True,False]),'keep_empty_features':random.choice(seq=[True,False])};
            elif imputer_type=='SimpleImputer':imputer_hyperparams={'strategy':random.choice(seq=['mean','median','most_frequent','constant']),'fill_value':None,'copy':True,'add_indicator':random.choice(seq=[True,False]),'keep_empty_features':random.choice(seq=[True,False])};
    else:
        print(f'Выбрано use_imputer==False, imputer не применяется');
        imputer_hyperparams=None;imputer_type=None;

    #3.2. Установка типа и гиперпараметров var_thresholder
    if use_var_thresholder==True:
        if var_thresholder_type is None:#Выбор случайного var_thresholder_type из списка:
            var_thresholder_types:list[str]=['VarianceThreshold'];
            var_thresholder_type=random.choice(seq=var_thresholder_types);
        if var_thresholder_hyperparams is None:
            var_thresholder_hyperparams:dict[str,int|float|bool|str];
            if var_thresholder_type=='VarianceThreshold':var_thresholder_hyperparams={'threshold':10**random.uniform(a=-15,b=1)};
    else:
        print(f'Выбрано use_var_thresholder==False, var_thresholder не применяется');
        var_thresholder_hyperparams=None;var_thresholder_type=None;

    #3.3. Установка типа и гиперпараметров scaler
    if use_scaler==True:
        if scaler_type is None:#Выбор случайного scaler из списка:
            scaler_types:list[str]=conf_dict['pipeline_params']['all_scaler_types'];
            if prefered_scaler_types is not None:scaler_types=list(set(scaler_types).intersection(set(prefered_scaler_types)));
            scaler_type:str=random.choice(seq=scaler_types);
        if scaler_hyperparams is None:
            scaler_hyperparams:dict[str,int|float|bool|str];
            if scaler_type=='MaxAbsScaler':scaler_hyperparams={'copy':True};
            elif scaler_type=='MinMaxScaler':scaler_hyperparams={'feature_range':(0,1),'copy':True,'clip':False};
            elif scaler_type=='RobustScaler':scaler_hyperparams={'with_centering':true_with_prob(p=0.85),'with_scaling':true_with_prob(p=0.85),'quantile_range':(random.uniform(a=0.0,b=40.0),random.uniform(a=60.0,b=100.0)),'copy':True,'unit_variance':true_with_prob(p=0.1)};
            elif scaler_type=='StandardScaler':scaler_hyperparams={'copy':True,'with_mean':true_with_prob(p=0.85),'with_std':true_with_prob(p=0.85)};
    else:
        print(f'Выбрано use_scaler==False, scaler не применяется');
        scaler_hyperparams=None;scaler_type=None;

    #3.4. Установка типа и гиперпараметров feature_selector
    #Фактически все feature_selector делятся на 2 вида:
    #1) которые используют score_func (значимость признака оценивается по значению этой функции между этим признаком и target)
    #2) которые используют estimator (значимость признака оценивается по влиянию этого признака на предсказания модели)
    #3 [условно]) VarianceThreshold (условно, так как он используется отдельно перед scaler)
    if use_feature_selector==True:
        if feature_selector_type is None:#Выбор случайного feature_selector из списка:
            feature_selector_types_all:list[str]=conf_dict['pipeline_params']['feature_selector_types_all'];
            feature_selector_types_estimator_rfe:list[str]=conf_dict['pipeline_params']['feature_selector_types_estimator_rfe'];
            feature_selector_types_estimator_not_rfe:list[str]=conf_dict['pipeline_params']['feature_selector_types_estimator_not_rfe'];
            feature_selector_types_estimator_all:list[str]=feature_selector_types_estimator_rfe+feature_selector_types_estimator_not_rfe;
            feature_selector_types_score_func_based:list[str]=conf_dict['pipeline_params']['feature_selector_types_score_func_based'];
            if prefered_feature_selector_types is not None:feature_selector_types_all=list(set(feature_selector_types_all).intersection(set(prefered_feature_selector_types)));
            feature_selector_type:str=random.choice(seq=feature_selector_types_all);
        if feature_selector_hyperparams is None:
            feature_selector_hyperparams:dict[str,int|float|bool|str];
            if feature_selector_type in feature_selector_types_estimator_rfe:#Нужно выбрать тип и гиперпараметры для estimator
                if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):fs_estimator_types:list[str]=['LinearSVC','NuSVC','SVC'];
                elif problem_type=='regression':fs_estimator_types:list[str]=['LinearSVR','NuSVR','SVR'];
            elif feature_selector_type in feature_selector_types_estimator_not_rfe:#Нужно выбрать тип и гиперпараметры для estimator
                if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
                    fs_estimator_types:list[str]=['LogisticRegression','PassiveAggressiveClassifier','Perceptron','RidgeClassifier','SGDClassifier'];
                elif problem_type=='regression':
                    fs_estimator_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','OrthogonalMatchingPursuit','ARDRegression','BayesianRidge','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor'];
            if feature_selector_type in feature_selector_types_estimator_all:#Если есть estimator, то нужно выбрать для него гиперпараметры
                if fs_estimator_type is None:
                    if prefered_fs_estimator_types is not None:fs_estimator_types=list(set(fs_estimator_types).intersection(set(prefered_fs_estimator_types)));
                    fs_estimator_type:str=random.choice(seq=fs_estimator_types);
                    if fs_estimator_hyperparams is None:
                        fs_estimator_hyperparams:dict[str,int|float|bool|str];
                        if fs_estimator_type=='LogisticRegression':fs_estimator_hyperparams={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'dual':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=0),'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']),'max_iter':random.randint(a=50,b=500),'warm_start':random.choice(seq=[True,False]),'l1_ratio':random.uniform(a=0.0,b=1.0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='PassiveAggressiveClassifier':fs_estimator_hyperparams={'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=3000),'tol':10**random.uniform(a=-7,b=1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'shuffle':random.choice(seq=[True,False]),'loss':random.choice(seq=['hinge','squared_hinge']),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='Perceptron':fs_estimator_hyperparams={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-12,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='RidgeClassifier':fs_estimator_hyperparams={'alpha':10**random.uniform(a=-2,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-8,b=0),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='SGDClassifier':fs_estimator_hyperparams={'loss':random.choice(seq=['hinge','log_loss','modified_huber','squared_hinge','perceptron','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-8,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'epsilon':10**random.uniform(a=-9,b=3),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=1),'power_t':random.uniform(a=-5,b=6),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                        elif fs_estimator_type=='LinearRegression':fs_estimator_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-9,b=-3),'positive':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='Ridge':fs_estimator_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=500,b=20000),'tol':10**random.uniform(a=-9,b=-0.1),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':bool(random.randint(a=0,b=1)),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='SGDRegressor':fs_estimator_hyperparams={'loss':random.choice(seq=['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-9,b=1.0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=20000),'tol':10**random.uniform(a=-6,b=0),'epsilon':10**random.uniform(a=-3,b=1),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=0),'power_t':random.uniform(a=-100,b=100),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0,b=1),'n_iter_no_change':random.randint(a=2,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='ElasticNet':fs_estimator_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'l1_ratio':random.uniform(a=0,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-7,b=-1),'warm_start':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='Lars':fs_estimator_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'n_nonzero_coefs':random.randint(a=10,b=100),'eps':10**random.uniform(a=-5,b=-1),'fit_path':random.choice(seq=[True,False]),'jitter':10**random.uniform(a=-9,b=-1),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='Lasso':fs_estimator_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(100,2000),'tol':10**random.uniform(a=-8,b=-1),'warm_start':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='LassoLars':fs_estimator_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=1500),'eps':10**random.uniform(a=-10,b=-5),'fit_path':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'jitter':10**random.uniform(a=-9,b=-1),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='LassoLarsIC':fs_estimator_hyperparams={'criterion':random.choice(seq=['aic','bic']),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=1500),'eps':10**random.uniform(a=-16,b=-10),'positive':random.choice(seq=[True,False]),'noise_variance':10**random.uniform(a=-5,b=-1)};
                        elif fs_estimator_type=='OrthogonalMatchingPursuit':fs_estimator_hyperparams={'n_nonzero_coefs':None if true_with_prob(p=0.5)==True else random.randint(a=5,b=50),'tol':None if true_with_prob(p=0.5)==True else 10**random.uniform(a=-2,b=2),'fit_intercept':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='ARDRegression':fs_estimator_hyperparams={'max_iter':random.randint(a=100,b=700),'tol':10**random.uniform(a=-7,b=-1),'alpha_1':10**random.uniform(a=-10,b=-2),'alpha_2':10**random.uniform(a=-10,b=-2),'lambda_1':10**random.uniform(a=-10,b=-2),'lambda_2':10**random.uniform(a=-10,b=-2),'compute_score':random.choice(seq=[True,False]),'threshold_lambda':random.uniform(a=5000,b=15000),'fit_intercept':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='BayesianRidge':fs_estimator_hyperparams={'max_iter':random.randint(a=100,b=700),'tol':10**random.uniform(a=-5,b=-1),'alpha_1':10**random.uniform(a=-10,b=-2),'alpha_2':10**random.uniform(a=-10,b=-2),'lambda_1':10**random.uniform(a=-10,b=-2),'lambda_2':10**random.uniform(a=-10,b=-2),'alpha_init':random.uniform(a=0.01,b=1.0),'lambda_init':random.uniform(a=0.01,b=1.0),'compute_score':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='HuberRegressor':fs_estimator_hyperparams={'epsilon':random.uniform(a=1.0,b=10.0),'max_iter':random.randint(a=20,b=200),'alpha':10**random.uniform(a=-8,b=0),'warm_start':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=-2)};
                        elif fs_estimator_type=='QuantileRegressor':
                            fs_estimator_hyperparams={'quantile':random.uniform(a=0.0,b=1.0),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['highs-ds','highs-ipm','highs','interior-point','revised simplex'])};
                            #Solver interior-point is not anymore available in SciPy >= 1.11.0
                            if fs_estimator_hyperparams['solver']=='interior-point':fs_estimator_hyperparams['solver']=random.choice(seq=['highs-ds','highs-ipm','highs','revised simplex']);
                        elif fs_estimator_type=='RANSACRegressor':fs_estimator_hyperparams={'min_samples':random.uniform(a=0.0,b=1.0),'max_trials':random.randint(a=50,b=150),'max_skips':random.randint(a=500,b=1000),'stop_n_inliers':random.randint(a=500,b=1000),'stop_score':10**random.uniform(a=3,b=10),'stop_probability':random.uniform(a=0.95,b=1.00),'loss':random.choice(seq=['absolute_error','squared_error']),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='TheilSenRegressor':fs_estimator_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'max_subpopulation':10**random.uniform(a=-6,b=-2),'n_subsamples':random.randint(a=opened_data.shape[1]+1,b=opened_data.shape[0]),'max_iter':random.randint(a=100,b=500),'tol':10**random.uniform(a=-5,b=-1),'n_jobs':n_cpu_cores,'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='GammaRegressor':fs_estimator_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-9,b=-2),'warm_start':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='PoissonRegressor':fs_estimator_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='TweedieRegressor':fs_estimator_hyperparams={'power':random.choice(seq=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.01,1.02,1.03,1.04,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,1.96,1.97,1.98,1.99,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'link':random.choice(seq=['auto','identity','log']),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
                        elif fs_estimator_type=='PassiveAggressiveRegressor':fs_estimator_hyperparams={'C':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-5,b=-1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=0.9),'n_iter_no_change':random.randint(a=2,b=10),'loss':random.choice(seq=['epsilon_insensitive','squared_epsilon_insensitive']),'epsilon':random.uniform(a=0.05,b=0.15),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='LinearSVC':fs_estimator_hyperparams={'penalty':random.choice(seq=['l1','l2']),'loss':random.choice(seq=['hinge','squared_hinge']),'dual':random.choice(seq=['auto',True,False]),'tol':10**random.uniform(a=-6,b=-2),'C':10**random.uniform(a=-2,b=2),'fit_intercept':random.choice(seq=[True,False]),'intercept_scaling':10**random.uniform(a=-2,b=2),'random_state':hyperparam_random_state,'max_iter':random.randint(a=500,b=5000)};
                        elif fs_estimator_type=='NuSVC':fs_estimator_hyperparams={'nu':random.uniform(a=0.0,b=1.0),'kernel':random.choice(seq=['linear','poly','rbf','sigmoid','precomputed']),'degree':random.randint(a=0,b=5),'gamma':random.choice(seq=['auto','scale',random.uniform(a=0.0,b=0.1)]),'shrinking':random.choice(seq=[True,False]),'probability':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-4,b=-2),'class_weight':random.choice(seq=['balanced',None]),'max_iter':random.randint(a=500,b=2000),'decision_function_shape':random.choice(seq=['ovo','ovr']),'break_ties':random.choice(seq=[True,False]),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='SVC':fs_estimator_hyperparams={'C':10**random.uniform(a=-2,b=2),'kernel':random.choice(seq=['linear','poly','rbf','sigmoid','precomputed']),'degree':random.randint(a=0,b=5),'gamma':random.choice(seq=['auto','scale',random.uniform(a=0.0,b=0.1)]),'shrinking':random.choice(seq=[True,False]),'probability':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-4,b=-2),'class_weight':random.choice(seq=['balanced',None]),'max_iter':random.randint(a=500,b=2000),'decision_function_shape':random.choice(seq=['ovo','ovr']),'break_ties':random.choice(seq=[True,False]),'random_state':hyperparam_random_state};
                        elif fs_estimator_type=='LinearSVR':fs_estimator_hyperparams={'epsilon':0.0,'tol':10**random.uniform(a=-6,b=-2),'C':10**random.uniform(a=-2,b=2),'loss':random.choice(seq=['epsilon_insensitive','squared_epsilon_insensitive']),'fit_intercept':random.choice(seq=[True,False]),'intercept_scaling':10**random.uniform(a=-2,b=2),'dual':random.choice(seq=['auto',True,False]),'random_state':hyperparam_random_state,'max_iter':random.randint(a=500,b=5000)};
                        elif fs_estimator_type=='NuSVR':fs_estimator_hyperparams={'nu':random.uniform(a=0.0,b=1.0),'C':10**random.uniform(a=-2,b=2),'kernel':random.choice(seq=['linear','poly','rbf','sigmoid','precomputed']),'degree':random.randint(a=0,b=5),'gamma':random.choice(seq=['auto','scale',random.uniform(a=0.0,b=0.1)]),'shrinking':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-4,b=-2),'max_iter':random.randint(a=500,b=5000)};
                        elif fs_estimator_type=='SVR':fs_estimator_hyperparams={'kernel':random.choice(seq=['linear','poly','rbf','sigmoid','precomputed']),'degree':random.randint(a=0,b=5),'gamma':random.choice(seq=['auto','scale',random.uniform(a=0.0,b=0.1)]),'tol':10**random.uniform(a=-4,b=-2),'C':10**random.uniform(a=-2,b=2),'epsilon':random.uniform(a=0.05,b=0.15),'shrinking':random.choice(seq=[True,False]),'max_iter':random.randint(a=500,b=5000)};

            if feature_selector_type=='GenericUnivariateSelect':
                feature_selector_hyperparams={'mode':random.choice(seq=['percentile','k_best','fpr','fdr','fwe'])};
                if feature_selector_hyperparams['mode']=='percentile':feature_selector_hyperparams['param']=random.choice(seq=[1,2,5,10,25,50,75,80,90,95,98,99]);
                elif feature_selector_hyperparams['mode']=='k_best':feature_selector_hyperparams['param']=random.choice(seq=[5,10,20,30,50,100]);
                elif feature_selector_hyperparams['mode']in['fpr','fdr','fwe']:feature_selector_hyperparams['param']=10.0**random.uniform(a=-6.0,b=-1.0);
                if feature_selector_hyperparams['mode']=='k_best':
                    if feature_selector_hyperparams['param']>n_features_selected_randomly:feature_selector_hyperparams['param']=n_features_selected_randomly;
                if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):score_func_types:list[str]=['f_classif','mutual_info_classif','chi2'];
                elif problem_type=='regression':score_func_types:list[str]=['f_regression','mutual_info_regression'];
                #Проверка (и исправление при необходимости) совместимости problem_type и fs_score_func_type:
                if (problem_type=='classification_binary')or(problem_type=='classification_multiclass')and(fs_score_func_type in ['f_regression','mutual_info_regression']):
                    fs_score_func_type=random.choice(seq=['f_classif','mutual_info_classif','chi2']);
                if problem_type=='regression'and fs_score_func_type in ['f_classif','mutual_info_classif','chi2']:
                    fs_score_func_type=random.choice(seq=['f_regression','mutual_info_regression']);
                #Выбор заданной пользователем или случайной функции для feature_selector:
                if fs_score_func_type==None:fs_score_func_type:str=random.choice(seq=score_func_types);
                #В итоге переменная score_func_hyperparams не используется, так как в score_func передаётся именно функция, а не результат
                #вызова функции с некоторым набором параметров
                if fs_score_func_type=='f_classif':score_func_hyperparams:dict={};
                elif fs_score_func_type=='mutual_info_classif':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
                elif fs_score_func_type=='chi2':score_func_hyperparams:dict={};
                elif fs_score_func_type=='f_regression':score_func_hyperparams:dict={'center':random.choice(seq=[True,False]),'force_finite':random.choice(seq=[True,False])};
                elif fs_score_func_type=='mutual_info_regression':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
            #Если feature_selector_type - это один из RFE,RFECV,SelectFromModel,SequentialFeatureSelector, то для него нужна отдельная
            #модель (estimator), по которой определяется важность признаков. Пока что все эти случаи закрыты заглушками (то есть просто
            #считаем, что feature_selector не используется)
            elif feature_selector_type=='RFE':#estimator будем использовать уже отдельно для cross_valid, final_test и production
                feature_selector_hyperparams={'n_features_to_select':random.uniform(a=0.1,b=0.9),'step':random.randint(a=1,b=2),'importance_getter':'auto'};
            elif feature_selector_type=='RFECV':#estimator будем использовать уже отдельно для cross_valid, final_test и production
                feature_selector_hyperparams={'step':random.randint(a=1,b=2),'min_features_to_select':random.randint(a=1,b=10),'cv':random.randint(a=3,b=15),'scoring':None,'n_jobs':n_cpu_cores,'importance_getter':'auto'};
            elif feature_selector_type in ['SelectFdr','SelectFpr','SelectFwe']:
                feature_selector_hyperparams={'alpha':10.0**random.uniform(a=-4.0,b=-0.1)};
                if problem_type=='classification':score_func_types:list[str]=['f_classif','mutual_info_classif','chi2'];
                elif problem_type=='regression':score_func_types:list[str]=['f_regression','mutual_info_regression'];
                fs_score_func_type:str=random.choice(seq=score_func_types);
                if fs_score_func_type=='f_classif':score_func_hyperparams:dict={};
                elif fs_score_func_type=='mutual_info_classif':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
                elif fs_score_func_type=='chi2':score_func_hyperparams:dict={};
                elif fs_score_func_type=='f_regression':score_func_hyperparams:dict={'center':random.choice(seq=[True,False]),'force_finite':random.choice(seq=[True,False])};
                elif fs_score_func_type=='mutual_info_regression':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
            elif feature_selector_type=='SelectFromModel':#estimator будем использовать уже отдельно для cross_valid, final_test и production
                feature_selector_hyperparams={'threshold':None,'prefit':False,'norm_order':random.randint(a=1,b=3),'max_features':random.randint(a=50,b=150),'importance_getter':'auto'};
            elif feature_selector_type=='SelectKBest':
                feature_selector_hyperparams={'k':random.randint(a=10,b=100)};
                if feature_selector_hyperparams['k']>n_features_selected_randomly:feature_selector_hyperparams['k']=n_features_selected_randomly;
                if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
                    score_func_types:list[str]=['f_classif','mutual_info_classif','chi2'];
                elif problem_type=='regression':
                    score_func_types:list[str]=['f_regression','mutual_info_regression'];
                fs_score_func_type:str=random.choice(seq=score_func_types);
                if fs_score_func_type=='f_classif':score_func_hyperparams:dict={};
                elif fs_score_func_type=='mutual_info_classif':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
                elif fs_score_func_type=='chi2':score_func_hyperparams:dict={};
                elif fs_score_func_type=='f_regression':score_func_hyperparams:dict={'center':random.choice(seq=[True,False]),'force_finite':random.choice(seq=[True,False])};
                elif fs_score_func_type=='mutual_info_regression':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
            elif feature_selector_type=='SelectPercentile':
                feature_selector_hyperparams={'percentile':random.randint(a=10,b=100)};
                if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
                    score_func_types:list[str]=['f_classif','mutual_info_classif','chi2'];
                elif problem_type=='regression':
                    score_func_types:list[str]=['f_regression','mutual_info_regression'];
                fs_score_func_type:str=random.choice(seq=score_func_types);
                if fs_score_func_type=='f_classif':score_func_hyperparams:dict={};
                elif fs_score_func_type=='mutual_info_classif':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
                elif fs_score_func_type=='chi2':score_func_hyperparams:dict={};
                elif fs_score_func_type=='f_regression':score_func_hyperparams:dict={'center':random.choice(seq=[True,False]),'force_finite':random.choice(seq=[True,False])};
                elif fs_score_func_type=='mutual_info_regression':score_func_hyperparams:dict={'discrete_features':random.choice(seq=['auto',True,False]),'n_neighbors':random.randint(a=1,b=5),'random_state':score_func_random_state,'n_jobs':n_cpu_cores};
            elif feature_selector_type=='SequentialFeatureSelector':#estimator будем использовать уже отдельно для cross_valid, final_test и production
                feature_selector_hyperparams={'n_features_to_select':random.uniform(a=0.1,b=0.9),'tol':10**random.uniform(a=-2,b=0),'direction':random.choice(seq=['forward','backward']),'scoring':None,'cv':random.randint(a=3,b=15),'n_jobs':n_cpu_cores};
    else:
        print(f'Выбрано use_feature_selector==False, feature_selector не применяется');
        feature_selector_hyperparams=None;

    #3.5. Установка типа и гиперпараметров model
    models_having_coef_and_intercept_attributes:list[str]=conf_dict['pipeline_params']['models_having_coef_and_intercept_attributes'];
    models_with_predict_proba:list[str]=conf_dict['pipeline_params']['models_with_predict_proba'];
    if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
        if model_type is None:
            #Выбор случайной модели из списка:
            model_types:list[str]=['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier','HistGradientBoostingClassifier','RandomForestClassifier','XGBClassifier','LGBMClassifier','LogisticRegression','PassiveAggressiveClassifier','Perceptron','RidgeClassifier','SGDClassifier','XGBRFClassifier','DaskLGBMClassifier','GaussianProcessClassifier','BernoulliNB','ComplementNB','GaussianNB','MultinomialNB','KNeighborsClassifier','NearestCentroid','RadiusNeighborsClassifier','MLPClassifier'];
            if prefered_model_types is not None:model_types=list(set(model_types).intersection(set(prefered_model_types)));
            if use_only_models_with_predict_proba==True:model_types=list(set(model_types).intersection(set(models_with_predict_proba)));
            model_type:str=random.choice(seq=model_types);
        if model_hyperparams is None:
            model_hyperparams:dict[str,int|float|bool|str|tuple[int]];
            if model_type=='AdaBoostClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'learning_rate':10**random.uniform(a=-2,b=0.5),'random_state':hyperparam_random_state};
            elif model_type=='BaggingClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'max_samples':random.uniform(a=0.5,b=0.9),'max_features':random.uniform(a=0.5,b=0.9),'bootstrap':bool(random.randint(a=0,b=1)),'bootstrap_features':bool(random.randint(a=0,b=1)),'oob_score':bool(random.randint(a=0,b=1)),'warm_start':bool(random.randint(a=0,b=1)),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='ExtraTreesClassifier':model_hyperparams={'n_estimators':random.randint(a=20,b=500),'criterion':random.choice(seq=['gini','entropy','log_loss']),'max_depth':random.randint(a=5,b=50),'min_samples_split':random.randint(a=1,b=10),'min_samples_leaf':random.randint(a=1,b=10),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=5,b=50),'min_impurity_decrease':random.uniform(a=0.0,b=0.1),'bootstrap':bool(random.randint(a=0,b=1)),'oob_score':bool(random.randint(a=0,b=1)),'warm_start':bool(random.randint(a=0,b=1)),'class_weight':random.choice(seq=['balanced','balanced_subsample',None]),'ccp_alpha':random.uniform(a=0.0,b=0.1),'max_samples':random.uniform(a=0.001,b=1.0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='GradientBoostingClassifier':model_hyperparams={'loss':random.choice(seq=['log_loss','exponential']),'learning_rate':10**random.uniform(a=-5,b=2),'n_estimators':random.randint(a=20,b=500),'subsample':random.uniform(a=0.001,b=1.0),'criterion':random.choice(seq=['friedman_mse','squared_error']),'min_samples_split':random.randint(a=2,b=10),'min_samples_leaf':random.randint(a=1,b=50),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_depth':random.randint(a=1,b=20),'min_impurity_decrease':random.uniform(a=0.0,b=1.0),'init':random.choice(seq=['zero',None]),'max_features':random.choice(seq=['sqrt','log2']),'max_leaf_nodes':random.randint(a=2,b=50),'warm_start':bool(random.randint(a=0,b=1)),'validation_fraction':random.uniform(a=0.00001,b=0.999999),'n_iter_no_change':random.randint(a=1,b=100),'tol':10**random.uniform(a=-8,b=-1),'ccp_alpha':random.uniform(a=0.0,b=0.1),'random_state':hyperparam_random_state};
            elif model_type=='HistGradientBoostingClassifier':model_hyperparams={'learning_rate':10**random.uniform(a=-4,b=0),'max_iter':random.randint(a=20,b=400),'max_leaf_nodes':random.randint(a=2,b=50),'max_depth':random.randint(a=2,b=40),'min_samples_leaf':random.randint(a=5,b=100),'l2_regularization':10**random.uniform(a=-10,b=0),'max_features':random.uniform(a=0.8,b=1.0),'max_bins':random.randint(a=10,b=255),'warm_start':random.choice(seq=[True,False]),'n_iter_no_change':random.randint(a=3,b=30),'tol':10**random.uniform(a=-10,b=-3),'random_state':hyperparam_random_state};
            elif model_type=='RandomForestClassifier':
                model_hyperparams={'n_estimators':random.randint(a=10,b=500),'criterion':random.choice(seq=['gini','entropy','log_loss']),'max_depth':random.randint(a=2,b=50),'min_samples_split':random.randint(a=2,b=50),'min_samples_leaf':random.randint(a=1,b=5),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.2),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=10,b=100),'min_impurity_decrease':random.uniform(a=0.0,b=0.1),'bootstrap':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=0.2),'max_samples':random.uniform(a=0.00000001,b=1.0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
                #if model_hyperparams['bootstrap']==False:model_hyperparams['max_samples']=None;
            elif model_type=='XGBClassifier':model_hyperparams={'n_estimators':random.randint(a=10,b=500),'max_depth':random.randint(a=2,b=40),'max_leaves':random.randint(a=0,b=50),'max_bin':random.randint(a=5,b=100),'grow_policy':random.choice(seq=['depthwise','lossguide']),'learning_rate':10**random.uniform(a=-9,b=-1),'booster':random.choice(seq=['gbtree','gblinear','dart']),'gamma':random.uniform(a=0.0,b=1.0),'min_child_weight':random.uniform(a=0.01,b=0.1),'max_delta_step':random.uniform(a=0.1,b=2.0),'subsample':random.uniform(a=0.01,b=0.99),'sampling_method':random.choice(seq=['uniform','gradient_based']),'colsample_bytree':random.uniform(a=0.5,b=0.99),'colsample_bylevel':random.uniform(a=0.5,b=0.99),'colsample_bynode':random.uniform(a=0.5,b=0.99),'reg_alpha':10**random.uniform(a=-12,b=0),'reg_lambda':10**random.uniform(a=-12,b=0),'num_parallel_tree':random.randint(a=5,b=50),'importance_type':random.choice(seq=['gain','weight','cover','total_gain','total_cover']),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='LGBMClassifier':model_hyperparams={'num_leaves':random.randint(a=10,b=60),'max_depth':random.randint(4,40),'learning_rate':10**random.uniform(a=-4,b=1.5),'n_estimators':random.randint(a=20,b=500),'subsample_for_bin':random.randint(a=50_000,b=500_000),'min_child_weight':random.uniform(a=0.0001,b=0.01),'min_child_samples':random.randint(a=5,b=50),'reg_alpha':10**random.uniform(a=-12,b=0),'reg_lambda':10**random.uniform(a=-12,b=0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='LogisticRegression':model_hyperparams={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'dual':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=0),'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']),'max_iter':random.randint(a=50,b=500),'warm_start':random.choice(seq=[True,False]),'l1_ratio':random.uniform(a=0.0,b=1.0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='PassiveAggressiveClassifier':model_hyperparams={'C':random.uniform(a=0.5,b=1.5),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=3000),'tol':10**random.uniform(a=-7,b=1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'shuffle':random.choice(seq=[True,False]),'loss':random.choice(seq=['hinge','squared_hinge']),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='Perceptron':model_hyperparams={'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-12,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='RidgeClassifier':model_hyperparams={'alpha':10**random.uniform(a=-2,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-8,b=0),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='SGDClassifier':model_hyperparams={'loss':random.choice(seq=['hinge','log_loss','modified_huber','squared_hinge','perceptron','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-8,b=0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-6,b=0),'shuffle':random.choice(seq=[True,False]),'epsilon':10**random.uniform(a=-9,b=3),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=1),'power_t':random.uniform(a=-5,b=6),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=1.0),'n_iter_no_change':random.randint(a=3,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[True,False,1,2,3,4,5,6,7,8,9,10]),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores};
            elif model_type=='XGBRFClassifier':model_hyperparams={'n_estimators':random.randint(a=50,b=500),'max_depth':random.randint(a=5,b=15),'max_leaves':random.randint(a=10,b=100),'max_bin':random.randint(a=10,b=100),'grow_policy':random.choice(seq=['depthwise','lossguide']),'learning_rate':10**random.uniform(a=-5,b=2),'verbosity':0,'booster':random.choice(seq=['gbtree','gblinear','dart']),'tree_method':random.choice(seq=['auto','exact','approx','hist']),'n_jobs':n_cpu_cores,'gamma':random.uniform(a=0.0,b=0.05),'sampling_method':random.choice(seq=['uniform','gradient_based']),'reg_alpha':10**random.uniform(a=-3,b=1),'reg_lambda':10**random.uniform(a=-3,b=1),'random_state':hyperparam_random_state,'missing':random.choice(seq=[0.0,1.0,-1.0])};
            elif model_type=='DaskLGBMClassifier':model_hyperparams={'boosting_type':random.choice(seq=['gbdt','dart','goss']),'num_leaves':random.randint(a=10,b=60),'max_depth':random.randint(a=5,b=20),'learning_rate':10**random.uniform(a=-3,b=0.5),'n_estimators':random.randint(a=30,b=500),'subsample_for_bin':random.randint(a=50000,b=500000),'min_split_gain':random.uniform(a=0.0,b=0.05),'min_child_weight':10**random.uniform(a=-3.5,b=-2.5),'min_child_samples':random.randint(a=5,b=50),'subsample':random.uniform(a=0.8,b=1.0),'colsample_bytree':random.uniform(a=0.8,b=1.0),'reg_alpha':10**random.uniform(a=-4,b=0),'reg_lambda':10**random.uniform(a=-4,b=0),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores,'importance_type':random.choice(seq=['split','gain'])};
            elif model_type=='GaussianProcessClassifier':model_hyperparams={'max_iter_predict':random.randint(a=50,b=500),'warm_start':random.choice(seq=[True,False]),'random_state':hyperparam_random_state,'multi_class':random.choice(seq=['one_vs_rest','one_vs_one']),'n_jobs':n_cpu_cores};
            elif model_type=='BernoulliNB':model_hyperparams={'alpha':10**random.uniform(a=-10,b=0),'force_alpha':random.choice(seq=[True,False]),'binarize':random.choice(seq=[0.0,0.01,0.02,0.05,0.1,0.2,0.5,1.0]),'fit_prior':true_with_prob(p=0.7)};
            elif model_type=='ComplementNB':model_hyperparams={'alpha':10**random.uniform(a=-10,b=0),'force_alpha':random.choice(seq=[True,False]),'fit_prior':true_with_prob(p=0.7),'norm':true_with_prob(p=0.4)};
            elif model_type=='GaussianNB':model_hyperparams={'var_smoothing':10**random.uniform(a=-11,b=-7)};
            elif model_type=='MultinomialNB':model_hyperparams={'alpha':10**random.uniform(a=-10,b=0),'force_alpha':random.choice(seq=[True,False]),'fit_prior':true_with_prob(p=0.7)};
            elif model_type=='KNeighborsClassifier':model_hyperparams={'n_neighbors':random.randint(a=1,b=15),'weights':random.choice(seq=['uniform','distance']),'algorithm':random.choice(seq=['auto','ball_tree','kd_tree','brute']),'leaf_size':random.randint(a=10,b=50),'p':random.uniform(a=1.0,b=5.0),'metric':random.choice(seq=['minkowski','cityblock','cosine','euclidean','haversine','l1','l2']),'n_jobs':n_cpu_cores};
            elif model_type=='NearestCentroid':model_hyperparams={'metric':random.choice(seq=['euclidean','manhattan']),'priors':random.choice(seq=['uniform','empirical'])};
            elif model_type=='RadiusNeighborsClassifier':model_hyperparams={'radius':10**random.uniform(a=-1.5,b=1.5),'weights':random.choice(seq=['uniform','distance']),'algorithm':random.choice(seq=['auto','ball_tree','kd_tree','brute']),'leaf_size':random.randint(a=10,b=50),'p':random.uniform(a=1.0,b=5.0),'metric':random.choice(seq=['minkowski','cityblock','cosine','euclidean','haversine','l1','l2']),'n_jobs':n_cpu_cores};
            elif model_type=='MLPClassifier':model_hyperparams={'hidden_layer_sizes':generate_hidden_layer_sizes_tuple(n_layers=random.randint(a=1,b=5),n_in=n_features_selected_randomly,n_out=1,allow_increase=true_with_prob(p=0.1),log_scale=true_with_prob(p=0.85)),'activation':random.choice(seq=['logistic','tanh','relu','relu']),'solver':random.choice(seq=['lbfgs','sgd','adam']),'alpha':10**random.uniform(a=-6,b=-2),'learning_rate':random.choice(seq=['constant','invscaling','adaptive']),'learning_rate_init':10**random.uniform(a=-5,b=-1),'power_t':random.uniform(a=0.3,b=0.7),'max_iter':random.randint(a=100,b=1000),'shuffle':True,'random_state':hyperparam_random_state,'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False]),'momentum':random.uniform(a=0.8,b=1.0),'nesterovs_momentum':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.05,b=0.15),'beta_1':random.uniform(a=0.8,b=1.0),'beta_2':random.uniform(a=0.998,b=1.0),'epsilon':10**random.uniform(a=-10,b=-6),'n_iter_no_change':random.randint(a=1,b=20),'max_fun':random.randint(a=5000,b=35000)};

    elif problem_type=='regression':
        if model_type is None:#Если тип модели не указан, то он выбирается случайно из списка model_types
            if task_output=='mono_output':
                model_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegressor','BaggingRegressor','ExtraTreesRegressor','GradientBoostingRegressor','HistGradientBoostingRegressor','RandomForestRegressor','XGBRegressor','XGBRFRegressor','LGBMRegressor','DaskLGBMRegressor','GaussianProcessRegressor','KNeighborsRegressor','RadiusNeighborsRegressor','MLPRegressor'];
            elif task_output=='multi_output':
                model_types:list[str]=['LinearRegression','Ridge','SGDRegressor','ElasticNet','Lars','Lasso','LassoLars','LassoLarsIC','ARDRegression','BayesianRidge','MultiTaskElasticNet','MultiTaskLasso','HuberRegressor','QuantileRegressor','RANSACRegressor','TheilSenRegressor','GammaRegressor','PoissonRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegressor','BaggingRegressor','ExtraTreesRegressor','GradientBoostingRegressor','HistGradientBoostingRegressor','RandomForestRegressor','MLPRegressor'];
            else:
                print(f'Необходимо задать тип выхода (параметр task_output:str, значения: mono_output или multi_output)');
                return error_str;
            if use_only_linear_models==True:
                model_types:list[str]=list(set(model_types).intersection(set(models_having_coef_and_intercept_attributes)));
                print(f'Выбран параметр use_only_linear_models=True');
            if(non_negative_y_guarantee==False)and('PoissonRegressor'in model_types):model_types.remove('PoissonRegressor');#Some value(s) of y are negative which is not allowed for Poisson regression.
            if prefered_model_types is not None:model_types=list(set(model_types).intersection(set(prefered_model_types)));
            model_type:str=random.choice(seq=model_types);
        print(f'model_types: {model_types}');
        if model_hyperparams is None:#Если словарь гиперпараметров модели не указан, то значение каждого гиперпараметра выбирается случайным образом
            model_hyperparams:dict[str,int|float|bool|str];
            if model_type=='LinearRegression':model_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-9,b=-3),'positive':random.choice(seq=[True,False])};
            elif model_type=='Ridge':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=500,b=20000),'tol':10**random.uniform(a=-9,b=-0.1),'solver':random.choice(seq=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']),'positive':bool(random.randint(a=0,b=1)),'random_state':hyperparam_random_state};
            elif model_type=='SGDRegressor':model_hyperparams={'loss':random.choice(seq=['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive']),'penalty':random.choice(seq=['l1','l2','elasticnet',None]),'alpha':10**random.uniform(a=-9,b=1.0),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=20000),'tol':10**random.uniform(a=-6,b=0),'epsilon':10**random.uniform(a=-3,b=1),'learning_rate':random.choice(seq=['constant','optimal','invscaling','adaptive']),'eta0':10**random.uniform(a=-5,b=0),'power_t':random.uniform(a=-100,b=100),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0,b=1),'n_iter_no_change':random.randint(a=2,b=10),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),'random_state':hyperparam_random_state};
            elif model_type=='ElasticNet':model_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'l1_ratio':random.uniform(a=0,b=1),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=5000),'tol':10**random.uniform(a=-7,b=-1),'warm_start':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
            elif model_type=='Lars':model_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'n_nonzero_coefs':random.randint(a=10,b=100),'eps':10**random.uniform(a=-5,b=-1),'fit_path':random.choice(seq=[True,False]),'jitter':10**random.uniform(a=-9,b=-1),'random_state':hyperparam_random_state};
            elif model_type=='Lasso':model_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(100,2000),'tol':10**random.uniform(a=-8,b=-1),'warm_start':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
            elif model_type=='LassoLars':model_hyperparams={'alpha':10**random.uniform(a=-5,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=1500),'eps':10**random.uniform(a=-10,b=-5),'fit_path':random.choice(seq=[True,False]),'positive':random.choice(seq=[True,False]),'jitter':10**random.uniform(a=-9,b=-1),'random_state':hyperparam_random_state};
            elif model_type=='LassoLarsIC':model_hyperparams={'criterion':random.choice(seq=['aic','bic']),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=1500),'eps':10**random.uniform(a=-16,b=-10),'positive':random.choice(seq=[True,False]),'noise_variance':10**random.uniform(a=-5,b=-1)};
            elif model_type=='ARDRegression':model_hyperparams={'max_iter':random.randint(a=100,b=700),'tol':10**random.uniform(a=-7,b=-1),'alpha_1':10**random.uniform(a=-10,b=-2),'alpha_2':10**random.uniform(a=-10,b=-2),'lambda_1':10**random.uniform(a=-10,b=-2),'lambda_2':10**random.uniform(a=-10,b=-2),'compute_score':random.choice(seq=[True,False]),'threshold_lambda':random.uniform(a=5000,b=15000),'fit_intercept':random.choice(seq=[True,False])};
            elif model_type=='BayesianRidge':model_hyperparams={'max_iter':random.randint(a=100,b=700),'tol':10**random.uniform(a=-5,b=-1),'alpha_1':10**random.uniform(a=-10,b=-2),'alpha_2':10**random.uniform(a=-10,b=-2),'lambda_1':10**random.uniform(a=-10,b=-2),'lambda_2':10**random.uniform(a=-10,b=-2),'alpha_init':random.uniform(a=0.01,b=1.0),'lambda_init':random.uniform(a=0.01,b=1.0),'compute_score':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False])};
            elif model_type=='MultiTaskElasticNet':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'l1_ratio':random.uniform(a=0.0,b=1.0),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-8,b=-1),'warm_start':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
            elif model_type=='MultiTaskLasso':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-8,b=-1),'warm_start':random.choice(seq=[True,False]),'selection':random.choice(seq=['cyclic','random']),'random_state':hyperparam_random_state};
            elif model_type=='HuberRegressor':model_hyperparams={'epsilon':random.uniform(a=1.0,b=10.0),'max_iter':random.randint(a=20,b=200),'alpha':10**random.uniform(a=-8,b=0),'warm_start':random.choice(seq=[True,False]),'fit_intercept':random.choice(seq=[True,False]),'tol':10**random.uniform(a=-8,b=-2)};
            elif model_type=='QuantileRegressor':
                model_hyperparams={'quantile':random.uniform(a=0.0,b=1.0),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['highs-ds','highs-ipm','highs','interior-point','revised simplex'])};
                #Solver interior-point is not anymore available in SciPy >= 1.11.0
                if model_hyperparams['solver']=='interior-point':model_hyperparams['solver']=random.choice(seq=['highs-ds','highs-ipm','highs','revised simplex']);
            elif model_type=='RANSACRegressor':model_hyperparams={'min_samples':random.uniform(a=0.0,b=1.0),'max_trials':random.randint(a=50,b=150),'max_skips':random.randint(a=500,b=1000),'stop_n_inliers':random.randint(a=500,b=1000),'stop_score':10**random.uniform(a=3,b=10),'stop_probability':random.uniform(a=0.95,b=1.00),'loss':random.choice(seq=['absolute_error','squared_error']),'random_state':hyperparam_random_state};
            elif model_type=='TheilSenRegressor':model_hyperparams={'fit_intercept':random.choice(seq=[True,False]),'max_subpopulation':10**random.uniform(a=-6,b=-2),'n_subsamples':random.randint(a=opened_data.shape[1]+1,b=opened_data.shape[0]),'max_iter':random.randint(a=100,b=500),'tol':10**random.uniform(a=-5,b=-1),'n_jobs':-1,'random_state':hyperparam_random_state};
            elif model_type=='GammaRegressor':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-9,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='PoissonRegressor':model_hyperparams={'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='TweedieRegressor':model_hyperparams={'power':random.choice(seq=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.01,1.02,1.03,1.04,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,1.96,1.97,1.98,1.99,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]),'alpha':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'link':random.choice(seq=['auto','identity','log']),'solver':random.choice(seq=['lbfgs','newton-cholesky']),'max_iter':random.randint(a=20,b=200),'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False])};
            elif model_type=='PassiveAggressiveRegressor':model_hyperparams={'C':10**random.uniform(a=-4,b=2),'fit_intercept':random.choice(seq=[True,False]),'max_iter':random.randint(a=100,b=2000),'tol':10**random.uniform(a=-5,b=-1),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=0.9),'n_iter_no_change':random.randint(a=2,b=10),'loss':random.choice(seq=['epsilon_insensitive','squared_epsilon_insensitive']),'epsilon':random.uniform(a=0.05,b=0.15),'warm_start':random.choice(seq=[True,False]),'average':random.choice(seq=[False,False,False,False,False,False,False,False,False,False,False,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]),'random_state':hyperparam_random_state};
            elif model_type=='AdaBoostRegressor':model_hyperparams={'n_estimators':random.randint(a=10,b=500),'learning_rate':10**random.uniform(a=-3,b=2),'loss':random.choice(seq=['linear','square','exponential']),'random_state':hyperparam_random_state};
            elif model_type=='BaggingRegressor':
                model_hyperparams={'n_estimators':random.randint(a=5,b=30),'max_samples':random.uniform(a=0.2,b=1.0),'max_features':random.uniform(a=0.2,b=1.0),'bootstrap':random.choice(seq=[True,False]),'bootstrap_features':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'n_jobs':n_cpu_cores,'random_state':hyperparam_random_state};
                #Out of bag estimation only available if bootstrap=True
                if model_hyperparams['bootstrap']==False:model_hyperparams['oob_score']=False;
                #Out of bag estimate only available if warm_start=False
                if model_hyperparams['warm_start']==True:model_hyperparams['oob_score']=False;

            elif model_type=='ExtraTreesRegressor':model_hyperparams={'n_estimators':random.randint(a=20,b=200),'criterion':random.choice(seq=['squared_error','absolute_error','friedman_mse','poisson']),'max_depth':random.randint(a=5,b=20),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2',None]),'max_leaf_nodes':random.randint(a=5,b=10),'min_impurity_decrease':random.uniform(a=0.0,b=0.5),'bootstrap':random.choice(seq=[True,False]),'oob_score':random.choice(seq=[True,False]),'n_jobs':n_cpu_cores,'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=0.1),'max_samples':random.uniform(a=0.0,b=1.0),'random_state':hyperparam_random_state};
            elif model_type=='GradientBoostingRegressor':model_hyperparams={'loss':random.choice(seq=['squared_error','absolute_error','huber','quantile']),'learning_rate':10**random.uniform(a=-2,b=0),'n_estimators':random.randint(a=20,b=300),'subsample':random.uniform(a=0.0,b=1.0),'criterion':random.choice(seq=['friedman_mse','squared_error']),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_depth':random.randint(a=1,b=7),'min_impurity_decrease':random.uniform(a=0.0,b=1.0),'max_features':random.choice(seq=['sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2',0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,None,None,None,None,None,None]),'alpha':random.uniform(a=0.0,b=1.0),'max_leaf_nodes':random.randint(a=2,b=100),'warm_start':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.0,b=0.4),'n_iter_no_change':random.randint(a=5,b=20),'tol':10**random.uniform(a=-6,b=-2),'ccp_alpha':random.uniform(a=0.0,b=100.0),'random_state':hyperparam_random_state};
            elif model_type=='HistGradientBoostingRegressor':
                model_hyperparams={'loss':random.choice(seq=['squared_error','absolute_error','gamma','poisson','quantile']),'quantile':random.uniform(a=0.0,b=1.0),'learning_rate':10**random.uniform(a=-2,b=0),'max_iter':random.randint(a=20,b=200),'max_leaf_nodes':random.randint(a=2,b=60),'max_depth':random.randint(a=2,b=10),'min_samples_leaf':random.randint(a=5,b=50),'l2_regularization':random.uniform(a=0.0,b=1.0),'max_features':random.uniform(a=0.2,b=1.0),'max_bins':random.randint(a=10,b=255),'warm_start':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=['auto',True]),'scoring':random.choice(seq=['loss',None]),'validation_fraction':random.uniform(a=0.05,b=0.25),'n_iter_no_change':random.randint(a=3,b=30),'tol':10**random.uniform(a=-11,b=-3),'random_state':hyperparam_random_state};
                #loss='poisson' requires non-negative y and sum(y) > 0
                if non_negative_y_guarantee==False:model_hyperparams['loss']=random.choice(seq=['squared_error','absolute_error','gamma','quantile']);
            elif model_type=='RandomForestRegressor':
                model_hyperparams={'n_estimators':random.randint(a=20,b=200),'criterion':random.choice(seq=['squared_error','absolute_error','friedman_mse','poisson']),'max_depth':random.randint(a=2,b=20),'min_samples_split':random.uniform(a=0.0,b=1.0),'min_samples_leaf':random.uniform(a=0.0,b=1.0),'min_weight_fraction_leaf':random.uniform(a=0.0,b=0.5),'max_features':random.choice(seq=['sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2','sqrt','log2',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),'min_impurity_decrease':random.uniform(a=0.0,b=0.2),'bootstrap':random.choice(seq=[True,False]),'warm_start':random.choice(seq=[True,False]),'ccp_alpha':random.uniform(a=0.0,b=1.0),'max_samples':random.uniform(a=0.0,b=1.0),'n_jobs':n_cpu_cores,'random_state':hyperparam_random_state};
                #`max_sample` cannot be set if `bootstrap=False`. Either switch to `bootstrap=True` or set `max_sample=None`
                if model_hyperparams['bootstrap']==False:model_hyperparams['max_sample']=None;
                #Some value(s) of y are negative which is not allowed for Poisson regression
                if non_negative_y_guarantee==False:model_hyperparams['criterion']=random.choice(seq=['squared_error','absolute_error','friedman_mse']);
            elif model_type=='XGBRegressor':model_hyperparams={'n_estimators':random.randint(a=20,b=100),'max_depth':random.randint(a=5,b=20),'max_leaves':random.randint(a=10,b=50),'max_bin':random.randint(a=50,b=500),'grow_policy':random.choice(seq=['depthwise','lossguide']),'learning_rate':random.uniform(a=0.0,b=0.5),'verbosity':0,'booster':random.choice(seq=['gbtree','gblinear','dart']),'tree_method':random.choice(seq=['exact','approx','hist']),'n_jobs':n_cpu_cores,'gamma':random.uniform(a=0.0,b=1.0),'min_child_weight':random.randint(a=1,b=5),'max_delta_step':random.randint(a=0,b=5),'subsample':random.uniform(a=0.7,b=1.0),'sampling_method':random.choice(seq=['uniform','gradient_based']),'colsample_bytree':random.uniform(a=0.7,b=1.0),'colsample_bylevel':random.uniform(a=0.7,b=1.0),'colsample_bynode':random.uniform(a=0.7,b=1.0),'reg_alpha':random.uniform(a=0.0,b=0.4),'reg_lambda':random.uniform(a=0.0,b=0.4),'scale_pos_weight':10**random.uniform(a=-1,b=1),'random_state':hyperparam_random_state,'missing':random.choice(seq=[0,1,-1]),'num_parallel_tree':random.randint(a=1,b=10),'monotone_constraints':random.choice(seq=[1,-1,0,0,0,0,0]),'importance_type':random.choice(seq=['gain','weight','cover','total_gain','total_cover'])};
            elif model_type=='XGBRFRegressor':model_hyperparams={'n_estimators':random.randint(a=50,b=500),'max_depth':random.randint(a=3,b=7),'max_leaves':random.randint(a=50,b=1000),'max_bin':random.randint(a=50,b=500),'grow_policy':random.choice(seq=['depthwise','lossguide']),'learning_rate':random.uniform(a=0.0,b=0.5),'verbosity':0,'booster':random.choice(seq=['gbtree','gblinear','dart']),'tree_method':random.choice(seq=['exact','approx','hist']),'n_jobs':n_cpu_cores,'gamma':random.uniform(a=0.0,b=1.0),'min_child_weight':random.randint(a=1,b=5),'max_delta_step':random.randint(a=0,b=5),'subsample':random.uniform(a=0.7,b=1.0),'sampling_method':random.choice(seq=['uniform','gradient_based']),'colsample_bytree':random.uniform(a=0.7,b=1.0),'colsample_bylevel':random.uniform(a=0.7,b=1.0),'colsample_bynode':random.uniform(a=0.7,b=1.0),'reg_alpha':random.uniform(a=0.0,b=0.4),'reg_lambda':random.uniform(a=0.0,b=0.4),'scale_pos_weight':10**random.uniform(a=-1,b=1),'random_state':hyperparam_random_state,'missing':random.choice(seq=[0,1,-1]),'num_parallel_tree':random.randint(a=1,b=10),'monotone_constraints':random.choice(seq=[1,-1,0,0,0,0,0]),'importance_type':random.choice(seq=['gain','weight','cover','total_gain','total_cover'])};
            elif model_type=='LGBMRegressor':model_hyperparams={'boosting_type':random.choice(seq=['gbdt','dart','rf']),'num_leaves':random.randint(a=10,b=55),'max_depth':random.randint(a=5,b=20),'learning_rate':10**random.uniform(a=-1.5,b=-0.5),'n_estimators':random.randint(a=50,b=300),'subsample_for_bin':random.randint(a=50000,b=500000),'min_split_gain':random.uniform(a=0.0,b=0.02),'min_child_weight':10**random.uniform(a=-3.5,b=-2.5),'min_child_samples':random.randint(a=10,b=30),'subsample':random.uniform(a=0.9,b=1.0),'colsample_bytree':random.uniform(a=0.8,b=1.0),'reg_alpha':10**random.uniform(a=-5,b=-0.5),'reg_lambda':10**random.uniform(a=-5,b=-0.5),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores,'importance_type':random.choice(seq=['split','gain'])};
            elif model_type=='DaskLGBMRegressor':model_hyperparams={'boosting_type':random.choice(seq=['gbdt','dart','rf']),'num_leaves':random.randint(a=10,b=55),'max_depth':random.randint(a=5,b=20),'learning_rate':10**random.uniform(a=-1.5,b=-0.5),'n_estimators':random.randint(a=50,b=300),'subsample_for_bin':random.randint(a=50000,b=500000),'min_split_gain':random.uniform(a=0.0,b=0.02),'min_child_weight':10**random.uniform(a=-3.5,b=-2.5),'min_child_samples':random.randint(a=10,b=30),'subsample':random.uniform(a=0.9,b=1.0),'colsample_bytree':random.uniform(a=0.8,b=1.0),'reg_alpha':10**random.uniform(a=-5,b=-0.5),'reg_lambda':10**random.uniform(a=-5,b=-0.5),'random_state':hyperparam_random_state,'n_jobs':n_cpu_cores,'importance_type':random.choice(seq=['split','gain'])};
            elif model_type=='GaussianProcessRegressor':model_hyperparams={'alpha':10**random.uniform(a=-12,b=-8),'normalize_y':random.choice(seq=[True,False]),'random_state':hyperparam_random_state};
            elif model_type=='KNeighborsRegressor':model_hyperparams={'n_neighbors':random.randint(a=1,b=15),'weights':random.choice(seq=['uniform','distance']),'algorithm':random.choice(seq=['auto','ball_tree','kd_tree','brute']),'leaf_size':random.randint(a=10,b=50),'p':random.uniform(a=1.0,b=5.0),'metric':random.choice(seq=['minkowski','cityblock','cosine','euclidean','haversine','l1','l2']),'n_jobs':n_cpu_cores};
            elif model_type=='RadiusNeighborsRegressor':model_hyperparams={'radius':10**random.uniform(a=-1.5,b=1.5),'weights':random.choice(seq=['uniform','distance']),'algorithm':random.choice(seq=['auto','ball_tree','kd_tree','brute']),'leaf_size':random.randint(a=10,b=50),'p':random.uniform(a=1.0,b=5.0),'metric':random.choice(seq=['minkowski','cityblock','cosine','euclidean','haversine','l1','l2']),'n_jobs':n_cpu_cores};
            elif model_type=='MLPRegressor':
                model_hyperparams={'loss':random.choice(seq=['squared_error','poisson']),'hidden_layer_sizes':generate_hidden_layer_sizes_tuple(n_layers=random.randint(a=1,b=5),n_in=n_features_selected_randomly,n_out=1,allow_increase=true_with_prob(p=0.1),log_scale=true_with_prob(p=0.85)),'activation':random.choice(seq=['logistic','tanh','relu','relu']),'solver':random.choice(seq=['lbfgs','sgd','adam']),'alpha':10**random.uniform(a=-6,b=-2),'learning_rate':random.choice(seq=['constant','invscaling','adaptive']),'learning_rate_init':10**random.uniform(a=-5,b=-1),'power_t':random.uniform(a=0.3,b=0.7),'max_iter':random.randint(a=100,b=1000),'shuffle':True,'random_state':hyperparam_random_state,'tol':10**random.uniform(a=-6,b=-2),'warm_start':random.choice(seq=[True,False]),'momentum':random.uniform(a=0.8,b=1.0),'nesterovs_momentum':random.choice(seq=[True,False]),'early_stopping':random.choice(seq=[True,False]),'validation_fraction':random.uniform(a=0.05,b=0.15),'beta_1':random.uniform(a=0.8,b=1.0),'beta_2':random.uniform(a=0.998,b=1.0),'epsilon':10**random.uniform(a=-10,b=-6),'n_iter_no_change':random.randint(a=1,b=20),'max_fun':random.randint(a=5000,b=35000)};
                if non_negative_y_guarantee==False:model_hyperparams['loss']='squared_error';

    #3.6. Инициализация словарей для хранения конструкторов для imputer, scaler, fs_estimator, feature_selector, model
    imputer_constructors_dict:dict[str,AnyImputer]={'KNNImputer':KNNImputer,'SimpleImputer':SimpleImputer};
    scaler_constructors_dict:dict[str,AnyScaler]={'MaxAbsScaler':MaxAbsScaler,'MinMaxScaler':MinMaxScaler,'RobustScaler':RobustScaler,'StandardScaler':StandardScaler};
    fs_estimator_constructors_dict:dict[str,AnyFSEstimator]={'LogisticRegression':LogisticRegression,'PassiveAggressiveClassifier':PassiveAggressiveClassifier,'Perceptron':Perceptron,'RidgeClassifier':RidgeClassifier,'SGDClassifier':SGDClassifier,'LinearRegression':LinearRegression,'Ridge':Ridge,'SGDRegressor':SGDRegressor,'ElasticNet':ElasticNet,'Lars':Lars,'Lasso':Lasso,'LassoLars':LassoLars,'LassoLarsIC':LassoLarsIC,'OrthogonalMatchingPursuit':OrthogonalMatchingPursuit,'ARDRegression':ARDRegression,'BayesianRidge':BayesianRidge,'HuberRegressor':HuberRegressor,'QuantileRegressor':QuantileRegressor,'RANSACRegressor':RANSACRegressor,'TheilSenRegressor':TheilSenRegressor,'GammaRegressor':GammaRegressor,'PoissonRegressor':PoissonRegressor,'TweedieRegressor':TweedieRegressor,'PassiveAggressiveRegressor':PassiveAggressiveRegressor,'LinearSVC':LinearSVC,'NuSVC':NuSVC,'SVC':SVC,'LinearSVR':LinearSVR,'NuSVR':NuSVR,'SVR':SVR};
    feature_selector_constructors_dict:dict[str,AnyFeatureSelector]={'GenericUnivariateSelect':GenericUnivariateSelect,'RFE':RFE,'RFECV':RFECV,'SelectFdr':SelectFdr,'SelectFpr':SelectFpr,'SelectFromModel':SelectFromModel,'SelectFwe':SelectFwe,'SelectKBest':SelectKBest,'SelectPercentile':SelectPercentile,'SequentialFeatureSelector':SequentialFeatureSelector};
    model_constructors_dict:dict[str,AnyModel]={'LogisticRegression':LogisticRegression,'PassiveAggressiveClassifier':PassiveAggressiveClassifier,'Perceptron':Perceptron,'RidgeClassifier':RidgeClassifier,'SGDClassifier':SGDClassifier,'AdaBoostClassifier':AdaBoostClassifier,'BaggingClassifier':BaggingClassifier,'ExtraTreesClassifier':ExtraTreesClassifier,'GradientBoostingClassifier':GradientBoostingClassifier,'HistGradientBoostingClassifier':HistGradientBoostingClassifier,'RandomForestClassifier':RandomForestClassifier,'XGBClassifier':XGBClassifier,'LGBMClassifier':LGBMClassifier,'LinearRegression':LinearRegression,'Ridge':Ridge,'SGDRegressor':SGDRegressor,'ElasticNet':ElasticNet,'Lasso':Lasso,'LassoLarsIC':LassoLarsIC,'ARDRegression':ARDRegression,'OrthogonalMatchingPursuit':OrthogonalMatchingPursuit,'BayesianRidge':BayesianRidge,'MultiTaskElasticNet':MultiTaskElasticNet,'MultiTaskLasso':MultiTaskLasso,'HuberRegressor':HuberRegressor,'QuantileRegressor':QuantileRegressor,'RANSACRegressor':RANSACRegressor,'Lars':Lars,'TheilSenRegressor':TheilSenRegressor,'GammaRegressor':GammaRegressor,'PoissonRegressor':PoissonRegressor,'TweedieRegressor':TweedieRegressor,'PassiveAggressiveRegressor':PassiveAggressiveRegressor,'LassoLars':LassoLars,'AdaBoostRegressor':AdaBoostRegressor,'BaggingRegressor':BaggingRegressor,'ExtraTreesRegressor':ExtraTreesRegressor,'GradientBoostingRegressor':GradientBoostingRegressor,'HistGradientBoostingRegressor':HistGradientBoostingRegressor,'RandomForestRegressor':RandomForestRegressor,'XGBRFClassifier':XGBRFClassifier,'DaskLGBMClassifier':DaskLGBMClassifier,'XGBRegressor':XGBRegressor,'XGBRFRegressor':XGBRFRegressor,'LGBMRegressor':LGBMRegressor,'DaskLGBMRegressor':DaskLGBMRegressor,'GaussianProcessClassifier':GaussianProcessClassifier,'GaussianProcessRegressor':GaussianProcessRegressor,'BernoulliNB':BernoulliNB,'ComplementNB':ComplementNB,'GaussianNB':GaussianNB,'MultinomialNB':MultinomialNB,'KNeighborsClassifier':KNeighborsClassifier,'NearestCentroid':NearestCentroid,'RadiusNeighborsClassifier':RadiusNeighborsClassifier,'KNeighborsRegressor':KNeighborsRegressor,'RadiusNeighborsRegressor':RadiusNeighborsRegressor};
    
    # 4. Подготовка кросс-валидации
    #ТУТ ТОЖЕ нужно учесть группы по _aug (то есть для каждого фолда все образцы, полученные из одного с помощью n_aug аугментаций, должны быть или все в train,
    #или все в valid).
    print(f"""Начинаем обработку пайплайна [pipeline_id={pipeline_id}] со следующими значениями: n_features_all: {n_features_all}, n_features_selected_randomly: {n_features_selected_randomly}
    randomly_selected_indexes: {randomly_selected_indexes}
    use_var_thresholder: {use_var_thresholder}, var_thresholder_type: {var_thresholder_type}, var_thresholder_hyperparams: {var_thresholder_hyperparams}
    use_scaler: {use_scaler}, scaler_type: {scaler_type}, scaler_hyperparams: {scaler_hyperparams}
    use_feature_selector: {use_feature_selector}, feature_selector_type: {feature_selector_type}
    feature_selector_hyperparams: {feature_selector_hyperparams}
    fs_score_func_type: {fs_score_func_type}, fs_estimator_type: {fs_estimator_type}
    fs_estimator_hyperparams: {fs_estimator_hyperparams}
    model_type: {model_type}
    model_hyperparams: {model_hyperparams}
    split_random_state: {split_random_state}
    score_type: {score_type}""");
    #Этот код работал правильно, но только без групп по _aug:
    """
    if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):K_Fold:StratifiedKFold=StratifiedKFold(n_splits=num_folds,shuffle=True,random_state=split_random_state)
    elif problem_type=='regression':K_Fold:KFold=KFold(n_splits=num_folds,shuffle=True,random_state=split_random_state);
    """
    #StratifiedKFold предназначен ТОЛЬКО для классификационных задач, где целевая переменная имеет дискретные значения (бинарные или 
    #мультикласс). В регрессии же целевая переменная непрерывная (continuous), и StratifiedKFold не может работать с такими данными.

    #================Начало кода от DeepSeek для разделения на train и valid с учётом групп по _aug================
    n_groups_train_cv:int=X_train_cv.shape[0]//n_augs;#Количество групп в train_cv
    group_marks_train_cv:np.ndarray=np.repeat(a=np.arange(stop=n_groups_train_cv),repeats=n_augs);#первые n_augs элементов — группа 0, следующие — группа 1 и т.д.
    cv_splitter:StratifiedGroupKFold|GroupKFold;
    if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
        y_train_cv_int = y_train_cv.astype(int);# y_train_cv преобразуем в целочисленные метки (0/1), т.к. StratifiedGroupKFold ожидает дискретные классы
        cv_splitter:StratifiedGroupKFold=StratifiedGroupKFold(n_splits=num_folds,shuffle=True,random_state=split_random_state);
        y_for_cv_splitter:np.ndarray=y_train_cv_int;
    elif problem_type=='regression':
        cv_splitter:GroupKFold=GroupKFold(n_splits=num_folds);
        y_for_cv_splitter:np.ndarray=y_train_cv;
    scores_valid:list[float]=[];
    if conf_dict['save_ids_lists_to_txt']:
        buf_lst_s = f"group_marks_train_cv (внутри cross-valid): [{','.join([str(group) for group in group_marks_train_cv])}]";
        with open(file=f'group_marks_train_cv_inside_cross_valid.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);
    for fold_num,(indices_train,indices_valid)in enumerate(cv_splitter.split(X=X_train_cv,y=y_for_cv_splitter,groups=group_marks_train_cv)):
        print(f">>>Обрабатываем фолд {fold_num+1}/{num_folds} {cur_dt_to_str()}...",end=' ');
        X_train_fold:np.ndarray=X_train_cv[indices_train];
        y_train_fold:np.ndarray=y_train_cv[indices_train];
        ids_train_fold:np.ndarray=ids_train_cv[indices_train];
        X_valid_fold:np.ndarray=X_train_cv[indices_valid];
        y_valid_fold:np.ndarray=y_train_cv[indices_valid];
        ids_valid_fold:np.ndarray=ids_train_cv[indices_valid];
        indices_train_orig:np.ndarray=indices_train_cv[indices_train];#Получаем исходные индексы для текущего фолда относительно opened_data
        indices_valid_orig:np.ndarray=indices_train_cv[indices_valid];
        part_assignment[indices_valid_orig]=f'cv_fold{fold_num}';#Присваиваем метку фолда только валидационным образцам
        #indices_train_orig заполнятся в другом фолде (когда они будут в valid фолде)
        if conf_dict['print_ids_lists']:
            print(f'len(indices_train): {len(indices_train)}, list(indices_train) [первые {n_groups_to_print} групп]: {[int(indice) for indice in indices_train][:n_groups_to_print*n_augs]}');
            print(f'len(indices_valid): {len(indices_valid)}, list(indices_valid) [первые {n_groups_to_print} групп]: {[int(indice) for indice in indices_valid][:n_groups_to_print*n_augs]}');
            print(f'len(ids_train_fold): {len(ids_train_fold)}, list(ids_train_fold) [первые {n_groups_to_print} групп]: {[str(id) for id in ids_train_fold][:n_groups_to_print*n_augs]}');
            print(f'len(ids_valid_fold): {len(ids_valid_fold)}, list(ids_valid_fold) [первые {n_groups_to_print} групп]: {[str(id) for id in ids_valid_fold][:n_groups_to_print*n_augs]}');
        if conf_dict['save_ids_lists_to_txt']:
            buf_lst_s = f"indices_train (fold {fold_num+1}/{num_folds}): [{','.join([str(indice) for indice in indices_train])}]";
            with open(f'fold_{fold_num+1}_of_{num_folds}_indices_train_fold.txt', 'wt', encoding='UTF-8') as f:f.write(buf_lst_s);
            buf_lst_s = f"indices_valid (fold {fold_num+1}/{num_folds}): [{','.join([str(indice) for indice in indices_valid])}]";
            with open(f'fold_{fold_num+1}_of_{num_folds}_indices_valid_fold.txt', 'wt', encoding='UTF-8') as f:f.write(buf_lst_s);
            buf_lst_s = f"ids_train_fold (fold {fold_num+1}/{num_folds}): [{','.join([str(id) for id in ids_train_fold])}]";
            with open(f'fold_{fold_num+1}_of_{num_folds}_ids_train_fold.txt', 'wt', encoding='UTF-8') as f:f.write(buf_lst_s);
            buf_lst_s = f"ids_valid_fold (fold {fold_num+1}/{num_folds}): [{','.join([str(id) for id in ids_valid_fold])}]";
            with open(f'fold_{fold_num+1}_of_{num_folds}_ids_valid_fold.txt', 'wt', encoding='UTF-8') as f:f.write(buf_lst_s);
        print(f'Размеры X_train_fold: {X_train_fold.shape}, размеры X_valid_fold: {X_valid_fold.shape}');
        print(f'Доля X_train_fold от opened_data: {X_train_fold.shape[0]/opened_data_len:.3f}',end='; ');
        print(f'Доля X_valid_fold от opened_data: {X_valid_fold.shape[0]/opened_data_len:.3f};');
        print(f'Размеры y_train_fold: {y_train_fold.shape}, размеры y_valid_fold: {y_valid_fold.shape}');
        print(f'Доля y_train_fold от opened_data: {y_train_fold.shape[0]/opened_data_len:.3f}',end='; ');
        print(f'Доля y_valid_fold от opened_data: {y_valid_fold.shape[0]/opened_data_len:.3f};');
    #================Конец кода от DeepSeek для разделения на train и valid с учётом групп по _aug================
        """
        for fold_num,(train_index,valid_index) in enumerate(K_Fold.split(X=X_train_cv,y=y_train_cv)):
            print(f"  Обрабатываем фолд {fold_num+1}/{num_folds}...",end=' ');
            if conf_dict['print_ids_lists']==True:
                print(f'len(train_index): {len(train_index)}, list(train_index): {[int(ind)for ind in train_index]}');
                print(f'len(valid_index): {len(valid_index)}, list(valid_index): {[int(ind)for ind in valid_index]}');
            if conf_dict['save_ids_lists_to_txt']==True:
                buf_lst_s:str=f"train_index (fold {fold_num+1}/{num_folds}): [{','.join([str(ind)for ind in train_index])}]";
                with open(file=f'train_index_list_fold_{fold_num+1}_of_{num_folds}.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);
                buf_lst_s:str=f"valid_index (fold {fold_num+1}/{num_folds}): [{','.join([str(ind)for ind in valid_index])}]";
                with open(file=f'valid_index_list_fold_{fold_num+1}_of_{num_folds}.txt',mode='wt',encoding='UTF-8')as f:f.write(buf_lst_s);

            # 4.0 Разделение на train/valid для фолда
            X_train_fold:np.ndarray=X_train_cv[train_index];y_train_fold:np.ndarray=y_train_cv[train_index];
            X_valid_fold:np.ndarray=X_train_cv[valid_index];y_valid_fold:np.ndarray=y_train_cv[valid_index];

            print(f'Доля X_train_fold от opened_data: {X_train_fold.shape[0]/opened_data_len}',end='; ');#0.72
            print(f'Доля X_valid_fold от opened_data: {X_valid_fold.shape[0]/opened_data_len}',end='; ');#0.08
        """
        #4.1. Использование imputer (если выбрано) для cross_valid
        if use_imputer==True:
            if copy_data_arrays==True:#Копируем массивы
                imputer_cross_valid:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
                imputer_cross_valid.fit(X=X_train_fold,y=y_train_fold);
                X_train_fold_imputed=imputer_cross_valid.transform(X=X_train_fold);
                X_valid_fold_imputed=imputer_cross_valid.transform(X=X_valid_fold);
            else:#Перезаписываем те же самые массивы:
                imputer_cross_valid:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
                imputer_cross_valid.fit(X=X_train_fold,y=y_train_fold);
                X_train_fold=imputer_cross_valid.transform(X=X_train_fold);
                X_valid_fold=imputer_cross_valid.transform(X=X_valid_fold);
        else:
            if copy_data_arrays==True:#Копируем массивы
                X_train_fold_imputed=X_train_fold.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
                X_valid_fold_imputed=X_valid_fold.copy();
                imputer_cross_valid=None;
            else:#Перезаписываем те же самые массивы:
                imputer_cross_valid=None;#Если imputer не применяется и массивы перезаписываем, то ничего не делаем

        #4.2. Использование var_thresholder (если выбрано) для cross_valid
        if use_var_thresholder==True:
            if copy_data_arrays==True:#Копируем массивы
                if var_thresholder_type=='VarianceThreshold':var_thresholder_cross_valid=VarianceThreshold(**var_thresholder_hyperparams);
                var_thresholder_cross_valid.fit(X=X_train_fold_imputed,y=y_train_fold);
                X_train_fold_var_thresholded=var_thresholder_cross_valid.transform(X=X_train_fold_imputed);
                X_valid_fold_var_thresholded=var_thresholder_cross_valid.transform(X=X_valid_fold_imputed);
            else:#Перезаписываем те же самые массивы:
                if var_thresholder_type=='VarianceThreshold':var_thresholder_cross_valid=VarianceThreshold(**var_thresholder_hyperparams);
                var_thresholder_cross_valid.fit(X=X_train_fold,y=y_train_fold);
                X_train_fold=var_thresholder_cross_valid.transform(X=X_train_fold);
                X_valid_fold=var_thresholder_cross_valid.transform(X=X_valid_fold);
        else:
            if copy_data_arrays==True:#Копируем массивы
                X_train_fold_var_thresholded=X_train_fold_imputed.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
                X_valid_fold_var_thresholded=X_valid_fold_imputed.copy();
                var_thresholder_cross_valid=None;
            else:#Перезаписываем те же самые массивы:
                var_thresholder_cross_valid=None;#Если var_thresholder не применяется и массивы перезаписываем, то ничего не делаем

        #4.3. Использование scaler (если выбрано) для cross_valid
        if use_scaler==True:
            if copy_data_arrays==True:#Копируем массивы
                scaler_cross_valid:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
                scaler_cross_valid.fit(X=X_train_fold_var_thresholded);#fit только на train, без valid
                X_train_fold_scaled=scaler_cross_valid.transform(X=X_train_fold_var_thresholded);
                X_valid_fold_scaled=scaler_cross_valid.transform(X=X_valid_fold_var_thresholded);
            else:#Перезаписываем массивы
                scaler_cross_valid:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
                scaler_cross_valid.fit(X=X_train_fold);#fit только на train, без valid
                X_train_fold=scaler_cross_valid.transform(X=X_train_fold);
                X_valid_fold=scaler_cross_valid.transform(X=X_valid_fold);
        else:#Если use_scaler==False
            if copy_data_arrays==True:#Копируем массивы
                X_train_fold_scaled=X_train_fold_var_thresholded.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
                X_valid_fold_scaled=X_valid_fold_var_thresholded.copy();
                scaler_cross_valid=None;
            else:#Перезаписываем массивы
                scaler_cross_valid=None;#Если scaler не применяется и массивы перезаписываем, то ничего не делаем

        #4.4. Использование feature_selector (если выбрано) для cross_valid
        if use_feature_selector==True:
            if feature_selector_type in ['GenericUnivariateSelect','SelectFdr','SelectFpr','SelectFwe','SelectKBest','SelectPercentile']:
                if fs_score_func_type=='f_classif':score_func_cross_valid:callable=f_classif;
                elif fs_score_func_type=='mutual_info_classif':score_func_cross_valid:callable=mutual_info_classif;
                elif fs_score_func_type=='chi2':score_func_cross_valid:callable=chi2;
                elif fs_score_func_type=='f_regression':score_func_cross_valid:callable=f_regression;
                elif fs_score_func_type=='mutual_info_regression':score_func_cross_valid:callable=mutual_info_regression;
                feature_selector_cross_valid:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](**feature_selector_hyperparams);
            elif feature_selector_type in feature_selector_types_estimator_all:
                fs_estimator_cross_valid:AnyFSEstimator=fs_estimator_constructors_dict[fs_estimator_type](**fs_estimator_hyperparams);
                feature_selector_cross_valid:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](estimator=fs_estimator_cross_valid,**feature_selector_hyperparams);
            if copy_data_arrays==True:#Копируем массивы
                feature_selector_cross_valid.fit(X=X_train_fold_scaled,y=y_train_fold);
                X_train_fold_feature_selected=feature_selector_cross_valid.transform(X=X_train_fold_scaled);
                X_valid_fold_feature_selected=feature_selector_cross_valid.transform(X=X_valid_fold_scaled);
            else:#Перезаписываем массивы
                feature_selector_cross_valid.fit(X=X_train_fold,y=y_train_fold);
                X_train_fold=feature_selector_cross_valid.transform(X=X_train_fold);
                X_valid_fold=feature_selector_cross_valid.transform(X=X_valid_fold);
        else:#Если use_feature_selector==False
            if copy_data_arrays==True:#Копируем массивы
                X_train_fold_feature_selected=X_train_fold_scaled.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
                X_valid_fold_feature_selected=X_valid_fold_scaled.copy();
                feature_selector_cross_valid=None;
            else:#Перезаписываем массивы
                feature_selector_cross_valid=None;#Если feature_selector не применяется и массивы перезаписываем, то ничего не делаем
        if copy_data_arrays==True:#Копируем массивы
            if (X_train_fold_feature_selected.shape[1]==0)or(X_valid_fold_feature_selected.shape[1]==0):
                print(f'После применения feature_selector типа {feature_selector_type} на этапе cross_valid количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
                return error_str;
        else:#Перезаписываем массивы:
            if (X_train_fold.shape[1]==0)or(X_valid_fold.shape[1]==0):
                print(f'После применения feature_selector типа {feature_selector_type} на этапе cross_valid количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
                return error_str;

        #4.5. Использование model (всегда выбрано) для cross_valid
        model_cross_valid:AnyModel=model_constructors_dict[model_type](**model_hyperparams);
        if copy_data_arrays==True:#Копируем массивы
            model_cross_valid.fit(X=X_train_fold_feature_selected,y=y_train_fold);
            y_valid_pred:np.ndarray=model_cross_valid.predict(X_valid_fold_feature_selected);
        else:#Перезаписываем массивы
            model_cross_valid.fit(X=X_train_fold,y=y_train_fold);
            y_valid_pred:np.ndarray=model_cross_valid.predict(X_valid_fold);
        if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
            if copy_data_arrays==True:#Копируем массивы
                y_valid_pred_proba:np.ndarray=model_cross_valid.predict_proba(X_valid_fold_feature_selected);
            else:#Перезаписываем массивы
                y_valid_pred_proba:np.ndarray=model_cross_valid.predict_proba(X_valid_fold);
            y_valid_pred_proba_positive:np.ndarray=y_valid_pred_proba[:,1];#вероятность положительного класса для бинарной классификации
            if score_type=='accuracy_score':score_valid=accuracy_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='auc':
                if problem_type=='classification_binary':
                    fpr_cross_valid:np.ndarray;tpr_cross_valid:np.ndarray;thresholds_cross_valid:np.ndarray;
                    fpr_cross_valid,tpr_cross_valid,thresholds_cross_valid=roc_curve(y_true=y_valid_fold,y_score=y_valid_pred,pos_label=1);
                    score_valid=auc(x=fpr_cross_valid,y=tpr_cross_valid);
                elif problem_type=='classification_multiclass':
                    print(f'Для многоклассовой классификации НЕ следует использовать auc, лучше roc_auc_score');
                    score_valid=0.0;
            elif score_type=='average_precision_score':pass;
            elif score_type=='balanced_accuracy_score':score_valid=balanced_accuracy_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='brier_score_loss':#Требует вероятность положительного класса для бинарной классификации
                score_valid=brier_score_loss(y_true=y_valid_fold,y_prob=y_valid_pred_proba_positive);
            elif score_type=='cohen_kappa_score':score_valid=cohen_kappa_score(y1=y_valid_fold,y2=y_valid_pred);
            elif score_type=='dcg_score':pass;
            elif score_type=='f1_score':score_valid=f1_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='fbeta_score':score_valid=fbeta_score(y_true=y_valid_fold,y_pred=y_valid_pred,beta=fbeta_score_beta);
            elif score_type=='hamming_loss':score_valid=hamming_loss(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='hinge_loss':pass;
            elif score_type=='jaccard_score':score_valid=jaccard_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='log_loss':score_valid=log_loss(y_true=y_valid_fold,y_pred=y_valid_pred_proba);#log_loss требует полную матрицу вероятностей (n_samples,n_classes)
            elif score_type=='matthews_corrcoef':score_valid=matthews_corrcoef(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='ndcg_score':pass;
            elif score_type=='precision_score':score_valid=precision_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='recall_score':score_valid=recall_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='roc_auc_score':#Для бинарной классификации берём вероятность ПОЛОЖИТЕЛЬНОГО класса (второй столбец)
                if problem_type=='classification_binary':
                    score_valid=roc_auc_score(y_true=y_valid_fold,y_score=y_valid_pred_proba_positive);
                elif problem_type=='classification_multiclass':#Для мультикласса можно использовать 'ovr' или 'ovo' стратегии
                    score_valid=roc_auc_score(y_true=y_valid_fold,y_score=y_valid_pred_proba,multi_class='ovr');
            elif score_type=='zero_one_loss':score_valid=zero_one_loss(y_true=y_valid_fold,y_pred=y_valid_pred);
        elif problem_type=='regression':
            if score_type=='d2_absolute_error_score':score_valid=d2_absolute_error_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='d2_pinball_score':score_valid=d2_pinball_score(y_true=y_valid_fold,y_pred=y_valid_pred,alpha=d2_pinball_score_alpha);
            elif score_type=='d2_tweedie_score':score_valid=d2_tweedie_score(y_true=y_valid_fold,y_pred=y_valid_pred,power=d2_tweedie_score_power);
            elif score_type=='explained_variance_score':score_valid=explained_variance_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='max_error':score_valid=max_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_absolute_percentage_error':score_valid=mean_absolute_percentage_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_gamma_deviance':score_valid=mean_gamma_deviance(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_pinball_loss':score_valid=mean_pinball_loss(y_true=y_valid_fold,y_pred=y_valid_pred,alpha=mean_pinball_loss_alpha);
            elif score_type=='mean_poisson_deviance':score_valid=mean_poisson_deviance(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_squared_error':score_valid=mean_squared_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_squared_log_error':score_valid=mean_squared_log_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='mean_tweedie_deviance':score_valid=mean_tweedie_deviance(y_true=y_valid_fold,y_pred=y_valid_pred,power=mean_tweedie_deviance_power);
            elif score_type=='median_absolute_error':score_valid=median_absolute_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='r2_score':score_valid=r2_score(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='root_mean_squared_error':score_valid=root_mean_squared_error(y_true=y_valid_fold,y_pred=y_valid_pred);
            elif score_type=='root_mean_squared_log_error':score_valid=root_mean_squared_log_error(y_true=y_valid_fold,y_pred=y_valid_pred);
        print(f"Тип метрики: {score_type}, значение метрики на фолде {fold_num+1} из {num_folds}: {score_valid:.6f} {cur_dt_to_str()}");
        if score_valid_min_threshold!=None:
            if score_valid<score_valid_min_threshold:
                print(f'Значение метрики ниже установленного минимального порога ({score_valid_min_threshold:.6f}), валидация этого пайплайна с этим набором гиперпараметров прервана для экономии времени');
                return error_str;
        elif score_valid_max_threshold!=None:
            if score_valid>score_valid_max_threshold:
                print(f'Значение метрики выше установленного максимального порога ({score_valid_max_threshold:.6f}), валидация этого пайплайна с этим набором гиперпараметров прервана для экономии времени');
                return error_str;
        scores_valid.append(score_valid);
    if conf_dict['save_sample_partition']==True:
        sample_partition_df:pd.DataFrame=pd.DataFrame({'num':np.arange(stop=opened_data_len),'id':opened_ids,'part':part_assignment});#Для контроля попадания всех образцов
        #opened_data в один из фолдов кросс-валидации или test_final
        for fold in range(num_folds):sample_partition_df[f'fold{fold}']=(sample_partition_df['part']==f'cv_fold{fold}').astype(int);
        sample_partition_df['test_final']=(sample_partition_df['part']=='test_final').astype(int);
        sample_partition_df.to_parquet(path=f'sample_partition_for_pipeline_{pipeline_id}.parquet',index=False);#Для этого pyarrow или fastparquet
        print(f"Таблица распределения образцов сохранена в файл [sample_partition_for_pipeline_{pipeline_id}.parquet]");

    # 5. Расчет среднего значения метрики по CV (mean) и её среднеквадратического отклонения (std) [mean чем больше тем лучше или чем
    #меньше тем лучше - в зависимости от метрики, std всегда чем меньше тем лучше, так как чем меньше std, тем устойчивее пайплайн]
    score_valid_mean:float=np.mean(a=scores_valid,dtype=np.float64);
    score_valid_std:float=np.std(a=scores_valid,dtype=np.float64,ddof=0);
    print(f"\nСреднее значение метрики {score_type} по кросс-валидации: {score_valid_mean:.6f}, среднеквадратическое отклонение метрики {score_type} по кросс-валидации: {score_valid_std:.6f}");

    # 6. Оценка качества на отложенной выборке (20% открытых данных)
    #print("Оценка качества на отложенной выборке (20% открытых данных)...")

    #6.1. Использование imputer (если выбрано) для for_final_test
    if use_imputer==True:
        if copy_data_arrays==True:#Копируем массивы
            imputer_for_final_test:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
            imputer_for_final_test.fit(X=X_train_cv,y=y_train_cv);
            X_train_cv_imputed=imputer_for_final_test.transform(X=X_train_cv);
            X_test_final_imputed=imputer_for_final_test.transform(X=X_test_final);
        else:#Перезаписываем массивы
            imputer_for_final_test:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
            imputer_for_final_test.fit(X=X_train_cv,y=y_train_cv);
            X_train_cv=imputer_for_final_test.transform(X=X_train_cv);
            X_test_final=imputer_for_final_test.transform(X=X_test_final);
    else:
        if copy_data_arrays==True:#Копируем массивы
            X_train_cv_imputed=X_train_cv.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            X_test_final_imputed=X_test_final.copy();
            imputer_for_final_test=None;
        else:#Перезаписываем массивы:
            imputer_for_final_test=None;#Если не используем imputer и перезаписываем массивы, то ничего не делаем

    #6.2. Использование var_thresholder (если выбрано) для for_final_test
    if use_var_thresholder==True:
        if copy_data_arrays==True:#Копируем массивы
            if var_thresholder_type=='VarianceThreshold':var_thresholder_for_final_test=VarianceThreshold(**var_thresholder_hyperparams);
            var_thresholder_for_final_test.fit(X=X_train_cv_imputed,y=y_train_cv);
            X_train_cv_var_thresholded=var_thresholder_for_final_test.transform(X=X_train_cv_imputed);
            X_test_final_var_thresholded=var_thresholder_for_final_test.transform(X=X_test_final_imputed);
        else:#Перезаписываем массивы
            if var_thresholder_type=='VarianceThreshold':var_thresholder_for_final_test=VarianceThreshold(**var_thresholder_hyperparams);
            var_thresholder_for_final_test.fit(X=X_train_cv,y=y_train_cv);
            X_train_cv=var_thresholder_for_final_test.transform(X=X_train_cv);
            X_test_final=var_thresholder_for_final_test.transform(X=X_test_final);
    else:
        if copy_data_arrays==True:#Копируем массивы
            X_train_cv_var_thresholded=X_train_cv_imputed.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            X_test_final_var_thresholded=X_test_final_imputed.copy();
            var_thresholder_for_final_test=None;
        else:#Перезаписываем массивы
            var_thresholder_for_final_test=None;#Если не используем  var_thresholder и перезаписываем массивы, то ничего не делаем

    #6.3. Использование scaler (если выбрано) для for_final_test
    if use_scaler==True:
        if copy_data_arrays==True:#Копируем массивы
            scaler_for_final_test:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
            scaler_for_final_test.fit(X=X_train_cv_var_thresholded);#fit только на train, без valid
            X_train_cv_scaled=scaler_for_final_test.transform(X=X_train_cv_var_thresholded);
            X_test_final_scaled=scaler_for_final_test.transform(X=X_test_final_var_thresholded);
        else:#Перезаписываем массивы
            scaler_for_final_test:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
            scaler_for_final_test.fit(X=X_train_cv);#fit только на train, без valid
            X_train_cv=scaler_for_final_test.transform(X=X_train_cv);
            X_test_final=scaler_for_final_test.transform(X=X_test_final);
    else:#Если use_scaler==False
        if copy_data_arrays==True:#Копируем массивы
            X_train_cv_scaled=X_train_cv_var_thresholded.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            X_test_final_scaled=X_test_final_var_thresholded.copy();
            scaler_for_final_test=None;
        else:#Перезаписываем массивы
            scaler_for_final_test=None;#Если не используем  scaler и перезаписываем массивы, то ничего не делаем

    #6.4. Использование feature_selector (если выбрано) для for_final_test
    if use_feature_selector==True:
        if feature_selector_type in ['GenericUnivariateSelect','SelectFdr','SelectFpr','SelectFwe','SelectKBest','SelectPercentile']:
            if fs_score_func_type=='f_classif':score_func_for_final_test:callable=f_classif;
            elif fs_score_func_type=='mutual_info_classif':score_func_for_final_test:callable=mutual_info_classif;
            elif fs_score_func_type=='chi2':score_func_for_final_test:callable=chi2;
            elif fs_score_func_type=='f_regression':score_func_for_final_test:callable=f_regression;
            elif fs_score_func_type=='mutual_info_regression':score_func_for_final_test:callable=mutual_info_regression;
            feature_selector_for_final_test:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](**feature_selector_hyperparams);
        elif feature_selector_type in feature_selector_types_estimator_all:
            fs_estimator_for_final_test:AnyFSEstimator=fs_estimator_constructors_dict[fs_estimator_type](**fs_estimator_hyperparams);
            feature_selector_for_final_test:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](estimator=fs_estimator_for_final_test,**feature_selector_hyperparams);
        if copy_data_arrays==True:#Копируем массивы
            feature_selector_for_final_test.fit(X=X_train_cv_scaled,y=y_train_cv);
            X_train_cv_feature_selected=feature_selector_for_final_test.transform(X=X_train_cv_scaled);
            X_test_final_feature_selected=feature_selector_for_final_test.transform(X=X_test_final_scaled);
        else:#Перезаписываем массивы
            feature_selector_for_final_test.fit(X=X_train_cv,y=y_train_cv);
            X_train_cv=feature_selector_for_final_test.transform(X=X_train_cv);
            X_test_final=feature_selector_for_final_test.transform(X=X_test_final);
    else:#Если use_feature_selector==False
        if copy_data_arrays==True:#Копируем массивы
            X_train_cv_feature_selected=X_train_cv_scaled.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            X_test_final_feature_selected=X_test_final_scaled.copy();
            feature_selector_for_final_test=None;
        else:#Перезаписываем массивы
            feature_selector_for_final_test=None;#Если не используем  feature_selector и перезаписываем массивы, то ничего не делаем
    if copy_data_arrays==True:#Копируем массивы
        if (X_train_cv_feature_selected.shape[1]==0)or(X_test_final_feature_selected.shape[1]==0):
            print(f'После применения feature_selector типа {feature_selector_type} на этапе final_test количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
            return error_str;
    else:#Перезаписываем массивы
        if (X_train_cv.shape[1]==0)or(X_test_final.shape[1]==0):
            print(f'После применения feature_selector типа {feature_selector_type} на этапе final_test количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
            return error_str;
    
    #6.5. Использование model (всегда выбрано) для for_final_test
    model_for_final_test:AnyModel=model_constructors_dict[model_type](**model_hyperparams);
    if copy_data_arrays==True:#Копируем массивы
        model_for_final_test.fit(X=X_train_cv_feature_selected,y=y_train_cv);#Обучение на 80% открытых данных
        y_test_pred:np.ndarray=model_for_final_test.predict(X_test_final_feature_selected);#Тестирование на 20% открытых данных
    else:#Перезаписываем массивы
        model_for_final_test.fit(X=X_train_cv,y=y_train_cv);#Обучение на 80% открытых данных
        y_test_pred:np.ndarray=model_for_final_test.predict(X_test_final);#Тестирование на 20% открытых данных
    if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
        if copy_data_arrays==True:#Копируем массивы
            y_test_pred_proba:np.ndarray=model_for_final_test.predict_proba(X_test_final_feature_selected);
        else:#Перезаписываем массивы
            y_test_pred_proba:np.ndarray=model_for_final_test.predict_proba(X_test_final);
        y_test_pred_proba_positive:np.ndarray=y_test_pred_proba[:,1];#вероятность положительного класса для бинарной классификации
        if score_type=='accuracy_score':score_test=accuracy_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='auc':
            if problem_type=='classification_binary':
                fpr_for_final_test:np.ndarray;tpr_for_final_test:np.ndarray;thresholds_for_final_test:np.ndarray;
                fpr_for_final_test,tpr_for_final_test,thresholds_for_final_test=roc_curve(y_true=y_test_final,y_score=y_test_pred,pos_label=1);
                score_test=auc(x=fpr_for_final_test,y=tpr_for_final_test);
            elif problem_type=='classification_multiclass':
                print(f'Для многоклассовой классификации НЕ следует использовать auc, лучше roc_auc_score');
                score_test=0.0;
        elif score_type=='average_precision_score':pass;
        elif score_type=='balanced_accuracy_score':score_test=balanced_accuracy_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='brier_score_loss':#Требует вероятность положительного класса для бинарной классификации
            score_test=brier_score_loss(y_true=y_test_final,y_prob=y_test_pred_proba_positive);
        elif score_type=='cohen_kappa_score':score_test=cohen_kappa_score(y1=y_test_final,y2=y_test_pred);
        elif score_type=='dcg_score':pass;
        elif score_type=='f1_score':score_test=f1_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='fbeta_score':score_test=fbeta_score(y_true=y_test_final,y_pred=y_test_pred,beta=fbeta_score_beta);
        elif score_type=='hamming_loss':score_test=hamming_loss(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='hinge_loss':pass;
        elif score_type=='jaccard_score':score_test=jaccard_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='log_loss':score_test=log_loss(y_true=y_test_final,y_pred=y_test_pred_proba);
        elif score_type=='matthews_corrcoef':score_test=matthews_corrcoef(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='ndcg_score':pass;
        elif score_type=='precision_score':score_test=precision_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='recall_score':score_test=recall_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='roc_auc_score':#Для бинарной классификации берём вероятность ПОЛОЖИТЕЛЬНОГО класса (второй столбец)
            if problem_type=='classification_binary':
                score_test=roc_auc_score(y_true=y_test_final,y_score=y_test_pred_proba_positive);
            elif problem_type=='classification_multiclass':#Для мультикласса можно использовать 'ovr' или 'ovo' стратегии
                score_test=roc_auc_score(y_true=y_test_final,y_score=y_test_pred_proba,multi_class='ovr');
        elif score_type=='zero_one_loss':score_test=zero_one_loss(y_true=y_test_final,y_pred=y_test_pred);
    elif problem_type=='regression':
        if score_type=='d2_absolute_error_score':score_test=d2_absolute_error_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='d2_pinball_score':score_test=d2_pinball_score(y_true=y_test_final,y_pred=y_test_pred,alpha=d2_pinball_score_alpha);
        elif score_type=='d2_tweedie_score':score_test=d2_tweedie_score(y_true=y_test_final,y_pred=y_test_pred,power=d2_tweedie_score_power);
        elif score_type=='explained_variance_score':score_test=explained_variance_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='max_error':score_test=max_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_absolute_percentage_error':score_test=mean_absolute_percentage_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_gamma_deviance':score_test=mean_gamma_deviance(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_pinball_loss':score_test=mean_pinball_loss(y_true=y_test_final,y_pred=y_test_pred,alpha=mean_pinball_loss_alpha);
        elif score_type=='mean_poisson_deviance':score_test=mean_poisson_deviance(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_squared_error':score_test=mean_squared_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_squared_log_error':score_test=mean_squared_log_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='mean_tweedie_deviance':score_test=mean_tweedie_deviance(y_true=y_test_final,y_pred=y_test_pred,power=mean_tweedie_deviance_power);
        elif score_type=='median_absolute_error':score_test=median_absolute_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='r2_score':score_test=r2_score(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='root_mean_squared_error':score_test=root_mean_squared_error(y_true=y_test_final,y_pred=y_test_pred);
        elif score_type=='root_mean_squared_log_error':score_test=root_mean_squared_log_error(y_true=y_test_final,y_pred=y_test_pred);
    print(f"Тип метрики: {score_type}, значение метрики на тесте (отложенная выборка, 20% открытых данных): {score_test:.6f} {cur_dt_to_str()}");
    
    #7. ОБУЧЕНИЕ ФИНАЛЬНОГО ПАЙПЛАЙНА НА 100% ОТКРЫТЫХ ДАННЫХ И СОХРАНЕНИЕ ПАЙПЛАЙНА В *.pkl файл
    print("Обучение финального production-пайплайна на ВСЕХ открытых данных opened_data...");
    X_all:np.ndarray=opened_data.copy();y_all:np.ndarray=opened_target.copy();

    #7.1. Использование imputer (если выбрано) для production
    if use_imputer==True:
        if copy_data_arrays==True:#Копируем массивы
            imputer_production:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
            imputer_production.fit(X=X_all,y=y_all);
            X_all_imputed=imputer_production.transform(X=X_all);
        else:#Перезаписываем массивы
            imputer_production:AnyImputer=imputer_constructors_dict[imputer_type](**imputer_hyperparams);
            imputer_production.fit(X=X_all,y=y_all);
            X_all=imputer_production.transform(X=X_all);
    else:
        if copy_data_arrays==True:#Копируем массивы
            X_all_imputed=X_all.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            imputer_production=None;
        else:#Перезаписываем массивы
            imputer_production=None;#Если не используем imputer и перезаписываем массивы, то ничего не делаем

    #7.2. Использование var_thresholder (если выбрано) для production
    if use_var_thresholder==True:
        if copy_data_arrays==True:#Копируем массивы
            if var_thresholder_type=='VarianceThreshold':var_thresholder_production=VarianceThreshold(**var_thresholder_hyperparams);
            var_thresholder_production.fit(X=X_all_imputed,y=y_all);
            X_all_var_thresholded=var_thresholder_production.transform(X=X_all_imputed);
        else:#Перезаписываем массивы
            if var_thresholder_type=='VarianceThreshold':var_thresholder_production=VarianceThreshold(**var_thresholder_hyperparams);
            var_thresholder_production.fit(X=X_all,y=y_all);
            X_all=var_thresholder_production.transform(X=X_all);
    else:
        if copy_data_arrays==True:#Копируем массивы
            X_all_var_thresholded=X_all_imputed.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            var_thresholder_production=None;
        else:#Перезаписываем массивы
            var_thresholder_production=None;#Если не используем var_thresholder и перезаписываем массивы, то ничего не делаем

    #7.3. Использование scaler (если выбрано) для production
    if use_scaler==True:
        if copy_data_arrays==True:#Копируем массивы
            scaler_production:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
            scaler_production.fit(X=X_all_var_thresholded);
            X_all_scaled=scaler_production.transform(X=X_all_var_thresholded);
        else:#Перезаписываем массивы
            scaler_production:AnyScaler=scaler_constructors_dict[scaler_type](**scaler_hyperparams);
            scaler_production.fit(X=X_all);
            X_all=scaler_production.transform(X=X_all);
    else:
        if copy_data_arrays==True:#Копируем массивы
            X_all_scaled=X_all_var_thresholded.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            scaler_production=None;#Переменной scaler_production нужно что-то присвоить чтобы сохранить в pkl файле
        else:#Перезаписываем массивы
            scaler_production=None;#Если не используем scaler и перезаписываем массивы, то ничего не делаем

    #7.4. Использование feature_selector (если выбрано) для production
    if use_feature_selector==True:
        if feature_selector_type in ['GenericUnivariateSelect','SelectFdr','SelectFpr','SelectFwe','SelectKBest','SelectPercentile']:
            if fs_score_func_type=='f_classif':score_func_production:callable=f_classif;
            elif fs_score_func_type=='mutual_info_classif':score_func_production:callable=mutual_info_classif;
            elif fs_score_func_type=='chi2':score_func_production:callable=chi2;
            elif fs_score_func_type=='f_regression':score_func_production:callable=f_regression;
            elif fs_score_func_type=='mutual_info_regression':score_func_production:callable=mutual_info_regression;
            feature_selector_production:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](**feature_selector_hyperparams);
        elif feature_selector_type in feature_selector_types_estimator_all:
            fs_estimator_production:AnyFSEstimator=fs_estimator_constructors_dict[fs_estimator_type](**fs_estimator_hyperparams);
            feature_selector_production:AnyFeatureSelector=feature_selector_constructors_dict[feature_selector_type](estimator=fs_estimator_production,**feature_selector_hyperparams);
        if copy_data_arrays==True:#Копируем массивы
            feature_selector_production.fit(X=X_all_scaled,y=y_all);
            X_all_feature_selected=feature_selector_production.transform(X=X_all_scaled);
        else:#Перезаписываем массивы
            feature_selector_production.fit(X=X_all,y=y_all);
            X_all=feature_selector_production.transform(X=X_all);
    else:#Если use_feature_selector==False
        if copy_data_arrays==True:#Копируем массивы
            X_all_feature_selected=X_all_scaled.copy();#Без .copy() объекты будут ссылаться на один и тот же объект в памяти
            feature_selector_production=None;#Переменной feature_selector_production нужно что-то присвоить чтобы сохранить в pkl файле
        else:#Перезаписываем массивы
            feature_selector_production=None;#Если не используем feature_selectr и перезаписываем массивы, то ничего не делаем
    if copy_data_arrays==True:#Копируем массивы
        if X_all_feature_selected.shape[1]==0:
            print(f'После применения feature_selector типа {feature_selector_type} на этапе production количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
            return error_str;
    else:#Перезаписываем массивы
        if X_all.shape[1]==0:
            print(f'После применения feature_selector типа {feature_selector_type} на этапе production количество отобранных признаков равно нулю, обработка этого пайплайна прервана');
            return error_str;

    #7.5. Использование model (всегда выбрано) для production
    model_production:AnyModel=model_constructors_dict[model_type](**model_hyperparams);
    if copy_data_arrays==True:#Копируем массивы
        model_production.fit(X=X_all_feature_selected,y=y_all);
    else:#Перезаписываем массивы
        model_production.fit(X=X_all,y=y_all);

    # 8. Генерация ID пайплайна и сохранение Production-пайплайна (update: id сгенерируем раньше для сохранения sample_partition в parquet)
    pipeline_filename:str=f"pipeline_{pipeline_id}.pkl";

    # Сохраняем production-пайплайн (именно production imputer, var_thresholder, scaler, feature_selector, model)
    with open(file=pipeline_filename,mode='wb')as f:pickle.dump(obj={'n_features_selected_randomly':n_features_selected_randomly,'randomly_selected_indexes':randomly_selected_indexes,'imputer':imputer_production,'var_thresholder':var_thresholder_production,'scaler':scaler_production,'feature_selector':feature_selector_production,'fs_score_func_type':fs_score_func_type,'fs_estimator_type':fs_estimator_type,'model':model_production},file=f);
    pipeline_file_size:int=os.path.getsize(filename=pipeline_filename);
    scores_valid_str:str='['+', '.join([f"{s:.6f}" for s in scores_valid])+']';

    seconds_pipe_finish:float=time.time();#Для лога (чтобы вычислить время обработки этого пайплайна)
    seconds_processing:float=seconds_pipe_finish-seconds_pipe_start;

    # 9. Логирование
    log_record_txt:str=f"""
--- pipeline ID: {pipeline_id} ---
n_features_all: {n_features_all}
n_features_selected_randomly: {n_features_selected_randomly}
randomly_selected_indexes: {randomly_selected_indexes}
use_imputer: {use_imputer}
imputer_type: {imputer_type}
imputer_hyperparams: {imputer_hyperparams}
use_var_thresholder: {use_var_thresholder}
var_thresholder_type: {var_thresholder_type}
var_thresholder_hyperparams: {var_thresholder_hyperparams}
use_scaler: {use_scaler}
scaler_type: {scaler_type}
scaler_hyperparams: {scaler_hyperparams}
use_feature_selector: {use_feature_selector}
feature_selector_type: {feature_selector_type}
feature_selector_hyperparams: {feature_selector_hyperparams}
fs_score_func_type: {fs_score_func_type}
fs_estimator_type: {fs_estimator_type}
fs_estimator_hyperparams: {fs_estimator_hyperparams}
model_type: {model_type}
model_hyperparams: {model_hyperparams}
split_random_state: {split_random_state}
score_type: {score_type}
scores_valid: {scores_valid_str}
score_valid_mean: {score_valid_mean}
score_valid_std: {score_valid_std}
score_test: {score_test}
dt_pipe_start_str: {dt_pipe_start_str}
seconds_pipe_start: {seconds_pipe_start}, seconds_pipe_finish: {seconds_pipe_finish}, seconds_processing: {seconds_processing}
pipeline_file_size (bytes): {pipeline_file_size}
---------------------------------------
"""
    #Никакие гиперпараметры (feature_selector_hyperparams, scaler_hyperparams, model_hyperparams) не записываются в csv файл, так как они
    #представляют собой словарь (dict), пары key:value которого записываются через запятую, что нарушило бы консистентность строк csv файла,
    #потому что эти словари у разных пайплайнов имеют разное количество значений (у разных типов моделей разное количество киперпараметров)
    log_record_csv:str=f'{pipeline_id},{n_features_all},{n_features_selected_randomly},{use_imputer},{imputer_type},{use_var_thresholder},{var_thresholder_type},{use_scaler},{scaler_type},{use_feature_selector},{feature_selector_type},{fs_score_func_type},{fs_estimator_type},{model_type},{score_type},{score_valid_mean},{score_valid_std},{score_test},{dt_pipe_start_str},{seconds_processing},{pipeline_file_size}\n';
    print(log_record_txt);
    with open(file='log_pipelines.txt',mode='at',encoding='UTF-8')as log_file:log_file.write(log_record_txt);
    with open(file='log_pipelines.csv',mode='at',encoding='UTF-8')as log_file:log_file.write(log_record_csv);
    return pipeline_id;

def float_list_to_comma_separated_str(float_list:list[float],digits:int=2):
    float_list_buf=list(np.round(np.array(float_list),digits));
    return ','.join([str(x)for x in float_list_buf]);

def int_list_to_comma_separated_str(int_list:list[int]):
    int_list_buf=int_list;
    return ','.join([str(x)for x in int_list_buf]);

def create_predictions_files(problem_type:str,n_classes:int,pipeline_ids:list[str],digits_round_min:int=1,digits_round_max:int=18,print_all_predictions:bool=False,calculate_score_for_closed_data:bool=False,score_type:str=None)->None:
    '''Функция вычисляет предсказания и сохраняет их в файлах tsv, json и npy, позволяя выбирать количество цифр. problem_type: regression, classification_binary, classification_multiclass. n_classes имеет значение только если problem_type==classification_multiclass.'''
    dt_prediction_files_start:datetime.datetime=datetime.datetime.now();#Для лога
    dt_prediction_files_start_str:str=dt_prediction_files_start.strftime(format='%Y-%m-%d_%H-%M-%S');
    seconds_prediction_files_start:float=time.time();#Для лога (чтобы вычислить время вычисления предсказаний)
    n_samples_done_in_cur_pipeline_to_print:int=conf_dict['creating_prediction_files_params']['n_samples_done_in_cur_pipeline_to_print'];
    targets_dict:dict[str:float]={};
    for sample_id in closed_ids:#Изначально всем значениям словаря targets_dict присваивается 0.0
        targets_dict[sample_id]=0.0;
    print(f'targets_dict: {targets_dict}');
    print(f'len(targets_dict): {len(targets_dict)}');
    pipeline_ids_str:str=' '.join(pipeline_ids);
    results_file_id:str=''.join(random.choices(population=string.ascii_uppercase+string.digits,k=16));
    n_pipelines:int=len(pipeline_ids);
    print(f'Предсказания выполняются усреднением результатов для n_pipelines={n_pipelines} пайплайнов со значениями id: {pipeline_ids}');
    buf_s:str='';buf_s_proba:str='';buf_s_labels:str='';
    n_samples_opened:int=opened_ids.shape[0];
    n_samples_closed:int=closed_ids.shape[0];    
    #Изначально предсказания для всех образцов инициализируем нулями так как 0+a=a для любого действительного числа a, каждое
    #предсказание - это число с плавающей точкой (которое обозначает target для regression или вероятность того, что target
    #принадлежит положительному классу для classification_binary) или список из n_classes чисел с плавающей точкой для
    #classification_multiclass.
    if problem_type=='regression':
        targets_list:list[float]=[0.0 for i in range(n_samples_closed)];
    elif problem_type=='classification_binary':
        targets_list_proba:list[float]=[0.0 for i in range(n_samples_closed)];
        targets_list_logits:list[float]=[0.0 for i in range(n_samples_closed)];
        targets_list_labels:list[float]=[0.0 for i in range(n_samples_closed)];
    elif problem_type=='classification_multiclass':
        targets_list_proba:list[list[float]]=[[0.0 for j in range(n_classes)]for i in range(n_samples_closed)];
        targets_list_logits:list[list[float]]=[[0.0 for j in range(n_classes)]for i in range(n_samples_closed)];
        targets_list_labels:list[float]=[0.0 for i in range(n_samples_closed)];#Для многоклассовой классификации список вероятностей вложенный, но список меток нет
    if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
        n_pipelines_with_predict_proba:int=0;
        pipeline_ids_with_predict_proba:list[str]=[];
        count_mean_for_logits:bool=str_or_bool_to_bool(s=conf_dict['creating_prediction_files_params']['count_mean_for_logits']);#Вычислять среднее для вероятностей или для логитов
    pipeline_inference_times:dict[str,dict[str,float]]={};#Для сохранения времён инференса каждого из пайплайнов по отдельности
    #Суммирование предсказаний пайплайнов:
    for pipeline_id in pipeline_ids:#Перебираем пайплайны
        pkl_file_name:str=f'pipeline_{pipeline_id}.pkl';
        with open(file=pkl_file_name,mode='rb')as f:pipeline_dict:dict=pickle.load(file=f);
        seconds_pipe_infer_start:float=time.time();
        print(f'pipeline_id: {pipeline_id}, pipeline_dict: {pipeline_dict}');
        n_features_selected_randomly:int=pipeline_dict['n_features_selected_randomly'];
        randomly_selected_indexes:list[int]=pipeline_dict['randomly_selected_indexes'];
        imputer:AnyImputer=pipeline_dict['imputer'];#imputer может быть None
        var_thresholder:VarianceThreshold=pipeline_dict['var_thresholder'];#var_thresholder может быть None
        scaler:AnyScaler|None=pipeline_dict['scaler'];#scaler может быть None
        feature_selector:AnyFeatureSelector|None=pipeline_dict['feature_selector'];#feature_selector может быть None
        if problem_type=='regression':
            model:AnyModelRegressor=pipeline_dict['model'];#model не может быть None
        elif (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
            model:AnyModelClassifier=pipeline_dict['model'];#model не может быть None
        print(f'pipeline_id: {pipeline_id}');
        if conf_dict['creating_prediction_files_params']['print_model_dict']==True:
            print(f'model.__dict__: {model.__dict__}');#model точно не None
        if feature_selector is not None:print(f'feature_selector.__dict__: {feature_selector.__dict__}');
        else:print(f'feature_selector: {feature_selector}');
        if scaler is not None:print(f'scaler.__dict__: {scaler.__dict__}');
        else:print(f'scaler: {scaler}');        
        num_processed:int=0;
        opened_data:np.ndarray=np.zeros((n_samples_opened,n_features_selected_randomly));#Создание новых массивов
        closed_data:np.ndarray=np.zeros((n_samples_closed,n_features_selected_randomly));
        #Использование списка randomly_selected_indexes и его сохранение в лог делает эсперименты воспроизводимыми
        for col_idx,feature_idx in enumerate(randomly_selected_indexes):#Копирование столбцов с выбранными индексами из исходных массивов
            opened_data[:,col_idx]=opened_data_all_features[:,feature_idx];
            closed_data[:,col_idx]=closed_data_all_features[:,feature_idx];
        #Добавление элементов в список pipeline_ids_with_predict_proba переносим СЮДА (выносим его из цикла по n_samples_closed, иначе n_pipelines_with_predict_proba
        #будет в n_samples_closed раз больше, чем должно быть, а все вероятности станут в n_samples_closed раз меньше):
        if problem_type=='classification_binary':
            n_pipelines_with_predict_proba=n_pipelines_with_predict_proba+1;
            pipeline_ids_with_predict_proba.append(pipeline_id);
        elif problem_type=='classification_multiclass':
            n_pipelines_with_predict_proba=n_pipelines_with_predict_proba+1;
            pipeline_ids_with_predict_proba.append(pipeline_id);
        n_samples_done_in_cur_pipeline:int=0;#Обнуляем счётчик количества обработанных текущим пайплайном образцов ПЕРЕД началом перебора образцов
        for sample_num in range(n_samples_closed):#Перебираем образцы закрытых данных
            sample_id:str=closed_ids[sample_num];
            features:np.ndarray=closed_data[sample_num].reshape(1,-1);#features - это признаковое описание ОДНОГО образца closed_data, а не вся матрица
            #1. Использование imputer (если есть в пайплайне) для prediction
            if imputer is not None:features_imputed=imputer.transform(X=features);
            else:features_imputed=features.copy();
            #2. Использование var_thresholder (если есть в пайплайне) для prediction
            if var_thresholder is not None:features_var_thresholded=var_thresholder.transform(X=features_imputed);
            else:features_var_thresholded=features_imputed.copy();
            #3. Использование scaler (если есть в пайплайне) для prediction
            if scaler is not None:features_scaled=scaler.transform(X=features_var_thresholded);
            else:features_scaled=features_var_thresholded.copy();
            #4. Использование feature_selector (если есть в пайплайне) для prediction
            if feature_selector is not None:features_feature_selected=feature_selector.transform(X=features_scaled);
            else:features_feature_selected=features_scaled.copy();
            #5. Использование model (всегда есть в пайплайне) для prediction
            if problem_type=='regression':
                target_predicted:float=model.predict(features_feature_selected)[0];
            elif problem_type=='classification_binary':
                target_predicted_label:float=model.predict(features_feature_selected)[0];
                if hasattr(model,'predict_proba'):#hasattr() takes no keyword arguments
                    target_predicted_proba:float=model.predict_proba(features_feature_selected)[0][1];#По соглашению scikit-learn
                    #положительный класс — это второй класс в отсортированном порядке меток (model.classes_[1])
                    #n_pipelines_with_predict_proba=n_pipelines_with_predict_proba+1;
                    #pipeline_ids_with_predict_proba.append(pipeline_id);
                else:
                    print(f'Пайплайн с id={pipeline_id} имеет модель класса {type(model)}, у этого класса нет атрибута (метода) predict_proba, поэтому этот пайплайн не учитывается при усреднении предсказаний');
                    target_predicted_proba:float=0.0;#Если этот пайпайн не учитывается, то прибавляем 0, так как a+0=a
            elif problem_type=='classification_multiclass':#Returns:p - ndarray of shape (n_samples,n_classes)
                target_predicted_label:float=model.predict(features_feature_selected)[0];
                if hasattr(model,'predict_proba'):#hasattr() takes no keyword arguments
                    target_predicted_proba:list[float]=[float(a)for a in model.predict_proba(features_feature_selected)[0]];
                    #n_pipelines_with_predict_proba=n_pipelines_with_predict_proba+1;
                    #pipeline_ids_with_predict_proba.append(pipeline_id);
                else:
                    print(f'Пайплайн с id={pipeline_id} имеет модель класса {type(model)}, у этого класса нет атрибута (метода) predict_proba, поэтому этот пайплайн не учитывается при усреднении предсказаний');
                    target_predicted_proba:list[float]=[0.0 for i in range(n_classes)];
            num_processed=num_processed+1;
            #targets_list.append(target_predicted);
            if problem_type=='regression':#Для регрессии target_predicted:float
                targets_list[sample_num]=targets_list[sample_num]+target_predicted;
            elif problem_type=='classification_binary':#Для бинарной классификации target_predicted_proba:float
                if count_mean_for_logits==True:#Вычисляем среднее для логитов (тут суммируем, чтобы потом поделить на количество):
                    #На этой строке [targets_list_logits[sample_num]=targets_list_logits[sample_num]+logit_float(x=target_predicted_proba);] иногда возникает такая ошибка:
                    #TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'
                    #targets_list_logits НЕ МОЖЕТ быть None, так как эта переменная всегда инициализируется. Добавим вывод для отладки:
                    #Проблема оказалась в пайпайне 3GHRO2K704L7S4HC, logit_float(x=target_predicted_proba): None
                    #target_predicted_proba: 0.49999094009399414, type(target_predicted_proba): <class 'numpy.float32'>
                    if (targets_list_logits[sample_num] is None)or(logit_float(x=target_predicted_proba,prob_eps=prob_eps) is None):
                        print(f'Одно из значений targets_list_logits[sample_num] или logit_float(x=target_predicted_proba) является None');
                        print(f'pipeline_id: {pipeline_id}, problem_type: {problem_type}, sample_num: {sample_num}');
                        print(f'type(targets_list_logits[sample_num]): {type(targets_list_logits[sample_num])}');
                        print(f'type(logit_float(x=target_predicted_proba,prob_eps=prob_eps)): {type(logit_float(x=target_predicted_proba))}');
                        print(f'targets_list_logits[sample_num]: {targets_list_logits[sample_num]}');
                        print(f'logit_float(x=target_predicted_proba,prob_eps=prob_eps): {logit_float(x=target_predicted_proba,prob_eps=prob_eps)}');
                        print(f'target_predicted_proba: {target_predicted_proba}, type(target_predicted_proba): {type(target_predicted_proba)}');                    
                    targets_list_logits[sample_num]=targets_list_logits[sample_num]+logit_float(x=target_predicted_proba);
                else:#Вычисляем среднее для вероятностей (тут суммируем, чтобы потом поделить на количество):
                    targets_list_proba[sample_num]=targets_list_proba[sample_num]+target_predicted_proba;
            elif problem_type=='classification_multiclass':#Для многоклассовой классификации target_predicted_proba:list[float]
                if count_mean_for_logits==True:#Вычисляем среднее для логитов (тут суммируем, чтобы потом поделить на количество):
                    for i in range(n_classes):targets_list_logits[sample_num][i]=targets_list_logits[sample_num][i]+logit_float(x=target_predicted_proba[i],prob_eps=prob_eps);
                else:#Вычисляем среднее для вероятностей (тут суммируем, чтобы потом поделить на количество):
                    for i in range(n_classes):targets_list_proba[sample_num][i]=targets_list_proba[sample_num][i]+target_predicted_proba[i];
            if print_all_predictions==True:
                print(f'sample_num: {sample_num:5d}/[0..{(n_samples_closed-1):5d}], sample_id: {sample_id}, target_predicted_proba: {target_predicted_proba}, type(target_predicted_proba): {type(target_predicted_proba)}');
            n_samples_done_in_cur_pipeline=n_samples_done_in_cur_pipeline+1;
            if (n_samples_done_in_cur_pipeline_to_print is None)==False:
                if (n_samples_done_in_cur_pipeline%n_samples_done_in_cur_pipeline_to_print==0)or(n_samples_done_in_cur_pipeline==n_samples_closed):
                    print(f'Пайплайн с id={pipeline_id} применён к {n_samples_done_in_cur_pipeline}/{n_samples_closed} образцов закрытых данных {cur_dt_to_str()}');
        seconds_pipe_infer_finish:float=time.time();
        seconds_pipe_infer_last_for_n_samples_closed:float=seconds_pipe_infer_finish-seconds_pipe_infer_start;
        seconds_pipe_infer_last_per_sample:float=seconds_pipe_infer_last_for_n_samples_closed/n_samples_closed;
        pipeline_inference_times[pipeline_id]={'start':seconds_pipe_infer_start,'finish':seconds_pipe_infer_finish,'for_all':seconds_pipe_infer_last_for_n_samples_closed,'per_sample':seconds_pipe_infer_last_per_sample};
    #Приведение списка targets_list из list[numpy.float64] в list[float]
    if problem_type=='regression':
        for sample_num in range(n_samples_closed):targets_list[sample_num]=float(targets_list[sample_num]);
    elif problem_type=='classification_binary':
        for sample_num in range(n_samples_closed):targets_list_proba[sample_num]=float(targets_list_proba[sample_num]);
    elif problem_type=='classification_multiclass':
        for sample_num in range(n_samples_closed):
            for i in range(n_classes):targets_list_proba[sample_num][i]=float(targets_list_proba[sample_num][i]);
    #Усреднение предсказаний пайплайнов:
    if ((problem_type=='classification_binary')or(problem_type=='classification_multiclass'))and(n_pipelines_with_predict_proba==0):
        n_pipelines_with_predict_proba:int=-1;#Если в каждом из пайплайнов модель не имеет атрибута predict_proba, значит все
        #предсказания (вероятности) равны 0.0. Минус единица присваивается только для того, чтобы не было деления на ноль.
        no_predict_proba_warn_str:str=f'МОДЕЛЬ КАЖДОГО ИЗ ПАЙПЛАЙНОВ НЕ ИМЕЕТ АТРИБУТА predict_proba, ИЗ-ЗА ЧЕГО УСРЕДНИТЬ ВЕРОЯТНОСТИ ПРЕДСКАЗАНИЙ НЕВОЗМОЖНО';
        print(no_predict_proba_warn_str);
    if problem_type=='regression':#Результат - одномерный массив для записи в npy
        closed_target_predicted:np.ndarray=np.zeros(shape=(n_samples_closed,),dtype=np.float64);
    elif problem_type=='classification_binary':#Результат - два одномерных массива для записи в npy (метки и вероятности)
        closed_target_predicted_labels:np.ndarray=np.zeros(shape=(n_samples_closed,),dtype=np.float64);
        closed_target_predicted_proba:np.ndarray=np.zeros(shape=(n_samples_closed,),dtype=np.float64);
    elif problem_type=='classification_multiclass':#Результат - одномерный массив меток и двумерный массив вероятностей для записи в npy
        closed_target_predicted_labels:np.ndarray=np.zeros(shape=(n_samples_closed,),dtype=np.float64);
        closed_target_predicted_proba:np.ndarray=np.zeros(shape=(n_samples_closed,n_classes),dtype=np.float64);
    for sample_num in range(n_samples_closed):
        if problem_type=='regression':
            targets_list[sample_num]=targets_list[sample_num]/n_pipelines;#Деление суммы предсказаний пайплайнов на количество пайплайнов
            closed_target_predicted[sample_num]=np.float64(targets_list[sample_num]);
            buf_s=buf_s+closed_ids[sample_num]+'\t'+str(targets_list[sample_num])+'\n';#Добавление информации в строку для вывода в tsv файл
        elif problem_type=='classification_binary':
            if count_mean_for_logits==True:#Вычисляем среднее для логитов
                targets_list_proba[sample_num]=sigmoid_float(x=targets_list_logits[sample_num]/n_pipelines_with_predict_proba);#Учёт только тех пайплайнов, в model которых есть атрибут predict_proba
            else:#Вычисляем среднее для вероятностей
                targets_list_proba[sample_num]=targets_list_proba[sample_num]/n_pipelines_with_predict_proba;
            closed_target_predicted_proba[sample_num]=np.float64(targets_list_proba[sample_num]);
            #Тут нужно получить массив closed_target_predicted_labels из closed_target_predicted_proba, вероятно нужно передать в функцию пороговое значение
            #вероятности, при превышении которого метка 1
            buf_s_proba=buf_s_proba+closed_ids[sample_num]+'\t'+str(targets_list_proba[sample_num])+'\n';
            buf_s_labels=buf_s_labels+closed_ids[sample_num]+'\t'+str(targets_list_labels[sample_num])+'\n';
        elif problem_type=='classification_multiclass':
            if count_mean_for_logits==True:#Вычисляем среднее для логитов
                for i in range(n_classes):targets_list_proba[sample_num][i]=sigmoid_float(x=targets_list_logits[sample_num][i]/n_pipelines_with_predict_proba);
            else:#Вычисляем среднее для вероятностей
                for i in range(n_classes):targets_list_proba[sample_num][i]=targets_list_proba[sample_num][i]/n_pipelines_with_predict_proba;
            for class_num in range(n_classes):closed_target_predicted_proba[sample_num][class_num]=np.float64(targets_list[sample_num][class_num]);
            #Тут нужно получить массив closed_target_predicted_labels из closed_target_predicted_proba
            buf_s_proba=buf_s_proba+closed_ids[sample_num]+'\t'+' '.join([str(a)for a in targets_list_proba[sample_num]])+'\n';
            buf_s_labels=buf_s_labels+closed_ids[sample_num]+'\t'+' '.join([str(a)for a in targets_list_labels[sample_num]])+'\n';
    if problem_type=='regression':
        tsv_filename:str=f'result_{results_file_id}.tsv';
        with open(file=tsv_filename,mode='wt',encoding='UTF-8')as tsv_file:tsv_file.write(buf_s);
        npy_filename:str=f'result_{results_file_id}_closed_target.npy';
        np.save(file=npy_filename,arr=closed_target_predicted);
    elif problem_type=='classification_binary':
        tsv_filename_proba:str=f'result_{results_file_id}_proba.tsv';
        tsv_filename_labels:str=f'result_{results_file_id}_labels.tsv';
        with open(file=tsv_filename_proba,mode='wt',encoding='UTF-8')as tsv_file:tsv_file.write(buf_s_proba);
        with open(file=tsv_filename_labels,mode='wt',encoding='UTF-8')as tsv_file:tsv_file.write(buf_s_labels);
        npy_filename_proba:str=f'result_{results_file_id}_closed_target_proba.npy';
        npy_filename_labels:str=f'result_{results_file_id}_closed_target_labels.npy';
        np.save(file=npy_filename_proba,arr=closed_target_predicted_proba);
        np.save(file=npy_filename_labels,arr=closed_target_predicted_labels);
    elif problem_type=='classification_multiclass':
        tsv_filename_proba:str=f'result_{results_file_id}_proba.tsv';
        tsv_filename_labels:str=f'result_{results_file_id}_labels.tsv';
        with open(file=tsv_filename_proba,mode='wt',encoding='UTF-8')as tsv_file:tsv_file.write(buf_s_proba);
        with open(file=tsv_filename_labels,mode='wt',encoding='UTF-8')as tsv_file:tsv_file.write(buf_s_labels);
        npy_filename_proba:str=f'result_{results_file_id}_closed_target_proba.npy';
        npy_filename_labels:str=f'result_{results_file_id}_closed_target_labels.npy';
        np.save(file=npy_filename_proba,arr=closed_target_predicted_proba);
        np.save(file=npy_filename_labels,arr=closed_target_predicted_labels);
    #targets_ndarray:np.ndarray=np.ndarray(shape=(n_samples_closed,),dtype=np.float32);
    #for i in range(n_samples_closed):targets_ndarray[i]=targets_list[i];
    #targets_ndarray=targets_ndarray.round(decimals=2);
    if problem_type=='regression':
        for dig in range(digits_round_min,digits_round_max+1):#Чтобы можно было создать несколько json файлов с разным количеством цифр
            predictions_str:str=float_list_to_comma_separated_str(float_list=targets_list,digits=dig);
            if conf_dict['creating_prediction_files_params']['print_targets_list']==True:
                print(f'targets_list: {targets_list}, type(targets_list): {type(targets_list)}, targets_list[0]: {targets_list[0]}, type(targets_list[0]): {type(targets_list[0])}');
            json_dict:dict={'predictions':predictions_str};
            json_filename:str=f'result_{results_file_id}_{dig}_dig.json';
            with open(file=json_filename,mode='wt',encoding='UTF-8')as json_file:json.dump(obj=json_dict,fp=json_file);
    elif problem_type=='classification_binary':
        for dig in range(digits_round_min,digits_round_max+1):#Чтобы можно было создать несколько json файлов с разным количеством цифр для вероятностей
            predictions_str_proba:str=float_list_to_comma_separated_str(float_list=targets_list_proba,digits=dig);
            if conf_dict['creating_prediction_files_params']['print_targets_list']==True:
                print(f'targets_list_proba: {targets_list_proba}, type(targets_list_proba): {type(targets_list_proba)}, targets_list_proba[0]: {targets_list_proba[0]}, type(targets_list_proba[0]): {type(targets_list_proba[0])}');
            json_dict_proba:dict={'predictions_proba':predictions_str_proba};
            json_filename_proba:str=f'result_{results_file_id}_{dig}_dig_proba.json';
            with open(file=json_filename_proba,mode='wt',encoding='UTF-8')as json_file:json.dump(obj=json_dict_proba,fp=json_file);
        #Для меток не будем делить на количества цифр
        predictions_str_labels:str=int_list_to_comma_separated_str(int_list=targets_list_labels);
        if conf_dict['creating_prediction_files_params']['print_targets_list']==True:
            print(f'targets_list_labels: {targets_list_labels}, type(targets_list_labels): {type(targets_list_labels)}, targets_list_labels[0]: {targets_list_labels[0]}, type(targets_list_labels[0]): {type(targets_list_labels[0])}');
        json_dict_labels:dict={'predictions_labels':predictions_str_labels};
        json_filename_labels:str=f'result_{results_file_id}_labels.json';
        with open(file=json_filename_labels,mode='wt',encoding='UTF-8')as json_file:json.dump(obj=json_dict_labels,fp=json_file);
    elif problem_type=='classification_multiclass':
        pass;#Возможно позже стоит добавить сюда код для записи вероятностей и меток многоклассовой классификации в json файлы
    n_pipeline_ids:int=len(pipeline_ids);
    if (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
        pipeline_ids_with_predict_proba_str:str=' '.join(pipeline_ids_with_predict_proba);
    elif problem_type=='regression':
        pipeline_ids_with_predict_proba_str:str=f'problem_type=regression => у моделей нет метода predict_proba';
        n_pipelines_with_predict_proba:int=0;
    seconds_prediction_files_end:float=time.time();#Для лога (чтобы вычислить время вычисления предсказаний)
    seconds_prediction_files_creating:float=seconds_prediction_files_end-seconds_prediction_files_start;
    if problem_type=='regression':
        print(f'Пайплайны с id {pipeline_ids_str} применены, их усреднённые результаты в файлах {tsv_filename}, result_{results_file_id}_{digits_round_min}_dig.json, ..., result_{results_file_id}_{digits_round_max}_dig.json, {npy_filename}');
    elif problem_type=='classification_binary':
        print(f'Пайплайны с id {pipeline_ids_str} применены, их усреднённые результаты (вероятности) в файлах {tsv_filename_proba}, result_{results_file_id}_{digits_round_min}_dig_proba.json, ..., result_{results_file_id}_{digits_round_max}_dig_proba.json, {npy_filename_proba}; предсказанные метки в файлах {tsv_filename_labels}, result_{results_file_id}_{digits_round_min}_dig_labels.json, ..., result_{results_file_id}_{digits_round_max}_dig_labels.json, {npy_filename_labels}');
    elif problem_type=='classification_multiclass':
        print(f'Пайплайны с id {pipeline_ids_str} применены, их усреднённые результаты (вектора вероятностей) в файлах {tsv_filename_proba}, {npy_filename_proba}; предсказанные метки в файлах {tsv_filename_labels}, {npy_filename_labels}');
    score_for_closed_data_log_str:str=f'Выбрано не вычислять значения target для закрытых данных. Возможно значений target для закрытых данных нет. Тогда для оценивания качества разработанного ансамбля пайплайнов нужно загрузить полученный файл с предсказаниями в проверяющую систему. Если значения target для закрытых данных есть, нужно изменить параметр calculate_score_for_closed_data в файле конфигурации и указать правильное название файла с истинными таргетами для закрытых данных.';
    if calculate_score_for_closed_data==True:#Если значение score для closed_data нужно вычислить самому, а не отправлять свои предсказания в проверяющую систему
        closed_target_npy_filename:str=conf_dict['data_files_names']['closed_target_npy'];
        if os.path.exists(path=closed_target_npy_filename):
            closed_target_actual:np.ndarray=np.load(file=closed_target_npy_filename);
            if problem_type=='regression':
                if score_type=='d2_absolute_error_score':score_closed=d2_absolute_error_score(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='d2_pinball_score':score_closed=d2_pinball_score(y_true=closed_target_actual,y_pred=closed_target_predicted,alpha=d2_pinball_score_alpha);
                elif score_type=='d2_tweedie_score':score_closed=d2_tweedie_score(y_true=closed_target_actual,y_pred=closed_target_predicted,power=d2_tweedie_score_power);
                elif score_type=='explained_variance_score':score_closed=explained_variance_score(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='max_error':score_closed=max_error(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='mean_absolute_percentage_error':score_closed=mean_absolute_percentage_error(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='mean_gamma_deviance':score_closed=mean_gamma_deviance(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='mean_pinball_loss':score_closed=mean_pinball_loss(y_true=closed_target_actual,y_pred=closed_target_predicted,alpha=mean_pinball_loss_alpha);
                elif score_type=='mean_poisson_deviance':score_closed=mean_poisson_deviance(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='mean_squared_error':score_closed=mean_squared_error(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='mean_squared_log_error':score_closed=mean_squared_log_error(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='mean_tweedie_deviance':score_closed=mean_tweedie_deviance(y_true=closed_target_actual,y_pred=closed_target_predicted,power=mean_tweedie_deviance_power);
                elif score_type=='median_absolute_error':score_closed=median_absolute_error(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='r2_score':score_closed=r2_score(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='root_mean_squared_error':score_closed=root_mean_squared_error(y_true=closed_target_actual,y_pred=closed_target_predicted);
                elif score_type=='root_mean_squared_log_error':score_closed=root_mean_squared_log_error(y_true=closed_target_actual,y_pred=closed_target_predicted);
            elif (problem_type=='classification_binary')or(problem_type=='classification_multiclass'):
                #Строка closed_target_predicted_proba_positive:np.ndarray=closed_target_predicted_proba[:,1] вывела ошибку:
                #IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed, поэтому для отладки:
                print(f'closed_target_predicted_proba.shape: {closed_target_predicted_proba.shape}, closed_target_predicted_proba: {closed_target_predicted_proba}');
                #closed_target_predicted_proba.shape: (5600,), closed_target_predicted_proba: [9.99977878e-13 9.99977878e-13 9.99977878e-13 ... 1.00000000e+00 1.00000000e+00 1.00000000e+00]
                if closed_target_predicted_proba.ndim==2:
                    closed_target_predicted_proba_positive:np.ndarray=closed_target_predicted_proba[:,1];#вероятность положительного класса для бинарной классификации
                elif closed_target_predicted_proba.ndim==1:
                    closed_target_predicted_proba_positive:np.ndarray=closed_target_predicted_proba[:];
                if score_type=='accuracy_score':score_closed=accuracy_score(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
                elif score_type=='auc':
                    if problem_type=='classification_binary':
                        fpr_closed:np.ndarray;tpr_closed:np.ndarray;thresholds_closed:np.ndarray;
                        fpr_closed,tpr_closed,thresholds_closed=roc_curve(y_true=closed_target_actual,y_score=closed_target_predicted_labels,pos_label=1);
                        score_closed=auc(x=fpr_closed,y=tpr_closed);
                    elif problem_type=='classification_multiclass':
                        print(f'Для многоклассовой классификации НЕ следует использовать auc, лучше roc_auc_score');
                        score_closed=0.0;
                elif score_type=='average_precision_score':pass;
                elif score_type=='balanced_accuracy_score':score_closed=balanced_accuracy_score(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
                elif score_type=='brier_score_loss':#Требует вероятность положительного класса для бинарной классификации
                    score_closed=brier_score_loss(y_true=closed_target_actual,y_prob=closed_target_predicted_proba_positive);
                elif score_type=='cohen_kappa_score':score_closed=cohen_kappa_score(y1=closed_target_actual,y2=closed_target_predicted_labels);
                elif score_type=='dcg_score':pass;
                elif score_type=='f1_score':score_closed=f1_score(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
                elif score_type=='fbeta_score':score_closed=fbeta_score(y_true=closed_target_actual,y_pred=closed_target_predicted_labels,beta=fbeta_score_beta);
                elif score_type=='hamming_loss':score_closed=hamming_loss(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
                elif score_type=='hinge_loss':pass;
                elif score_type=='jaccard_score':score_closed=jaccard_score(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
                elif score_type=='log_loss':score_closed=log_loss(y_true=closed_target_actual,y_pred=closed_target_predicted_proba);#log_loss требует полную матрицу вероятностей (n_samples,n_classes)
                elif score_type=='matthews_corrcoef':score_closed=matthews_corrcoef(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
                elif score_type=='ndcg_score':pass;
                elif score_type=='precision_score':score_closed=precision_score(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
                elif score_type=='recall_score':score_closed=recall_score(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
                elif score_type=='roc_auc_score':#Для бинарной классификации берём вероятность ПОЛОЖИТЕЛЬНОГО класса (второй столбец)
                    if problem_type=='classification_binary':
                        score_closed=roc_auc_score(y_true=closed_target_actual,y_score=closed_target_predicted_proba_positive);
                    elif problem_type=='classification_multiclass':#Для мультикласса можно использовать 'ovr' или 'ovo' стратегии
                        score_closed=roc_auc_score(y_true=closed_target_actual,y_score=closed_target_predicted_proba,multi_class='ovr');
                elif score_type=='zero_one_loss':score_closed=zero_one_loss(y_true=closed_target_actual,y_pred=closed_target_predicted_labels);
            print(f'score_type: {score_type}, score_closed: {score_closed} (значение метрики на закрытых данных)');
            score_for_closed_data_log_str:str=f'''Вычислено значение score для закрытых данных (score_closed) для ансамбля из этого набора пайплайнов. score_type: {score_type}, score_closed: {score_closed}
''';
        else:
            raise(ValueError(f'Файл [{closed_target_npy_filename}] со значениями таргетов для closed_data не существует. Или он должен существовать, или необходимо установить calculate_score_for_closed_data=False.'));
        inference_times_log_str:str=f"""Времена инференса пайпайнов (в секундах), без времён усреднения результатов и вычисления метрики для closed_data (если выбрано эту метрику вычислять):""";
        for pipeline_id in pipeline_ids:inference_times_log_str=inference_times_log_str+f'\npipeline_id: {pipeline_id}, pipeline_inference_times: {pipeline_inference_times[pipeline_id]}';
        results_log_record:str=f"""
results_file_id: {results_file_id}
dt_prediction_files_start_str: {dt_prediction_files_start_str}
seconds_prediction_files_creating: {seconds_prediction_files_creating}
digits_round_min: {digits_round_min}, digits_round_max: {digits_round_max}
n_pipeline_ids: {n_pipeline_ids}
pipeline_ids_str: {pipeline_ids_str}
n_pipelines_with_predict_proba: {n_pipelines_with_predict_proba}
pipeline_ids_with_predict_proba_str: {pipeline_ids_with_predict_proba_str}
{inference_times_log_str}
{score_for_closed_data_log_str}
================""";
    with open(file='log_results.txt',mode='at',encoding='UTF-8')as results_lof_file:print(results_log_record,file=results_lof_file);

def create_coefs_and_bias_files(pipeline_ids:list[str],digits_round_min:int=2,digits_round_max:int=18)->None:
    """Функция принимает список id пайплайнов с одинаковыми значениями n_features_selected_randomly, randomly_selected_indexes, scaler (mean,var,
    scale) и создаёт txt файл, в который записывает усреднённые значения coef: array([...]) (массив из n_features_selected_randomly чисел
    типа float) и intercept.\n\nПонятно, что эта функция имеет смысл ТОЛЬКО в том случае, когда по условию задачи необходимо, чтобы
    модель была линейной. Вообще ансамблирование только линейных моделей всегда оставляет результирующую модель в классе линейных,
    поэтому если только условие об обязательной линейной модели не стоит, лучше НЕ использовать эту функцию"""
    pipelines_num:int=len(pipeline_ids);#Количество пайплайнов, по которым производится усреднение их coef и intercept
    #Считываем первый пайплайн для инициализации некоторых значений:
    print(f'ИНИЦИАЛИЗАЦИЯ НЕКОТОРЫХ ЗНАЧЕНИЙ ПЕРЕД СУММИРОВАНИЕМ coefs и bias:');
    pipeline_id:str=pipeline_ids[0];
    pkl_file_name:str=f'pipeline_{pipeline_id}.pkl';
    with open(file=pkl_file_name,mode='rb')as f:pipeline_dict:dict=pickle.load(file=f);
    print(f'pipeline_id: {pipeline_id}, pipeline_dict: {pipeline_dict}');
    n_features_selected_randomly:int=pipeline_dict['n_features_selected_randomly'];#n_features_selected_randomly одинаковое у всех пайплайнов
    randomly_selected_indexes:list[int]=pipeline_dict['randomly_selected_indexes'];#randomly_selected_indexes одинаковое у всех пайплайнов
    print(f'n_features_selected_randomly: {n_features_selected_randomly}, randomly_selected_indexes: {randomly_selected_indexes}');
    model=pipeline_dict['model'];
    scaler=pipeline_dict['scaler'];
    print(f'pipeline_id: {pipeline_id}');
    print(f'model.__dict__: {model.__dict__}');#model точно не None
    if scaler is not None:print(f'scaler.__dict__: {scaler.__dict__}');
    else:print(f'scaler: {scaler}');
    coefs_total:list[float]=[0.0 for i in range(n_features_selected_randomly)];#Инициализация массива весовых коэффициентов
    bias_total:float=0.0;#Инициализация смещения (intercept или bias) [все эти значения инициализируются нулями, так как a+0.0=a]
    
    #Считывание пайплайнов в цикле для суммирования соответствующих весовых коэффициентов coefs и смещения bias
    num_pipelines_added:int=0;
    num_pipelines_have_coef_and_intercept_attributes:int=0;
    for pipeline_id in pipeline_ids:
        pkl_file_name:str=f'pipeline_{pipeline_id}.pkl';
        print(f'Добавление пайплайна номер {num_pipelines_added+1} из {len(pipeline_ids)}, pipeline_id: {pipeline_id}, file_name: {pkl_file_name}');
        with open(file=pkl_file_name,mode='rb')as f:pipeline_dict:dict=pickle.load(file=f);
        #!!!Для пайплайнов scaler в общем случае не обязан быть одинаковым, поэтому skaler для каждого пайплайна считываем
        scaler_current=pipeline_dict['scaler'];#заново (из обрабатываемого в данный момент пайплайна)
        
        scaler_mean_list_float:list[float]=[float(num)for num in scaler_current.__dict__['mean_']];
        scaler_var_list_float:list[float]=[float(num)for num in scaler_current.__dict__['var_']];
        scaler_scale_list_float:list[float]=[float(num)for num in scaler_current.__dict__['scale_']];
        scaler_scale_sqr_list_float:list[float]=[num**2 for num in scaler_scale_list_float];#Для проверки того, что var - это дисперсия,
        #а scale - это квадратный корень из дисперсии (или 1.0, если дисперсия равна 0.0)

        model_current=pipeline_dict['model'];
        coefs_current:np.ndarray=np.zeros(shape=(n_features_selected_randomly),dtype=np.float64);#Инициализация массива текущих считанных из
        #pkl файла коэффициентов (на тот случай, если такого атрибута нет)
        bias_current:np.ndarray=np.zeros(shape=(1),dtype=np.float64);#Инициализация текущего значения считанного из
        #pkl файла bias (на тот случай, если такого атрибута нет)
        #Считывание coef и intercept, если они оба есть у модели текущего пайплайна
        #ПОКА ЧТО СЧИТАЕМ, ЧТО ВСЕ scaler - это StandardScaler (если scaler_current - это НЕ StandardScaler, то пропускаем этот пайплайн)
        if (hasattr(model_current,'coef_'))and(hasattr(model_current,'intercept_'))and(isinstance(scaler_current,StandardScaler)==True):
            num_pipelines_have_coef_and_intercept_attributes=num_pipelines_have_coef_and_intercept_attributes+1;
            coefs_current:np.ndarray=model_current.__dict__['coef_'];
            bias_current:np.ndarray=model_current.__dict__['intercept_'];
            print(f'pipeline_id: {pipeline_id}, coefs_current: {coefs_current}, bias_current: {bias_current}');
            print(f'coefs_current: {coefs_current}, bias_current: {bias_current}');
            print(f'type(coefs_current): {type(coefs_current)}, type(bias_current): {type(bias_current)}');
            if len(coefs_current.shape)==1:#Если coefs_current: [6.77108822e-02 8.84446049e-01 3.14625849e+01 0.00000000e+00]
                for i in range(n_features_selected_randomly):coefs_total[i]=coefs_total[i]+float(coefs_current[i]);
            elif len(coefs_current.shape)==2:#Если coefs_current: [[ 0.00489083  0.78579048  0.         -0.23675243  0.79130886]]
                for i in range(n_features_selected_randomly):coefs_total[i]=coefs_total[i]+float(coefs_current[0][i]);
            if type(bias_current)==np.ndarray:bias_total=bias_total+float(bias_current[0]);
            elif type(bias_current)==np.float64:bias_total=bias_total+float(bias_current);
            elif type(bias_current)==float:bias_total=bias_total+bias_current;
            else:print(f'bias_current: {bias_current}, type(bias_current): {type(bias_current)}, значение bias_current НЕ ДОБАВЛЕНО, так как его тип не предусмотрен в коле функции, НЕОБХОДИМО предусмотреть обработку случая с этим типом');
        else:
            print(f"hasattr(model_current,'coef_'): {hasattr(model_current,'coef_')}, hasattr(model_current,'intercept_'): {hasattr(model_current,'intercept_')}, эта модель (модель из этого пайплайна) НЕ добавлена, так как у неё нет хотя бы одного из нужных атрибутов");
        num_pipelines_added=num_pipelines_added+1;
    print(f'Все пайплайны обработаны, из имеющих атрибуты coef_ и intercept_ в своих моделях (таких {num_pipelines_have_coef_and_intercept_attributes} из {pipelines_num} или {(100*num_pipelines_have_coef_and_intercept_attributes/pipelines_num):.4f}%) добавлены коэффициенты и смещение');
    for i in range(n_features_selected_randomly):coefs_total[i]=coefs_total[i]/num_pipelines_have_coef_and_intercept_attributes;
    bias_total=bias_total/num_pipelines_have_coef_and_intercept_attributes;
    print(f'После усреднения по {num_pipelines_have_coef_and_intercept_attributes} пайплайнам (до учёта scaler):');
    print(f'coefs: {coefs_total}, bias: {bias_total}');
    str_list_to_txt:list[str]=[];
    pipeline_ids_str:str=' '.join(pipeline_ids);
    str_list_to_txt.append(f'n_features_selected_randomly: {n_features_selected_randomly}, pipelines_num: {pipelines_num}, pipeline_ids_str: {pipeline_ids_str}\n');
    str_list_to_txt.append(f'{num_pipelines_have_coef_and_intercept_attributes} из {pipelines_num} пайплайнов (или {(100*num_pipelines_have_coef_and_intercept_attributes/pipelines_num):.4f}%) имеют атрибуты coef_ и intercept_, по этим пайплайнам выполняется усреднение\n');
    str_list_to_txt.append(f'После усреднения по {num_pipelines_have_coef_and_intercept_attributes} пайплайнам (до учёта scaler):\n');
    str_list_to_txt.append(f'coefs: {coefs_total}, bias: {bias_total}\n');
    str_list_to_txt.append(f'scaler_mean_list_float: {scaler_mean_list_float}\n');
    str_list_to_txt.append(f'scaler_var_list_float: {scaler_var_list_float}\n');
    str_list_to_txt.append(f'scaler_scale_list_float: {scaler_scale_list_float}\n');
    str_list_to_txt.append(f'scaler_scale_sqr_list_float: {scaler_scale_sqr_list_float}\n');
    
    
    coefs_after_scaler:list[float]=[0.0]*n_features_selected_randomly;
    bias_after_scaler:float=bias_total;
    for i in range(n_features_selected_randomly):
        #Каждый коэффициент делится на соответствующий scale
        coefs_after_scaler[i]=coefs_total[i]/scaler_scale_list_float[i] if scaler_scale_list_float[i]!=0.0 else 0.0;
        #Вычитаем вклад mean из bias
        bias_after_scaler=bias_after_scaler-coefs_after_scaler[i]*scaler_mean_list_float[i];
    str_list_to_txt.append(f'После усреднения по {num_pipelines_have_coef_and_intercept_attributes} пайплайнам (после учёта scaler):\n');
    print(f'coefs: {coefs_after_scaler}, bias: {bias_after_scaler}');
    str_list_to_txt.append(f'coefs: {coefs_after_scaler}, bias: {bias_after_scaler}\n');


    str_list_to_txt.append(f'================================\n');
    with open(file='log_results.txt',mode='at',encoding='UTF-8')as f_log:f_log.writelines(str_list_to_txt);
    pass;

if __name__=='__main__':
    config_loaded:bool=False;config_file_name_no_ext_default:str='conf1';#Загрузка глобальных переменных из файла конфигурации
    while config_loaded==False:
        print(f'Введите название config файла (без .json) или просто Enter для использования файла конфигурации по умолчанию ({config_file_name_no_ext_default}): ',end='');
        config_file_name_no_ext:str=input();
        if len(config_file_name_no_ext)==0:config_file_name_no_ext=config_file_name_no_ext_default;
        config_file_name:str=f'{config_file_name_no_ext}.json';
        #print(f'config_file_name_no_ext: {config_file_name_no_ext}, config_file_name: {config_file_name}');
        conf_dict:dict=load_config(config_file_name=config_file_name);
        if conf_dict!=-1:config_loaded=True;
    if conf_dict['print_config_after_loading']==True:
        print('='*60,'ЗАГРУЖЕННАЯ КОНФИГУРАЦИЯ (json + pprint):','='*60,json.dumps(obj=conf_dict,indent=4,ensure_ascii=False),'='*60 +'\n',sep='\n');
        pprint.pprint(object=conf_dict,indent=1,compact=False,sort_dicts=False,underscore_numbers=False);
    debug_info_depth:int=conf_dict['debug_info_depth'];#Параметр глубины вывода отладочной информации
    #debug_info_depth==0: минимум отладочной информации
    #debug_info_depth==5: максимум отладочной информации
    save_opened_and_closed_features_csvs:bool=conf_dict['save_opened_and_closed_features_csvs'];
    save_opened_parquet:bool=str_or_bool_to_bool(s=conf_dict['save_opened_parquet']);
    score_type:str=conf_dict['score_type'];
    d2_pinball_score_alpha:float=str_to_float(s=conf_dict['score_calculating_params']['d2_pinball_score_alpha'],num_min=-100.0,num_max=100.0,num_default=0.5);
    d2_tweedie_score_power:float=str_to_float(s=conf_dict['score_calculating_params']['d2_tweedie_score_power'],num_min=-100.0,num_max=100.0,num_default=0.0);
    mean_pinball_loss_alpha:float=str_to_float(s=conf_dict['score_calculating_params']['mean_pinball_loss_alpha'],num_min=-100.0,num_max=100.0,num_default=0.5);
    mean_tweedie_deviance_power:float=str_to_float(s=conf_dict['score_calculating_params']['mean_tweedie_deviance_power'],num_min=-100.0,num_max=100.0,num_default=0.0);
    fbeta_score_beta:float=str_to_float(s=conf_dict['score_calculating_params']['fbeta_score_beta'],num_min=1e-12,num_max=1e12,num_default=1.0);
    lnx_eps:float=str_to_float(s=conf_dict['lnx_eps'],num_min=1e-50,num_max=1e-6,num_default=1.0e-15);
    prob_eps:float=str_to_float(s=conf_dict['prob_eps'],num_min=1e-50,num_max=1e-6,num_default=1.0e-15);

    opened_data_all_features,opened_target,opened_ids,closed_data_all_features,closed_target,closed_ids=load_data_from_npy(save_opened_and_closed_features_csvs=save_opened_and_closed_features_csvs,save_opened_parquet=save_opened_parquet);
    create_log_files();

    #Основной цикл программы:
    command_num:int=0;
    while command_num!=-1:
        print(f'=====================================');
        print(f'1 => выполнить создание, кросс-валидацию и проверку на holdout одного пайплайна со случайными гиперпараметрами n раз (функция run_one_pipeline_experiment_v1)');
        print(f'2 => создать json и tsv файлы с предсказанием пайплайна или средним предсказанием нескольких пайплайнов из списка их id');
        print(f'3 => анализ csv лога с результатами пайплайнов (для отбора id лучших пайплайнов)');
        print(f'4 => анализ txt лога с результатами пайплайнов (для отбора индексов наиболее часто случайно выбираемых в лучших пайплайнах признаков)');
        print(f'5 => вывод информации о содержимом одного pkl файла');
        print(f'6 => создать txt файл со значениями coef и bias для пайплайна или усреднёнными значениями нескольких пайплайнов из списка их id');

        print(f'-1 => выйти из программы');
        print(f'=====================================');

        input_str_orig:str=input('Введите номер команды (0 ничего не делает): ');
        input_str_processed:str=input_str_orig;        
        input_str_processed:str=''.join(ch for ch in input_str_processed if ch in '-0123456789');#1. Оставляем только цифры и знак минус
        # 2 и 3. Если есть минус, оставляем от первого минуса и удаляем все остальные минусы
        if '-' in input_str_processed:input_str_processed='-'+input_str_processed[input_str_processed.index('-')+1:].replace('-','');
        if input_str_processed=='':input_str_processed='0';#Если в процессе преобразования из строки удалилось вообще всё, то запишем 0
        print(f'Введено: {input_str_orig}, обработано как: {input_str_processed}');
        command_num=int(input_str_processed);
        if command_num==-1:#-1 => выйти из программы
            pass;
        elif command_num==1:#1 => выполнить кросс-валидацию с проверкой на holdout n раз (фунция run_one_pipeline_experiment_v1)
            num_of_experiments:int=int(input('Введите количество экспериментов: '));
            #num_features_select_from_all_min:int=0;num_features_select_from_all_max:int=0;
            randomly_selected_indexes_str:str=input('Введите список номеров признаков для отбора через ПРОБЕЛ (список для отбора одинакового набора признаков во всех экспериментах или просто Enter для случайного задания списка в каждом эксперименте), пример списка (без квадратных скобок): [218 307 56 266 63 67 77 336 105 376 59 73 257 42]: ');
            #indexes_times_list_dicts (after sorting): [{'index': 218, 'times': 114}, {'index': 307, 'times': 113}, {'index': 56, 'times': 111}, {'index': 266, 'times': 111}, {'index': 63, 'times': 110}, {'index': 67, 'times': 110}, {'index': 77, 'times': 109}, {'index': 336, 'times': 108}, {'index': 84, 'times': 107}, {'index': 105, 'times': 107}, {'index': 376, 'times': 106}, {'index': 59, 'times': 105}, {'index': 73, 'times': 105}, {'index': 257, 'times': 105}, {'index': 42, 'times': 103}, {'index': 362, 'times': 103}, ...
            #14 наиболее часто использованных признаков (в виде списка): [218, 307, 56, 266, 63, 67, 77, 336, 84, 105, 376, 59, 73, 257]
            #Восьмым (считая с нуля) в этом списке является признак номер 84, то есть это признак X84 из файла "all_features_opened_data.csv". Все значения этого признака равны нулю. Вместо него используем следующий признак (следующий за признаком 257), это признак {'index': 42, 'times': 103}. Этот признак подходит, у него все 800 значений у открытых данных уникальны (то есть его дисперсия не равна нулю). Вместо списка [218, 307, 56, 266, 63, 67, 77, 336, 84, 105, 376, 59, 73, 257] испоьзуем список [218, 307, 56, 266, 63, 67, 77, 336, 105, 376, 59, 73, 257, 42].
            if len(randomly_selected_indexes_str)==0:#Если список индексов не задан, то они могут выбираться случайно (если их количество равно 0)
                randomly_selected_indexes:list[int]=None;
                num_features_select_from_all_min:int=conf_dict['pipeline_params']['num_features_select_from_all_min'];
                num_features_select_from_all_max:int=conf_dict['pipeline_params']['num_features_select_from_all_max'];
            else:randomly_selected_indexes:list[int]=[int(num)for num in randomly_selected_indexes_str.split(sep=' ')];
            if conf_dict['problem_type']=='regression':
                prefered_fs_estimator_types:list[str]=conf_dict['pipeline_params']['prefered_fs_estimator_types_regression'];
                prefered_model_types:list[str]=conf_dict['pipeline_params']['prefered_model_types_regression'];
            elif conf_dict['problem_type']in['classification_binary','classification_multiclass']:
                prefered_fs_estimator_types:list[str]=conf_dict['pipeline_params']['prefered_fs_estimator_types_classification'];
                prefered_model_types:list[str]=conf_dict['pipeline_params']['prefered_model_types_classification'];
            
            for i in range(num_of_experiments):
                try:
                    print(f'Эксперимент {i+1}/{num_of_experiments}... ',end='');
                    pipeline_id:str=run_one_pipeline_experiment_v1(num_features_select_from_all_min=conf_dict['pipeline_params']['num_features_select_from_all_min'],num_features_select_from_all_max=conf_dict['pipeline_params']['num_features_select_from_all_max'],randomly_selected_indexes=randomly_selected_indexes,problem_type=conf_dict['problem_type'],task_output=conf_dict['task_output'],score_type=conf_dict['score_type'],non_negative_y_guarantee=conf_dict['pipeline_params']['non_negative_y_guarantee'],
                    use_imputer_probability=conf_dict['pipeline_params']['use_imputer_probability'],use_var_thresholder_probability=conf_dict['pipeline_params']['use_var_thresholder_probability'],use_scaler_probability=conf_dict['pipeline_params']['use_scaler_probability'],prefered_scaler_types=conf_dict['pipeline_params']['prefered_scaler_types'],use_feature_selector_probability=conf_dict['pipeline_params']['use_feature_selector_probability'],prefered_feature_selector_types=conf_dict['pipeline_params']['prefered_feature_selector_types'],prefered_fs_estimator_types=prefered_fs_estimator_types,scaler_type=None,scaler_hyperparams=None,model_type=None,prefered_model_types=prefered_model_types,model_hyperparams=None,num_folds=conf_dict['n_cross_valid_folds'],score_valid_min_threshold=conf_dict['score_valid_min_threshold'],score_valid_max_threshold=conf_dict['score_valid_max_threshold'],use_only_linear_models=conf_dict['pipeline_params']['use_only_linear_models'],use_only_models_with_predict_proba=conf_dict['pipeline_params']['use_only_models_with_predict_proba'],n_cpu_cores=conf_dict['pipeline_params']['n_cpu_cores']);
                except Exception as ex:
                    print(f'Возникло исключение ex, type(ex): {type(ex)}, ex: {ex}');
        elif command_num==2:#2 => создать json и tsv файлы с предсказанием пайплайна или средним предсказанием нескольких пайплайнов из списка их id
            pipeline_ids_str:str=input('Введите id пайплайна или нескольких пайплайнов через запятую или пробел (например, [08JZRAWXBE5N43MX] или [08JZRAWXBE5N43MX,2352C29OXLDYGPAL,J0KZOWU71FHE3TCR,EENT8VMHI4CK4D24]) (БЕЗ КВАДРАТНЫХ СКОБОК): ');
            #digits_round_min,digits_round_max=[int(num_s)for num_s in input('Введите минимальное и максимальное количество цифр округления (например, 2 18): ').split(sep=' ')]
            if ', 'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=', ');
            elif ','in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=',');
            elif ' 'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=' ');
            elif '/'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep='/');
            elif '|'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep='|');#Если problem_type=='regression', то n_classes ни на что не влияет
            else:pipeline_ids_list:list[str]=[pipeline_ids_str];#Если нет ни одного разделителя, значит список id пайплайнов состоит из одного id, поэтому разделителей нет
            print_all_predictions:bool=str_or_bool_to_bool(s=conf_dict['creating_prediction_files_params']['print_all_predictions']);
            calculate_score_for_closed_data:bool=str_or_bool_to_bool(s=conf_dict['creating_prediction_files_params']['calculate_score_for_closed_data']);#Если True, то вычисляем значение метрии на closed_data (для этого должен быть файл closed_target.npy)
            create_predictions_files(pipeline_ids=pipeline_ids_list,digits_round_min=conf_dict['creating_prediction_files_params']['digits_round_min'],digits_round_max=conf_dict['creating_prediction_files_params']['digits_round_max'],problem_type=conf_dict['problem_type'],n_classes=conf_dict['n_classes'],print_all_predictions=print_all_predictions,calculate_score_for_closed_data=calculate_score_for_closed_data,score_type=score_type);
        elif command_num==3:#анализ csv лога с результатами пайплайнов
            score_valid_mean_threshold_min:float=conf_dict['csv_log_analize_params']['score_valid_mean_threshold_min'];
            score_valid_mean_threshold_max:float=conf_dict['csv_log_analize_params']['score_valid_mean_threshold_max'];
            score_test_threshold_min:float=conf_dict['csv_log_analize_params']['score_test_threshold_min'];
            score_test_threshold_max:float=conf_dict['csv_log_analize_params']['score_test_threshold_max'];
            n_features_selected_randomly_threshold_min:int=conf_dict['csv_log_analize_params']['n_features_selected_randomly_threshold_min'];
            n_features_selected_randomly_threshold_max:int=conf_dict['csv_log_analize_params']['n_features_selected_randomly_threshold_max'];
            pipeline_file_size_theshold_min:int=conf_dict['csv_log_analize_params']['pipeline_file_size_theshold_min'];
            pipeline_file_size_theshold_max:int=conf_dict['csv_log_analize_params']['pipeline_file_size_theshold_max'];
            log_pipelines_csv_file_name:str=conf_dict['csv_log_analize_params']['log_pipelines_csv_file_name'];
            analize_log_pipelines_csv(log_pipelines_csv_file_name=log_pipelines_csv_file_name,score_valid_mean_threshold_min=score_valid_mean_threshold_min,score_valid_mean_threshold_max=score_valid_mean_threshold_max,score_test_threshold_min=score_test_threshold_min,score_test_threshold_max=score_test_threshold_max,n_features_selected_randomly_threshold_min=n_features_selected_randomly_threshold_min,n_features_selected_randomly_threshold_max=n_features_selected_randomly_threshold_max,pipeline_file_size_theshold_min=pipeline_file_size_theshold_min,pipeline_file_size_theshold_max=pipeline_file_size_theshold_max);
            pass;
        elif command_num==4:#анализ txt лога с результатами пайплайнов
            log_pipelines_txt_file_name:str=conf_dict['txt_log_analize_params']['log_pipelines_txt_file_name'];
            analize_log_pipelines_txt(log_pipelines_txt_file_name=log_pipelines_txt_file_name);

            pass;
        elif command_num==5:
            pkl_file_name:str=input('Введите название pkl файла (например, pipeline_RM3W9PGWI65QRNXI.pkl) или просто id пайплайна (например, RM3W9PGWI65QRNXI): ');
            if 'pipeline_'not in pkl_file_name:pkl_file_name=f'pipeline_{pkl_file_name}.pkl';
            analize_one_pkl_file(pkl_file_name=pkl_file_name);

            pass;
        elif command_num==6:#6 => создать txt файл со значениями coef и bias для модели или усреднёнными значениями нескольких пайплайнов из списка их id
            pipeline_ids_str:str=input('Введите id пайплайна или нескольких пайплайнов через запятую или пробел (например, [93P121PG8FACD4L2] или [93P121PG8FACD4L2 0UVA9BKRDG2E7EX6 0271ZHM0Z5R3HXLS 04HVI5VS4FKMIVBG 05DZJE2OQWUXM93R 06SJSYUV7IEKGEXX]) (БЕЗ КВАДРАТНЫХ СКОБОК): ');
            digits_round_min,digits_round_max=[int(num_s)for num_s in input('Введите минимальное и максимальное количество цифр округления (например, 2 18): ').split(sep=' ')]
            if ','in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=',');
            elif ' 'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep=' ');
            elif '/'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep='/');
            elif '|'in pipeline_ids_str:pipeline_ids_list:list[str]=pipeline_ids_str.split(sep='|');
            create_coefs_and_bias_files(pipeline_ids=pipeline_ids_list,digits_round_min=digits_round_min,digits_round_max=digits_round_max);
        else:
            print(f'Введён номер несуществующей команды, необходимо ввести другой номер');

    print(f'Работа программы завершена');



