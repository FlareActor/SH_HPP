# coding: utf-8
import os
import re
from datetime import datetime
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 新房数据表路径
newdisk_path = '../../数据/AD_NewDisk.csv'
# 房源属性数据表路径
property_path = '../../数据/AD_Property.csv'
# 地址数据表路径
address_path = '../../数据/AD_NewDiskAddress.csv'
# 挂牌数据路径
data_path = '/Users/wangdexun/Documents/gubbins/房价大数据/数据/挂牌数据/'
# 测试数据路径
test_path = '../../数据/评估师数据/2017全年.xlsx'
# 结果保存路径
result_path = './result/17年11月.csv'
# 训练集起始月份
beginDate = '2016-12'
# 训练集终止月份
endDate = '2017-11'


def load_guapai(name, month):
    """读取挂牌数据"""
    with open(os.path.join(data_path, name, '挂牌.txt')) as f:
        l = []
        for i in f.readlines():
            l.append(i.split('\t'))
        df = pd.DataFrame(l)
        drop_col = [0, 15, 16, 18, 19, 20, 21]
        if len(df.columns) == 23:
            drop_col.append(22)
        df.drop(drop_col, axis=1, inplace=True)  # 去除无用列
        df.columns = ['area', 'address', 'name', 'price', 'unit_price', 'acreage', 'room_type', 'all_floor', 'floor',
                      'shore', 'house_type', 'fitment', 'time', 'Source', 'house_trait']
        df['month'] = month
        return df


def load_data():
    """加载训练数据"""
    print('加载挂牌训练数据...')
    cache_path = './cache/guapai_%s-%s.hdf' % (beginDate, endDate)
    if os.path.exists(cache_path):
        # 加载缓存
        meta_df = pd.read_hdf(cache_path, 'meta')
        all_df = pd.read_hdf(cache_path, 'data')
    else:
        pool = Pool()
        # files=[i for i in os.listdir(data_path) if '.' not in i]
        files = np.unique([datetime.strftime(x, '%Y-%m') for x in list(pd.date_range(start=beginDate, end=endDate))])
        files = sorted(files)
        dfs = [pool.apply_async(load_guapai, (name, month)) for month, name in enumerate(files)]
        pool.close()
        pool.join()
        dfs = [i.get() for i in dfs]
        print('共加载%s个月份的挂牌数据...' % len(dfs))
        all_df = pd.concat(dfs, ignore_index=True)

        # 获取经纬度信息
        newdisk_df = pd.read_csv(newdisk_path, usecols=['NewDiskID', 'PropertyID', 'NewDiskName', 'Coordinates'])
        newdisk_df.rename(columns={'NewDiskName': 'name'}, inplace=True)

        # 获取板块、环线信息
        property_df = pd.read_csv(property_path, usecols=['PropertyID', 'Area', 'Plate', 'Module', 'HousingName'])
        property_df.rename(columns={'Area': 'area', 'HousingName': 'name'}, inplace=True)

        # 获取楼盘地址信息
        address_df = pd.read_csv(address_path, usecols=['RoadLaneNo', 'NewDiskID'])
        address_df.rename(columns={'RoadLaneNo': 'address'}, inplace=True)

        # merge them
        meta_df = pd.merge(newdisk_df, property_df.drop('name', axis=1), on='PropertyID', how='left')
        # meta_df=pd.merge(meta_df,address_df,on='NewDiskID',how='left')

        # 小区名称清洗
        index = meta_df.name.notnull()
        meta_df.loc[index, 'name'] = meta_df.loc[index, 'name'].apply(name_filter)
        all_df.name = all_df.name.apply(name_filter)

        address_df.address = address_df.address.apply(address_filter)
        all_df.address = all_df.address.apply(address_filter)

        # 转换数值类型 str->float
        numerical_columns = ['price', 'unit_price', 'acreage', 'all_floor', 'floor']
        all_df[numerical_columns] = all_df[numerical_columns].astype(float)

        all_df['No'] = range(all_df.shape[0])
        address_match = pd.merge(all_df[['No', 'address']], address_df, on='address', how='inner')
        name_match = pd.merge(all_df[['No', 'name', 'area']], meta_df[['name', 'area', 'NewDiskID']],
                              on=['name', 'area'],
                              how='inner')

        match = pd.concat((address_match[['No', 'NewDiskID']], name_match[['No', 'NewDiskID']]), ignore_index=True)
        match.drop_duplicates(keep='first', inplace=True)
        match = match.sort_values('No')

        all_df = all_df.loc[match.No]
        all_df['NewDiskID'] = match.NewDiskID.values
        all_df.drop('No', axis=1, inplace=True)

        all_df = pd.merge(all_df, meta_df[['NewDiskID', 'Coordinates', 'Plate', 'Module']], on='NewDiskID', how='left')
        meta_df.to_hdf(cache_path, 'meta')
        all_df.to_hdf(cache_path, 'data')
    return meta_df, all_df


def name_filter(name):
    """小区名正则过滤"""
    n = re.compile('\(|\（|一期|二期').split(name)[0]
    n = re.sub(r'\(.*?\)', '', re.sub(r'\（.*?\）', '', n))
    n = n.strip('*0123456789(（）)')
    n = n.split('第')[0]
    return n


def address_filter(address):
    """小区地址清洗"""
    n = re.compile(',|，|、').split(address)[0]
    n = re.sub(r'\(.*?\)', '', re.sub(r'\（.*?\）', '', n))
    n = n.strip('*0123456789')
    return n


def time_map(time):
    if type(time) == str:
        split_char = '/' if '/' in time else '-'
        return int(time.split(split_char)[0])
    return None


def make_coordinates(data):
    coors = []
    for i in tqdm(data):
        if type(i) == str and i != '公寓' and i != '商业' and i != '其它':
            coors.append(i.split(','))
        else:
            coors.append([None, None])
    coors = pd.DataFrame(coors, columns=['loc_x', 'loc_y'])
    # coors=pd.DataFrame([coor.split(',') for coor in all_df.Coordinates],columns=['loc_x','loc_y'],index=all_df.index)
    coors = coors.astype(float)
    return coors


# 楼层映射
floor_map = lambda x: list(pd.cut(x, [0, 3, 6, 9, np.inf], labels=['低层', '多层', '小高层', '高层']))


def preprocess(all_df):
    """特征预处理"""
    print('清洗挂牌数据...')
    cache_path = './cache/feats_%s-%s.hdf' % (beginDate, endDate)
    if os.path.exists(cache_path):
        all_df = pd.read_hdf(cache_path, 'data')
    else:
        # 修正面积
        acreage_log = np.log(all_df.acreage)
        mean = acreage_log.mean()
        std = acreage_log.std()
        i = acreage_log[(acreage_log <= mean + 2 * std) & (acreage_log >= mean - 1 * std)].index
        sns.set({'figure.figsize': (8, 4)})
        sns.boxplot(all_df.loc[i].acreage)
        all_df.loc[i].acreage.describe()
        all_df = all_df.loc[i]

        # 修正单价
        unit_price_log = np.log1p(all_df.unit_price)
        mean = unit_price_log.mean()
        std = unit_price_log.std()
        i = unit_price_log[(unit_price_log <= mean + 3 * std) & (unit_price_log >= mean - 3.2 * std)].index
        sns.set({'figure.figsize': (8, 4)})
        sns.boxplot(all_df.loc[i].unit_price)
        all_df.loc[i].unit_price.describe()
        all_df = all_df.loc[i]

        # 修复总价
        # 修复总价单位误差
        all_df.loc[all_df.price <= 10000, 'price'] *= 10000
        # 差价分布
        anomaly_price = np.abs(all_df.unit_price * all_df.acreage - all_df.price)
        anomaly_price_index = anomaly_price[anomaly_price > 100000].index  # 差价太多为异常点
        # 直接删除异常样本
        all_df.drop(anomaly_price_index, axis=0, inplace=True)

        # 环线
        all_df.loc[all_df[all_df.Module == '所有'].index, 'Module'] = '内环内'
        # sorted_module = all_df[['unit_price', 'Module']].groupby('Module').median().sort_values('unit_price')
        # i = pd.Series(range(0, sorted_module.shape[0]), index=sorted_module.index)
        # all_df.Module = all_df.Module.map(i.to_dict())

        # 楼层映射
        all_df.loc[all_df.floor < 0, 'floor'] = np.nan
        # 分段映射
        all_df['floor_section'] = floor_map(all_df.floor)

        # 朝向因素
        # 暂无为缺省字段
        all_df.shore.replace({'暂无数据': '暂无', '      ': '暂无', '': '暂无'}, inplace=True)
        sorted_shore = all_df[['unit_price', 'shore']].groupby('shore').mean().sort_values('unit_price')
        i = pd.Series(range(0, sorted_shore.shape[0]), index=sorted_shore.index)
        all_df.shore = all_df.shore.map(i.to_dict())

        # 房屋类型
        all_df.loc[all_df[(all_df.house_type == '其它') | (all_df.house_type == '工厂')].index, 'house_type'] = '公寓'
        sorted_house_type = all_df[['house_type', 'unit_price']].groupby('house_type').median().sort_values(
            'unit_price')
        i = pd.Series(range(0, sorted_house_type.shape[0]), index=sorted_house_type.index)
        i.to_dict()
        all_df.house_type = all_df.house_type.map(i.to_dict())

        # 装修情况
        default_fit = '暂无'  # 缺省字段填充
        all_df.fitment.replace({'': default_fit, '暂无数据': default_fit, '豪华装': '豪装', '其他': default_fit}, inplace=True)
        all_df.fitment = all_df.fitment.apply(lambda x: x.strip('修'))
        sorted_fitment = all_df[['fitment', 'unit_price']].groupby('fitment').median().sort_values('unit_price')
        i = pd.Series(range(0, sorted_fitment.shape[0]), index=sorted_fitment.index)
        all_df.fitment = all_df.fitment.map(i.to_dict())

        # 房型
        r = re.compile('室|厅|厨|卫')  # 正则提取房型数据
        l = [map(int, r.split(i)[:-1]) for i in all_df.room_type]
        room_type_df = pd.DataFrame(l, index=all_df.index, columns=['室', '厅', '厨', '卫'])
        all_df = pd.concat((all_df, room_type_df), axis=1)

        # 时间
        all_df.time = all_df.time.apply(lambda x: time_map(x)).astype(int)
        all_df.time = all_df.time.apply(lambda x: min(2018 - x, 100) if 0 < x <= 2018 else None)

        # 经纬度
        coors = make_coordinates(all_df.Coordinates.values)
        all_df.index = coors.index
        all_df = pd.concat((all_df, coors), axis=1).drop('Coordinates', axis=1)

        # 缓存特征矩阵
        all_df = all_df[all_df.unit_price.notnull()]
        all_df.to_hdf(cache_path, 'data')
    print('共有%d条训练数据' % all_df.shape[0])
    return all_df


def make_train_test_set(all_df, test_df=None, meta_df=None):
    '''计算楼盘基价'''
    # x_train = all_df[['area', 'Plate', 'Module', 'time', 'loc_x', 'loc_y', 'unit_price', 'NewDiskID']]
    #
    # property_df = pd.read_csv(property_path, usecols=['PropertyID', 'Area', 'Plate', 'Module'])
    # newdisk_df = pd.read_csv(newdisk_path,
    #                          usecols=['NewDiskID', 'SubmittedDate', 'PropertyID', 'Coordinates', 'NewDiskName',
    #                                   'NewDiskType'])
    # newdisk_df.SubmittedDate = newdisk_df.SubmittedDate.apply(time_map)
    # newdisk_df['time'] = newdisk_df.SubmittedDate.apply(lambda x: min(2018 - x, 100) if 0 < x <= 2018 else None)
    # x_test = pd.merge(newdisk_df, property_df, on=['PropertyID'], how='left')
    # coors = make_coordinates(x_test.Coordinates.values)
    # x_test.drop(['PropertyID', 'Coordinates', 'SubmittedDate', 'NewDiskType'], axis=1, inplace=True)
    # x_test.index = coors.index
    # x_test = pd.concat((x_test, coors), axis=1)
    # x_test.rename(columns={'Area': 'area', 'NewDiskName': 'name'}, inplace=True)
    #
    # y_train = x_train.unit_price
    # mid_price = x_train[['NewDiskID', 'unit_price']].groupby('NewDiskID', as_index=False)['unit_price'].agg(
    #     {'mid': 'median'})
    # x_train = pd.merge(x_train, mid_price, on='NewDiskID', how='left')
    # x_test = pd.merge(x_test, mid_price, on='NewDiskID', how='left')
    #
    # x_train.loc[x_train[x_train.Module == '所有'].index, 'Module'] = '内环内'
    # x_test.loc[x_test[x_test.Module == '所有'].index, 'Module'] = '内环内'
    # sorted_module = x_train[['unit_price', 'Module']].groupby('Module').mean().sort_values('unit_price')
    # i = pd.Series(range(0, sorted_module.shape[0]), index=sorted_module.index)
    # x_train.Module = x_train.Module.map(i.to_dict())
    # x_test.Module = x_test.Module.map(i.to_dict())
    #
    # # 将离散型变量转换成整型，用于lgb训练
    # encoder = LabelEncoder()  # 小板块
    # x_train.area = encoder.fit_transform(x_train.area)
    # x_test.area = encoder.transform(x_test.area)
    # x_train.Plate = encoder.fit_transform(x_train.Plate)
    # x_test.Plate = encoder.fit_transform(x_test.Plate)
    # x_train.drop(['NewDiskID', 'unit_price'], axis=1, inplace=True)
    # return x_train, y_train, x_test, None

    '''计算单套价格'''
    x_train = all_df[
        ['acreage', 'all_floor', 'floor', 'time', 'NewDiskID',
         'area', 'Plate', 'Module', 'floor_section', 'loc_x', 'loc_y']]
    y_train = all_df.unit_price
    # 计算小区房价中位数
    med_price = pd.concat((x_train.NewDiskID, y_train), axis=1)
    med_price = med_price.groupby('NewDiskID', as_index=False)['unit_price'].agg({'median': 'mean'})
    x_train = pd.merge(x_train, med_price, on='NewDiskID', how='left')
    # 计算经纬度geohash信息
    #     geo=[]
    #     for i in tqdm(x_train[['loc_y','loc_x']].values):
    #         geo.append(encode(i[0],i[1],6))
    #     x_train['geo']=[i[:-1] for i in geo]
    # 测试集
    have_unit_price = 'unit_price' in test_df.columns
    use_cols = ['index', 'time', 'all_floor', 'floor', 'acreage', 'NewDiskID']
    if have_unit_price:
        use_cols.append('unit_price')
    x_test = test_df[use_cols]
    x_test = pd.merge(x_test, meta_df.drop(['PropertyID'], axis=1), on='NewDiskID', how='inner')
    x_test = pd.merge(x_test, med_price, on='NewDiskID', how='left')
    x_test['floor_section'] = floor_map(x_test.floor)
    x_test.time = x_test.time.apply(lambda x: x.year if type(x) == datetime else x)
    x_test['time'] = x_test.time.apply(lambda x: min(2018 - x, 100) if 0 < x <= 2018 else None)

    # 将离散型变量转换成整型，用于lgb训练
    area_le = LabelEncoder()  # 大版块
    x_train.area = area_le.fit_transform(x_train.area)
    x_test.area = area_le.transform(x_test.area)
    plate_le = LabelEncoder()  # 小板块
    x_train.Plate = plate_le.fit_transform(x_train.Plate)
    x_test.Plate = plate_le.transform(x_test.Plate)
    # 环线
    sorted_module = all_df[['unit_price', 'Module']].groupby('Module').median().sort_values('unit_price')
    i = pd.Series(range(0, sorted_module.shape[0]), index=sorted_module.index).to_dict()
    x_train.Module = x_train.Module.map(i)
    x_test.Module = x_test.Module.map(i)

    # 经纬度
    coors = make_coordinates(x_test.Coordinates.values)
    x_test.index = x_test.index
    x_test = pd.concat((x_test, coors), axis=1).drop('Coordinates', axis=1)

    section_map = {'低层': 0, '小高层': 1, '多层': 2, '高层': 3}
    x_train.floor_section = x_train.floor_section.replace(section_map)
    x_test.floor_section = x_test.floor_section.replace(section_map)

    x_train.drop('NewDiskID', inplace=True, axis=1)
    x_test.drop('NewDiskID', inplace=True, axis=1)
    y_test = None
    if have_unit_price:
        y_test = x_test.unit_price
        x_test.drop('unit_price', axis=1, inplace=True)
    return x_train, y_train, x_test, y_test


def train_model(x_train, y_train):
    # * LightGBM
    cache_path = './cache/model_%s-%s_%s.txt' % (beginDate, endDate, x_train.shape[1])
    if os.path.exists(cache_path):
        gbm = lgb.Booster(model_file=cache_path)
    else:
        print('开始模型训练...')
        # 设置模型参数
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.2,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6,
            'max_depth': 14,
            'num_leaves': 220,
            'bagging_freq': 5,
            'min_data_in_leaf': 5,
            'min_gain_to_split': 0,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'verbose': 0,
        }

        categorical_feature = ['area', 'Plate', 'Module']
        if 'floor_section' in x_train.columns:
            categorical_feature.append('floor_section')
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_feature)
        gbm = lgb.train(params, lgb_train, num_boost_round=750)
        gbm.save_model(cache_path)
    return gbm


if __name__ == '__main__':
    '''加载数据'''
    meta_df, all_df = load_data()
    '''处理训练集'''
    all_df = preprocess(all_df)
    '''加载训练测试数据'''
    test_df = pd.read_excel(test_path, sheetname='2017年11月')
    dump_cols = list(test_df.columns)
    test_df.reset_index(inplace=True)
    print('测试数据量:%d' % test_df.shape[0])
    test_df.rename(columns={'楼盘名称': 'name',
                            '房屋类型': 'house_type',
                            '竣工年份': 'time',
                            '总楼层数': 'all_floor',
                            '所属楼层': 'floor',
                            '所属层': 'floor',
                            '建筑面积(m2)': 'acreage',
                            '楼盘编号': 'NewDiskID',
                            '项目名称': 'address',
                            '正常价格(元)': 'unit_price',
                            '总价': 'total_price'}, inplace=True)
    x_train, y_train, x_test, y_test = make_train_test_set(all_df, test_df, meta_df)
    '''分割训练/验证集'''
    # todo:
    # x1, x2, y1, y2 = train_test_split(x_train, y_train, train_size=0.9, random_state=0)
    '''模型训练'''
    gbm = train_model(x_train, y_train)
    print('开始测试')

    '''基价测试'''
    # preds = gbm.predict(x_test[x_train.columns])
    # result = pd.DataFrame({'NewDiskID': x_test.NewDiskID,
    #                        '楼盘': x_test.name,
    #                        '单价': preds,
    #                        'Mark': x_test.mid.notnull() * 1})
    # result = result[['NewDiskID', '楼盘', '单价', 'Mark']]
    # result.to_csv('基价.csv', index=False)

    '''单套测试'''
    y_pred_1 = gbm.predict(x_test[x_train.columns])
    x_train.drop('median', axis=1, inplace=True)
    gbm = train_model(x_train, y_train)
    y_pred_2 = gbm.predict(x_test[x_train.columns])

    prices = pd.DataFrame({'index': x_test['index'], 'price1': y_pred_1, 'price2': y_pred_2})
    ans_df = prices.groupby('index', as_index=False)[['price1', 'price2']].median()
    ans_df = pd.merge(test_df[['index', 'NewDiskID']], ans_df, on='index', how='left').drop('index', axis=1)
    scaler = 1.042
    price = []
    train_ids = all_df.NewDiskID.unique()
    meta_ids = meta_df.NewDiskID.unique()
    for row in ans_df.values:
        if row[0] in train_ids:
            price.append(row[1])
        elif row[0] in meta_ids:
            price.append(row[2])
        else:
            price.append(np.nan)
    price = [i * scaler for i in price]
    ans_df = test_df.drop('index', axis=1)
    ans_df['price'] = price

    # 评估
    gap_ratio = np.abs(ans_df.price - ans_df.unit_price) / ans_df.unit_price * 100
    gap_ratio.dropna(inplace=True)
    print('覆盖率：', gap_ratio.shape[0] / test_df.shape[0])
    counts = []
    for i in range(10, 21):
        counts.append(['%d%%' % i, (gap_ratio <= i).sum() / gap_ratio.shape[0]])
    print(pd.DataFrame(counts, columns=['误差', '百分比']))

    # 保存结果
    ans_df.columns = dump_cols + ['price']
    ans_df.to_csv(result_path, index=False)
