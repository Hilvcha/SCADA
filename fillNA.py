import os
import logging
import time
import pandas as pd
from algorithm import xgboosting
from common.utility import *

MODEL_PATH=os.path.join(os.path.dirname(__file__),'./model/')
LOG_PATH=os.path.join(os.path.dirname(__file__),'./log/')
BASE_PATH = os.path.abspath(os.path.join(__file__, '..'))
DATA_PATH = os.path.join(BASE_PATH, "./dataset/")
SUBMIT_PATH = os.path.join(BASE_PATH, "./submit/")
SUBMITDATA_PATH = os.path.join(BASE_PATH, "./submitset/")
TEMPLATE_PATH=os.path.join(BASE_PATH,'template_submit_result.csv')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d]: %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M');


class fillNa(object):

    def __init__(self):
        self.supervised_obj=xgboosting.XGBoosting()
    def value_predict(self, data):
        """
        Predict if the latest value is an outlier or not.

        :param data: The attributes are:
                    'window', the length of window,
                    'taskId', the id of detect model,
                    'dataC', a piece of data to learn,
                    'dataB', a piece of data to learn,
                    'dataA', a piece of data to learn and the latest value to be detected.
        :type data: Dictionary-like object
        :return: The attributes are:
                 DataFrame
        """
        if "taskId" in data and data["taskId"]:
            model_name = MODEL_PATH + data["taskId"] + "_model"
        else:
            model_name = MODEL_PATH + "xgb_default_model"
        combined_data = data["dataC"] + "," + data["dataB"] + "," + data["dataA"]
        time_series = map(int, combined_data.split(','))
        if "window" in data:
            window = data["window"]
        else:
            window = DEFAULT_WINDOW

        xgb_result = self.supervised_obj.predict(time_series, window, model_name)
        res_value = xgb_result[0]
        prob = xgb_result[1]

        return {res_value,prob}


class operateData(object):
    '''
    数据读取与合并
    '''
    def __init__(self):
        self.submitdata=pd.read_csv(TEMPLATE_PATH)[['ts','wtid']]
    def readData(self,data_id=1):
        """
        根据号码读取日志
        :param data_id: int 日志号
        :return:  dataframe df格式的日志文件
        """
        logging.info('读取'+str(data_id).zfill(3)+'号数据')
        data_path=os.path.join(DATA_PATH,str(data_id).zfill(3),'201807.csv')
        data=pd.read_csv(data_path)
        return data

    def saveSubmitData(self,data,submit_id=1):
        '''
        根据号码保存相应提交块
        :param data:
        :param submit_id:
        :return:
        '''
        logging.info('保存'+str(submit_id).zfill(3)+'号提交数据')
        submitdata_path=os.path.join(SUBMITDATA_PATH,str(submit_id).zfill(3)+'submit.csv')
        data.to_csv(submitdata_path)

    def combineSubmitData(self,):
        """
        提交数据合并持久化到提交目录
        :param data:
        """
        logging.info('保存提交数据')
        data=pd.DataFrame(columns=['ts','wtid','var001','var002','var003','var004','var005','var006','var007','var008','var009','var010','var011','var012','var013','var014','var015','var016','var017','var018','var019','var020','var021','var022','var023','var024','var025','var026','var027','var028','var029','var030','var031','var032','var033','var034','var035','var036','var037','var038','var039','var040','var041','var042','var043','var044','var045','var046','var047','var048','var049','var050','var051','var052','var053','var054','var055','var056','var057','var058','var059','var060','var061','var062','var063','var064','var065','var066','var067','var068'])
        for submit_id in range(1,34):
            submitdata_path = os.path.join(SUBMITDATA_PATH, str(submit_id).zfill(3) + 'submit.csv')
            df=pd.read_csv(submitdata_path,index_col=[0])
            data=data.append(df)
        # 转换枚举类型
        enumVar = ['var016', 'var020', 'var047', 'var053', 'var066']
        data[enumVar] = data[enumVar].apply(np.around).astype(int)

        return data



if __name__ == '__main__':


    fill_obj= fillNa()
    op=operateData()

    for data_id in range(1,34):
        df=op.readData(data_id)
        df=pd.merge(op.submitdata[op.submitdata['wtid']==data_id],df,how='outer',on=['ts','wtid'])
        df.sort_values(by=['ts'],inplace=True)
        df.interpolate(method='spline',order=2,inplace=True)
        df=pd.merge(op.submitdata[op.submitdata['wtid']==data_id],df,how='left',on=['ts','wtid'])

        op.saveSubmitData(data=df,submit_id=data_id)

    data=op.combineSubmitData()

    data.to_csv(os.path.join(SUBMIT_PATH, 'submit_{}.csv'.format(time.strftime("%Y-%m-%d_%H-%M", time.localtime()))),
                index=False,
                header=True,encoding = "utf-8")
    # import matplotlib.pyplot as plt
    #
    # df.plot(subplots=True, sharex=True, figsize=(10, 10))
    # plt.show()
