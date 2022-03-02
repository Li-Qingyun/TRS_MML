"""
Date: 2022.02.28
Author: Qingyun Li
Refer to: Ruo chen
"""

from qcloudsms_py import SmsMultiSender, SmsSingleSender
from qcloudsms_py.httpclient import HTTPError
from django.conf import settings


class Message:
    def __init__(self, trainer):
        self.trainer = trainer
        self.message_id = '1400508980'
        self.message_key = '2b11dd94c69c8e6b8dc419772fea5440'
        self.sms_sign = 'AI训练日记'  # 自己腾讯云创建签名时填写的签名内容（使用公众号的话这个值一般是公众号全称或简称）
        self.phone_number

    def send_sms_single(phone_num, template_id, template_param_list):
        """
        单条发送短信
        :param phone_num: 手机号
        :param template_id: 腾讯云短信模板ID
        :param template_param_list: 短信模板所需参数列表，例如:【验证码：{1}，描述：{2}】，则传递参数 [888,666]按顺序去格式化模板
        :return:
        """
        sender = SmsSingleSender(appid, appkey)
        try:
            response = sender.send_with_param(86, phone_num, template_id, template_param_list, sign=sms_sign)
        except HTTPError as e:
            response = {'result': 1000, 'errmsg': "网络异常发送 失败"}
        return response

    def send_sms_multi(phone_num_list, template_id, param_list):
        """
        批量发送短信
        :param phone_num_list:手机号列表
        :param template_id:腾讯云短信模板ID
        :param param_list:短信模板所需参数列表，例如:【验证码：{1}，描述：{2}】，则传递参数 [888,666]按顺序去格式化模板
        :return:
        """
        sender = SmsMultiSender(appid, appkey)
        try:
            response = sender.send_with_param(86, phone_num_list, template_id, param_list, sign=sms_sign)
        except HTTPError as e:
            response = {'result': 1000, 'errmsg': "网络异常发送失败"}
        return response

if __name__ == '__main__':
    send_sms_single('18545525156', '928741', [' LQY_RTX ', ' 发短信程序 '])