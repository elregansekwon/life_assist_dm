import numpy as np
from types import SimpleNamespace as SN

import rclpy
from rclpy.node import Node

def make_cfg():
    cfg = SN()

    cfg.dm = SN()
    cfg.dm.service_list = []
    cfg.dm.gpt_model = ''
    cfg.dm.user_config = ''
    cfg.dm.stt = False
    cfg.dm.stt_ready = False
    cfg.dm.tts = False
    cfg.dm.call = False
    cfg.dm.srv_type = ''

    cfg.msg = SN()
    cfg.msg.call = '네. 말씀하세요.'
    cfg.msg.answer = '네, 알겠습니다.'


    cfg.user = SN()
    cfg.user.call = False
    cfg.user.config = ''
    cfg.user.name = ''
    cfg.user.command = ''

    cfg.rb = SN()
    cfg.rb.state = ''
    cfg.rb.location = ''
    cfg.rb.command = ''

    cfg.sqlite_path = '~/ros_ws/dm_ws/src/life_assist_dm/life_assist_dm/life_assist_dm/memory_db.sqlite'

    return cfg

class DialogManagerHeader:
    def __init__(self, node: Node):
        self.node = node

        self.cfg = None
        self.make_cfg()

        self.set_param()


    def make_cfg(self):
        self.cfg = make_cfg()

    def set_param(self):
        self.cfg.dm.service_list = self.node.declare_parameter('service_list', ['memory', 'emotion', 'physical']).get_parameter_value().string_array_value
        self.cfg.dm.gpt_model = self.node.declare_parameter('gpt_model', 'gpt-4o-mini-2024-07-18').get_parameter_value().string_value
        self.cfg.dm.preset_user_name = self.node.declare_parameter('preset_user_name', '권서연').get_parameter_value().string_value

        self.load_user_csv()

    def load_user_csv(self):
        pass

