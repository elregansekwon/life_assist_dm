import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from life_assist_dm.life_assist_dm.llm.whisper_utils import WhisperTranscriber
from life_assist_dm.life_assist_dm.llm.gpt_utils import SentenceCorrector

from life_assist_dm_msgs.srv import STTListen

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self.model_load = False
        self.set_param()
        self.load_model()

        self.raw_text = ""
        self.raw_texts = []
        self.sentence = ""

        self.call = False

        self.stt_start_service = self.create_service(STTListen, 'stt_listen', self.listening)

    def load_model(self):
        try:
            self.stt_model = WhisperTranscriber(model_name=self.whisper_model)
            self.gpt_corrector = SentenceCorrector(model_name=self.gpt_model)
            self.model_load = True
        except Exception as e:
            self.get_logger().error(f"{e}")


    def set_param(self):
        self.whisper_model = self.declare_parameter('whisper_model', 'base').get_parameter_value().string_value
        self.gpt_model = self.declare_parameter('gpt_model', 'gpt-4o-mini-2024-07-18').get_parameter_value().string_value
        self.duration  = self.declare_parameter('duration', 5).get_parameter_value().integer_value
        self.call_sign = self.declare_parameter('call_sign', ['로봇', 'robot']).get_parameter_value().string_array_value

        # self.get_logger().info(f"whisper model: {self.whisper_model}\n"
        #                        f"stt gpt model: {self.gpt_model}\n"
        #                        f"duration: {self.duration}\n"
        #                        f"call_sign: {self.call_sign}")

    # def send_stt_ready(self, request, response):
    #     if self.model_load:
    #         response.success = self.model_load
    #         return response

    def listening(self, request, response):
        stt_type  = request.type
        self.raw_text = ""
        self.raw_texts = []
        self.sentence = ""

        if stt_type == 'call':
            self.get_logger().info('stt listening - CALL')
            self.listening_call()
            if self.call:
                response.success = True
            else:
                response.success = False
            return response

        else:
            self.get_logger().info('stt listening - COMMAND')
            self.listening_loop()
            if self.sentence:
                response.success = True
                response.command = self.sentence
            else:
                response.success = False
                response.command = ""
            return response


    def listening_call(self):
        self.call = False

        # while not self.call:
        raw_text = self.stt_model.transcribe_from_mic(duration=self.duration)
        self.get_logger().info(f'stt raw_text: {raw_text}')

        if raw_text.strip() == "":
            self.raw_texts.clear()
        else:
            words = raw_text.split()

            if any(call in word or word in call for word in words for call in self.call_sign):
                self.call = True
                self.raw_text = raw_text
                self.get_logger().info(f'stt call detect: {self.raw_text}')



    def listening_loop(self):
        self.sentence = ""
        self.raw_text = ""
        self.raw_texts = []

        while True:
            self.get_logger().info('STT - command waiting...')
            raw_text = self.stt_model.transcribe_from_mic(duration=self.duration)
            raw_text = raw_text.strip()

            if raw_text:
                self.raw_text = raw_text
                self.raw_texts.append(self.raw_text)

            if len(self.raw_texts) and not raw_text:
                break

        raw_sentence = ''.join(self.raw_texts)

        if raw_sentence:
            corrected_sentence = self.gpt_corrector.correct(raw_sentence)
            self.sentence = corrected_sentence
            self.get_logger().info(f'STT - corredcted_sentence: {self.sentence}')


    # def send_call_request(self):
    #
    #     req = STTCall.Request()
    #     req.text = self.raw_text
    #     req.call = True
    #
    #     future = self.stt_call_client.call_async(req)
    #
    #     # 비동기 처리 → 응답 기다림 (옵션)
    #     rclpy.spin_until_future_complete(self, future)
    #     if future.result() is not None:
    #         self.get_logger().info(f'Service result: {future.result().success}')
    #     else:
    #         self.get_logger().warn('Service call failed')


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().info(f'{e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
