import rclpy
from rclpy.node import Node

from gtts import gTTS
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import io

from life_assist_dm_msgs.srv import TTSSpeak


class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        self.srv = self.create_service(TTSSpeak, 'tts_speak', self.handle_tts_request)

    def handle_tts_request(self, request, response):
        text = request.text
        self.get_logger().info(f'ROBOT -> USER: {text}')

        speak_success = self.speak_text(text)
        response.success = speak_success
        return response

    def speak_text(self, text):
        try:
            # gTTS로 음성 생성
            tts = gTTS(text=text, lang='ko')
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)

            # MP3 -> AudioSegment -> NumPy 배열
            audio = AudioSegment.from_file(mp3_fp, format="mp3")
            audio = audio.set_channels(1).set_frame_rate(44100)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples /= np.iinfo(audio.array_type).max  # Normalize

            # 재생
            sd.play(samples, samplerate=audio.frame_rate)
            sd.wait()
            return True
        except Exception as e:
            self.get_logger().error(f"TTS speaking error: {e}")
            return False


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().info(f'{e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
