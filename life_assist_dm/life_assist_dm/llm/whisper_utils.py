import tempfile

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav



class WhisperTranscriber:
    """
    """

    def __init__(self, model_name="base"):
        """
        - model_name: 'tiny', 'base', 'small', 'medium', 'large'
        """
        print(f"whisper model '{model_name}' is selected.")
        self.model = whisper.load_model(model_name)
        self.samplerate = 16000  # Whisper는 16kHz 샘플레이트를 사용합니다.
        self.channels = 1
        print("whisper model is loaded.")

    def _record_audio(self, duration: int) -> np.ndarray:
        """
        """
        print(f"\n🎙️  {duration} secs of recording")
        recording = sd.rec(
            int(duration * self.samplerate),
            samplerate=self.samplerate,
            channels=self.channels,
            dtype='int16'
        )
        sd.wait()
        print("recording is done.")
        return recording

    def transcribe_from_mic(self, duration: int = 5) -> str:
        """
        """
        audio_data = self._record_audio(duration)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            wav.write(f.name, self.samplerate, audio_data)
            result = self.model.transcribe(f.name, fp16=False, language="ko")

        return result.get('text', '').strip()