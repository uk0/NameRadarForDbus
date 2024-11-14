import platform
import os
import json
import pyaudio
import time
import difflib
import numpy as np
from collections import deque
from dotenv import load_dotenv
import wave
from tempfile import NamedTemporaryFile
from faster_whisper import WhisperModel


class NotificationSystem:
    """通知系统的抽象基类"""

    def send_notification(self, title, message):
        raise NotImplementedError


class LinuxNotification(NotificationSystem):
    """Linux下使用D-Bus的通知实现"""

    def __init__(self):
        print("Using Dbus notification system")
        import dbus
        self.bus = dbus.SessionBus()
        self.notifications = self.bus.get_object(
            'org.freedesktop.Notifications',
            '/org/freedesktop/Notifications'
        )
        self.interface = dbus.Interface(
            self.notifications,
            'org.freedesktop.Notifications'
        )

    def send_notification(self, title, message):
        self.interface.Notify(
            "NameRadar",
            0,
            "dialog-information",
            title,
            message,
            [],
            {},
            int(os.getenv('NOTIFICATION_TIMEOUT', '5000'))
        )


class MacOSNotification(NotificationSystem):
    """MacOS下使用osascript的通知实现"""

    def send_notification(self, title, message):
        print("Using MacOS notification system")
        sound_option = 'sound name "default"' if os.getenv(
            'NOTIFICATION_SOUND', 'true').lower() == 'true' else ''
        os.system("""
               osascript -e 'display notification "{}" with title "{}" "{}"'
               """.format(message.replace('"', '\\"'), title.replace('"', '\\"'), sound_option.replace('"', '\\"')))


class AudioProcessor:
    """音频处理类"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=int(os.getenv('AUDIO_BUFFER_SIZE', '3')))
        self.recording = []
        self.is_recording = False

    def start_recording(self):
        self.recording = []
        self.is_recording = True

    def stop_recording(self):
        self.is_recording = False
        return self.recording

    def process_audio(self, audio_data):
        if self.is_recording:
            self.recording.append(audio_data)

        # Convert to numpy array for noise filtering
        audio_array = np.frombuffer(audio_data, dtype=np.int16).copy()
        # 简单的阈值过滤
        threshold = np.std(audio_array) * float(os.getenv('NOISE_THRESHOLD', '0.5'))
        audio_array[np.abs(audio_array) < threshold] = 0
        return audio_array.tobytes()

    def save_audio(self, filename):
        if not self.recording:
            return None

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.recording))
        return filename


class NameMatcher:
    """名字匹配类"""

    def __init__(self, target_names, threshold=0.6):
        if isinstance(target_names, str):
            self.target_names = [target_names]
        else:
            self.target_names = target_names
        self.threshold = threshold
        print(f"Monitoring names: {', '.join(self.target_names)}")

    def is_similar(self, text):
        """检查文本是否与任何目标名字相似"""
        words = text.split()
        for word in words:
            for target_name in self.target_names:
                similarity = difflib.SequenceMatcher(
                    None, target_name, word).ratio()
                if similarity > self.threshold:
                    print(
                        f"Match found: '{word}' similar to '{target_name}' (similarity: {similarity:.2f})")
                    return True, target_name
        return False, None


class WhisperNameRadar:
    def __init__(self, names, model_path, device="cpu", compute_type="int8", similarity_threshold=0.6,
                 notification_cooldown=2, confidence_threshold=0.5):
        self.names = names if isinstance(names, list) else [names]
        self.setup_audio()
        self.notification_system = None
        self.setup_notification()
        self.audio_processor = AudioProcessor()
        self.name_matcher = NameMatcher(
            self.names, threshold=similarity_threshold)
        self.last_notification_time = 0
        self.notification_cooldown = notification_cooldown
        self.confidence_threshold = confidence_threshold

        # Initialize Whisper model
        print("Loading Whisper model...")
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
        print("Whisper model loaded")

    def setup_notification(self):
        """根据操作系统设置适当的通知系统"""
        system = platform.system()
        if system == 'Darwin':
            self.notification_system = MacOSNotification()
        elif system == 'Linux':
            self.notification_system = LinuxNotification()
        else:
            raise OSError(f"Unsupported operating system: {system}")

    def setup_audio(self):
        self.audio = pyaudio.PyAudio()
        best_device = self.get_best_input_device()
        print(f"\nUsing input device: {best_device['name']}")

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=int(os.getenv('AUDIO_FRAME_SIZE', '8000')),
            input_device_index=best_device['index']
        )

    def get_best_input_device(self):
        """获取最佳输入设备"""
        print("\nAvailable audio input devices:")

        try:
            default_device_info = self.audio.get_default_input_device_info()
            default_device_index = default_device_info['index']
            best_device = {'index': default_device_index,
                           'name': default_device_info['name']}

            for i in range(self.audio.get_device_count()):
                try:
                    dev_info = self.audio.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:
                        print(f"Device {i}: {dev_info['name']}")
                        if i == default_device_index:
                            print(
                                f"*** Default Device {i}: {dev_info['name']} ***")
                except Exception as e:
                    print(f"Error getting info for device {i}: {e}")

            print(
                f"\nSelected device: {best_device['name']} (index: {best_device['index']})")
            return best_device

        except Exception as e:
            print(f"Error getting default device: {e}")
            print("Falling back to default device index None")
            return {'index': None, 'name': 'system default'}

    def transcribe_audio(self, audio_file):
        """使用Whisper转录音频"""
        try:
            segments, info = self.model.transcribe(
                audio_file,
                language="zh",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200
                )
            )

            # 将生成器转换为列表
            segments_list = list(segments)

            # 如果没有识别到任何片段，直接返回
            if not segments_list:
                return "", 0.0

            # 合并文本并计算平均置信度
            text = " ".join([segment.text for segment in segments_list])
            confidence = sum([segment.avg_logprob for segment in segments_list]) / len(segments_list)
            print(f"Transcription confidence: {confidence:.2f}")
            print(f"Transcription text: {text}")
            return text, confidence
        except Exception as e:
            print(f"Transcription error: {e}")
            return "", 0.0

    def should_notify(self, text):
        """判断是否应该发送通知"""
        current_time = time.time()
        if current_time - self.last_notification_time < self.notification_cooldown:
            return False, None

        is_match, matched_name = self.name_matcher.is_similar(text)
        if is_match:
            self.last_notification_time = current_time
            return True, matched_name
        return False, None

    def start_monitoring(self):
        print(f"开始监听以下名字: {', '.join(self.names)}...")
        RECORD_SECONDS = 2  # 录音时长

        try:
            while True:
                print("Recording...")
                self.audio_processor.start_recording()

                # Record for RECORD_SECONDS
                for _ in range(0, int(16000 / 1024 * RECORD_SECONDS)):
                    data = self.stream.read(1024, exception_on_overflow=False)
                    self.audio_processor.process_audio(data)

                # Save recording to temporary file
                with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    self.audio_processor.save_audio(temp_audio.name)

                    # Transcribe audio
                    text, confidence = self.transcribe_audio(temp_audio.name)

                    # Clean up temporary file
                    os.unlink(temp_audio.name)

                if text and confidence > self.confidence_threshold:
                    print(f"识别到 [{confidence:.2f}]: {text}")

                    should_notify, matched_name = self.should_notify(text)
                    if should_notify:
                        message = f"检测到名字 '{matched_name}'!\n原文: {text}"
                        print(message)
                        self.notification_system.send_notification(
                            "Name Detected!", message)

                self.audio_processor.stop_recording()
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n停止监听")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        except Exception as e:
            print(f"Cleanup error: {e}")


def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []

    try:
        import faster_whisper
    except ImportError:
        missing_deps.append("faster-whisper")

    try:
        import pyaudio
    except ImportError:
        missing_deps.append("pyaudio")

    if platform.system() == 'Linux':
        try:
            import dbus
        except ImportError:
            missing_deps.append("dbus-python")

    try:
        import dotenv
    except ImportError:
        missing_deps.append("python-dotenv")

    if missing_deps:
        print("Missing dependencies:", ", ".join(missing_deps))
        print("Please install them using:")
        print("pip install " + " ".join(missing_deps))
        return False
    return True


def load_config():
    """加载配置"""
    load_dotenv()

    # 获取并处理名字列表
    names_str = os.getenv('NAMES_TO_MONITOR', '赵希,小赵')
    names = [name.strip() for name in names_str.split(',')]

    # 获取其他配置，提供默认值
    config = {
        'names': names,
        'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', '0.6')),
        'notification_cooldown': float(os.getenv('NOTIFICATION_COOLDOWN', '2')),
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
        'model_path': os.getenv('WHISPER_MODEL_PATH', 'models/whisper-small-ct2.bin'),
        'device': os.getenv('DEVICE', 'cpu'),
        'compute_type': os.getenv('COMPUTE_TYPE', 'int8')
    }

    return config


def main():
    if not check_dependencies():
        return

    config = load_config()
    print("\nLoaded configuration:")
    print(f"Names to monitor: {config['names']}")
    print(f"Model path: {config['model_path']}")
    print(f"Device: {config['device']}")
    print(f"Compute type: {config['compute_type']}")
    print(f"Similarity threshold: {config['similarity_threshold']}")
    print(f"Notification cooldown: {config['notification_cooldown']}s")
    print(f"Confidence threshold: {config['confidence_threshold']}")

    # 确保模型文件存在
    if not os.path.exists(config['model_path']):
        print(f"Model file not found at: {config['model_path']}")
        return

    radar = WhisperNameRadar(
        names=config['names'],
        model_path=config['model_path'],
        device=config['device'],
        compute_type=config['compute_type'],
        similarity_threshold=config['similarity_threshold'],
        notification_cooldown=config['notification_cooldown'],
        confidence_threshold=config['confidence_threshold']
    )
    radar.start_monitoring()


if __name__ == "__main__":
    main()