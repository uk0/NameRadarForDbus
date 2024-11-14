import platform
import os
import json
from vosk import Model, KaldiRecognizer
import pyaudio
import time
import difflib
import numpy as np
from collections import deque
from dotenv import load_dotenv


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
        # 使用osascript发送通知
        sound_option = 'sound name "default"' if os.getenv('NOTIFICATION_SOUND', 'true').lower() == 'true' else ''
        os.system("""
               osascript -e 'display notification "{}" with title "{}" "{}"'
               """.format(message.replace('"', '\\"'), title.replace('"', '\\"'),sound_option.replace('"', '\\"')))


class AudioProcessor:
    """音频处理类"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer = deque(maxlen=int(os.getenv('AUDIO_BUFFER_SIZE', '3')))  # 配置缓冲区大小

    def filter_noise(self, audio_data):
        """简单的噪声过滤"""
        # 将字节转换为numpy数组，并确保是可写的
        audio_array = np.frombuffer(audio_data, dtype=np.int16).copy()
        # 简单的阈值过滤
        threshold = np.std(audio_array) * float(os.getenv('NOISE_THRESHOLD', '0.5'))
        audio_array[np.abs(audio_array) < threshold] = 0
        return audio_array.tobytes()


class NameMatcher:
    """名字匹配类"""

    def __init__(self, target_names, threshold=0.6):
        # 支持字符串或列表作为输入
        if isinstance(target_names, str):
            self.target_names = [target_names]
        else:
            self.target_names = target_names
        self.threshold = threshold
        print(f"Monitoring names: {', '.join(self.target_names)}")

    def is_similar(self, text):
        """检查文本是否与任何目标名字相似"""
        # 将文本分割成单词
        words = text.split()
        for word in words:
            for target_name in self.target_names:
                similarity = difflib.SequenceMatcher(None, target_name, word).ratio()
                if similarity > self.threshold:
                    print(f"Match found: '{word}' similar to '{target_name}' (similarity: {similarity:.2f})")
                    return True, target_name
        return False, None


class NameRadar:
    def __init__(self, names, model_path, similarity_threshold=0.6,
                 notification_cooldown=2, confidence_threshold=0.5):
        self.names = names if isinstance(names, list) else [names]
        self.model_path = model_path
        self.setup_vosk()
        self.notification_system = None
        self.setup_notification()
        self.audio_processor = AudioProcessor()
        self.name_matcher = NameMatcher(self.names, threshold=similarity_threshold)
        self.last_notification_time = 0
        self.notification_cooldown = notification_cooldown
        self.confidence_threshold = confidence_threshold

    def setup_notification(self):
        """根据操作系统设置适当的通知系统"""
        system = platform.system()
        if system == 'Darwin':  # MacOS
            self.notification_system = MacOSNotification()
            print("Using MacOS notification system")
        elif system == 'Linux':
            self.notification_system = LinuxNotification()
            print("Using Linux notification system")
        else:
            raise OSError(f"Unsupported operating system: {system}")

    def setup_vosk(self):
        print(f"Loading model from: {self.model_path}")
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, int(os.getenv('AUDIO_SAMPLE_RATE', '16000')))
        self.recognizer.SetWords(True)  # 启用详细输出

        self.audio = pyaudio.PyAudio()

        # 获取最佳输入设备
        best_device = self.get_best_input_device()
        print(f"\nUsing input device: {best_device['name']}")

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=int(os.getenv('AUDIO_SAMPLE_RATE', '16000')),
            input=True,
            frames_per_buffer=int(os.getenv('AUDIO_FRAME_SIZE', '8000')),
            input_device_index=best_device['index']
        )
        print("\nAudio stream opened successfully")

    def get_best_input_device(self):
        """获取最佳输入设备"""
        print("\nAvailable audio input devices:")

        try:
            # 获取默认输入设备的索引
            default_device_info = self.audio.get_default_input_device_info()
            default_device_index = default_device_info['index']
            best_device = {'index': default_device_index, 'name': default_device_info['name']}

            # 打印所有可用设备信息
            for i in range(self.audio.get_device_count()):
                try:
                    dev_info = self.audio.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:  # 只显示输入设备
                        print(f"Device {i}: {dev_info['name']}")
                        if i == default_device_index:
                            print(f"*** Default Device {i}: {dev_info['name']} ***")
                except Exception as e:
                    print(f"Error getting info for device {i}: {e}")

            print(f"\nSelected device: {best_device['name']} (index: {best_device['index']})")
            return best_device

        except Exception as e:
            print(f"Error getting default device: {e}")
            print("Falling back to default device index None")
            return {'index': None, 'name': 'system default'}

    def should_notify(self, text):
        """判断是否应该发送通知"""
        current_time = time.time()
        if current_time - self.last_notification_time < self.notification_cooldown:
            return False, None

        # 检查名字匹配
        is_match, matched_name = self.name_matcher.is_similar(text)
        if is_match:
            self.last_notification_time = current_time
            return True, matched_name
        return False, None

    def process_result(self, result):
        """处理识别结果"""
        if not isinstance(result, dict):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                return None, 0

        text = result.get("text", "")
        if not text:
            return None, 0

        # 获取置信度（如果有的话）
        confidence = 1.0
        if "result" in result and result["result"]:
            confidence = sum(word.get("conf", 1.0) for word in result["result"]) / len(result["result"])

        return text, confidence

    def start_monitoring(self):
        print(f"开始监听以下名字: {', '.join(self.names)}...")
        try:
            while True:
                data = self.stream.read(int(os.getenv('AUDIO_FRAME_SIZE', '4000')),
                                        exception_on_overflow=False)
                if len(data) == 0:
                    continue

                filtered_data = self.audio_processor.filter_noise(data)
                self.audio_processor.buffer.append(filtered_data)

                if self.recognizer.AcceptWaveform(filtered_data):
                    text, confidence = self.process_result(self.recognizer.Result())

                    if text and confidence > self.confidence_threshold:
                        print(f"识别到 [{confidence:.2f}]: {text}")

                        should_notify, matched_name = self.should_notify(text)
                        if should_notify:
                            message = f"检测到名字 '{matched_name}'!\n原文: {text}"
                            print(message)
                            self.notification_system.send_notification("Name Detected!", message)
                else:
                    # 如果启用了调试模式，才打印部分结果
                    if os.getenv('LOG_PARTIAL_RESULTS', 'false').lower() == 'true':
                        partial_text, _ = self.process_result(self.recognizer.PartialResult())
                        if partial_text:
                            print(f"Partial: {partial_text}")

                time.sleep(float(os.getenv('PROCESSING_INTERVAL', '0.1')))

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


def get_default_model_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, os.getenv('VOSK_MODEL_PATH', "vosk-model-small-cn-0.3"))


def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []

    try:
        import vosk
    except ImportError:
        missing_deps.append("vosk")

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
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    }

    return config


def main():
    """主函数"""
    if not check_dependencies():
        return

    # 加载配置
    config = load_config()
    print("\nLoaded configuration:")
    print(f"Names to monitor: {config['names']}")
    print(f"Similarity threshold: {config['similarity_threshold']}")
    print(f"Notification cooldown: {config['notification_cooldown']}s")
    print(f"Confidence threshold: {config['confidence_threshold']}")

    # 获取默认模型路径
    model_path = get_default_model_path()

    # 确保模型目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Please download the Vosk model and place it in: {model_path}")
        print("You can download models from: https://alphacephei.com/vosk/models")
        return

    # 创建实例并开始监听
    radar = NameRadar(
        names=config['names'],
        model_path=model_path,
        similarity_threshold=config['similarity_threshold'],
        notification_cooldown=config['notification_cooldown'],
        confidence_threshold=config['confidence_threshold']
    )
    radar.start_monitoring()


if __name__ == "__main__":
    main()