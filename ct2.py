import ctranslate2
from ctranslate2.converters import TransformersConverter

model_name_or_path = "nb-whisper-small-ct2"
output_dir = "nb-whisper-small-ct2-copy"

converter = TransformersConverter(model_name_or_path)
converter.convert(output_dir, quantization="float16", force=True)