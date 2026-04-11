# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense

# print("TensorFlow:", tf.__version__)
# print("Keras OK, semua import aman")


from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("indobenchmark/indobert-base-p2")
cfg.save_pretrained("Indobert_Model")
print("config.json berhasil disimpan")


