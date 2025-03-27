import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
bert_preprocess = hub.KerasLayer(os.path.join(base_dir, "..", "tfhub_modules", "bert_preprocess"), trainable=False)
bert_encoder = hub.KerasLayer(os.path.join(base_dir, "..", "tfhub_modules", "bert_encoder"), trainable=False)

def get_bert_embeddings(texts):
    text_inputs = tf.constant(texts)
    tokenized_text = bert_preprocess(text_inputs)
    outputs = bert_encoder(tokenized_text)
    print(outputs.keys())
    return outputs['pooled_output'].numpy()