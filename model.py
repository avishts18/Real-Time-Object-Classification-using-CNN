
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Embedding, RepeatVector, Concatenate, Activation, Input
from keras.applications import ResNet50
from keras.preprocessing.sequence import pad_sequences

EMBEDDING_SIZE = 128
MAX_LEN = 40

vocab = np.load('vocab.npy', allow_pickle=True).item()
inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

image_input = Input(shape=(2048,))
img_dense = Dense(EMBEDDING_SIZE, activation='relu')(image_input)
img_repeat = RepeatVector(MAX_LEN)(img_dense)

lang_input = Input(shape=(MAX_LEN,))
lang_embed = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE, input_length=MAX_LEN)(lang_input)
lang_lstm = LSTM(256, return_sequences=True)(lang_embed)
lang_td = TimeDistributed(Dense(EMBEDDING_SIZE))(lang_lstm)

merged = Concatenate()([img_repeat, lang_td])
decoder_lstm1 = LSTM(128, return_sequences=True)(merged)
decoder_lstm2 = LSTM(512, return_sequences=False)(decoder_lstm1)
output = Dense(vocab_size)(decoder_lstm2)
output = Activation('softmax')(output)

caption_model = Model(inputs=[image_input, lang_input], outputs=output)
caption_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
caption_model.load_weights('mine_model_weights.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))
    return img

def extract_features(img_array):
    return resnet.predict(img_array).reshape(1, 2048)

def predict_caption(image_path):
    img_array = preprocess_image(image_path)
    features = extract_features(img_array)
    in_text = ['startofseq']
    final_caption = ''
    for _ in range(20):
        sequence_input = [vocab[word] for word in in_text if word in vocab]
        padded_input = pad_sequences([sequence_input], maxlen=MAX_LEN, padding='post').reshape(1, MAX_LEN)
        prediction = caption_model.predict([features, padded_input])
        predicted_index = np.argmax(prediction)
        predicted_word = inv_vocab.get(predicted_index, '')
        if predicted_word == 'endofseq':
            break
        final_caption += ' ' + predicted_word
        in_text.append(predicted_word)
    return final_caption.strip()
