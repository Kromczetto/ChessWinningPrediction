import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Function dividing string to array
def divide_string_into_array(text):
    lines = text.strip().split('\n')
    return lines

# Reading data
data = pd.read_csv('games.csv')

# Prepair moves
moves = data['moves'].to_string(index=False)
moves = re.sub(r'\d+\.', '', moves)
arrayOfMoves = divide_string_into_array(moves)
arrayOfMoves = [move.strip() for move in arrayOfMoves]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(arrayOfMoves)
moves_seq = tokenizer.texts_to_sequences(arrayOfMoves)
max_sequence_length = max(len(seq) for seq in moves_seq)
moves_padded = pad_sequences(moves_seq, maxlen=max_sequence_length, padding='post')
#
# # Przygotowanie etykiet
# winner_numeric = np.array([1 if w == 'white' else 0 for w in data['winner']])
#
# # Dividing data into training and testing data
# x_train, x_test, y_train, y_test = train_test_split(
#     moves_padded, winner_numeric, test_size=0.4
# )
#
# # Creating model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 32, input_length=max_sequence_length),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(32, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
#
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
#
# # Model training
# history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
#
# model.evaluate(x_test, y_test, verbose=2)
#
# # Saving model
# model.save('chessWinningPredictionModel.h5')

# Loading model
model = tf.keras.models.load_model('chessWinningPredictionModel.h5')
# sample of moves
sample_moves = [
    "e4 e5 Nf3 d6 Bc4 Be6 d3 Bxc4 dxc4 c5 O-O h6 Nc3 Nf6 Nd5 Nxe4 Re1 Nf6 Nxf6+ Qxf6 Qd5 Nd7 Qxb7 Rb8 Qxa7 Be7 Bd2 Qf5 c3 Qc2 Rab1 O-O Rec1 Qd3 b4 Rfd8 bxc5 dxc5 Rxb8 Rxb8 Rd1 Rb1 Qa8+ Bf8 Qa4 Rxd1+ Qxd1 e4 Ne1 Qd6 h3 Qe5 a4 Bd6 f4 Qf5 Be3 Bxf4 Bxf4 Qxf4 Qxd7 Qe3+ Kh2 Qxe1 Qe8+ Kh7 Qxf7 e3 Qf5+ Kg8 Qd5+ Kh7 Qxc5 e2 Qf5+ Kg8 Qe6+ Kh7 Qe4+ Kg8 a5 Qd1 Qa8+ Kh7 Qe4+ Kg8 Qd5+ Qxd5 cxd5 e1=Q a6 Qa1 d6 Qxa6 d7 Qa8 c4 Qd8 c5 Qxd7 c6",
    "e4 e6 Nf3 d5 exd5 exd5 Qe2+ Be7 Nc3 Nf6 d4 O-O g4 Bxg4 Bg2 Nbd7 O-O c6 h3 Bh5 Bf4 Re8 Rfe1 Bd6 Be3 Nb6 Ng5 h6 Nf3 Nc4 b3 Nxe3 fxe3 Bf4 Nd1 Nd7 Qd2 Bd6 c4 dxc4 bxc4 a5 a4 Bb4 Nc3 Qc7 Rad1 Bxc3 Qxc3 Qg3 e4 Re6 d5 Rf6 Re3 Nc5 Ra1 Rg6 Ra2 Nxe4",
    "e4 Nc6 d4 e5 d5 Nce7 c3 Ng6 b4",
    "e4 c5 d4 cxd4 Qxd4 Nc6 Qa4 Nf6 Nc3 g6 Nf3 Bg7 Bb5 O-O O-O a6 Bg5 axb5 Qxa8 d5 Rab1 dxe4 Rbd1 Qa5 Qxa5 Nxa5 Nxb5 exf3 gxf3 Nc4 Na7 Nxb2 Rb1 Nc4 Rb4 Nd2 Rd1 Nxf3+ Kf1 Nxh2+ Kg1 Nf3+ Kg2 Nxg5 Rbd4 Bh3+ Kg3 Nge4+ Kxh3 Nxf2+ Kg2 Nxd1 Rxd1 Ne4 Rd4 Nc3 Rb4 Nxa2 Rxb7 Nc3 Rxe7 Nd5 Rc7 Nxc7",
    "e4 c5 c3 d6 Bc4 Nf6 d3 a6 a4 e6 Bg5 Be7 Bxf6 Bxf6 f4 Nc6 Nf3 Bd7 Nbd2 Qe7 O-O O-O g3 g6 e5 dxe5 fxe5 Nxe5 Nxe5 Bxe5 Qf3 Bc6 Qe3 Bg7 Nb3 b6 Nd2 Qd7 Nf3 Bxa4 Ne5 Bxe5 Qxe5 Bb5 b3 Bxc4 dxc4 Qc6 Qf6 a5 h4 h5 Qg5 Qe4 Rae1 Qd3 Rc1 Rad8 Kg2 Qd2+ Rf2 Qxg5 hxg5 Rd3 Rf3 Rfd8 Rcf1 Rxf3 Rxf3 Rd2+ Kh3 Rb2 g4 hxg4+ Kxg4 Rxb3 Kf4 Kf8 Ke5 Ke7 Rd3 Rb2 Rg3 Re2+ Kf4 Rf2+ Ke3 Rf5 Kd3 f6 gxf6+ Kxf6 Kc2 g5 Kb3 Rf4 Ka4 Rxc4+ Kb5 Rf4",
    "e4 c5 Nf3 d6 Bc4 Nf6 d3 a6 a3 g6 b4 Bg7 Bb2 O-O Nc3 cxb4 axb4 Bg4 h3 Bd7 O-O b5 Bb3 Nc6 Nd5 Nxd5 Bxg7 Kxg7 Bxd5 e6 Bb3 Nxb4 Qd2 Nc6 Qc3+ Qf6 Qxf6+ Kxf6 Ra3 Ne7 Rfa1 Bc8 Nd4 Bb7 c4 d5 cxb5 axb5 Rxa8 Rxa8 Rxa8 Bxa8 exd5 Nxd5 Nxb5 Nf4 f3 Nxd3 Bc2 Nc5 Nd6 Bd5 Kf2 h5 Ke3 Ke7 Nc8+ Kd7 Nb6+ Kc6 Nc8 Kd7 Nb6+ Kc6 Na4 Nxa4 Bxa4+ Kd6 Be8 Ke7 Bb5 h4 Kf4 Kf6 g3 hxg3 Kxg3 Kg5 f4+ Kf5 Be8 f6 h4 Bc4 Bc6 e5 fxe5 Kxe5 Be8 Bd3 Ba4 f5 Bb3 f4+ Kf2"

    #white
    #black
    #black
    #black
    #black
    #white
]

sample_moves_seq = tokenizer.texts_to_sequences(sample_moves)
sample_moves_padded = pad_sequences(sample_moves_seq, maxlen=max_sequence_length, padding='post')

# Resutl prediction
predictions = model.predict(sample_moves_padded)

# Show result
for i, pred in enumerate(predictions):
    print(f"Przewidywany wynik dla gry {i+1}: {pred[0]}")
    if pred[0] >= 0.5:
        print("Białe powinny wygrać.")
    else:
        print("Czarne powinny wygrać.")
