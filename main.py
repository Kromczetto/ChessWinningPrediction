import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Funkcja do podziału tekstu na sekwencje
def divide_string_into_array(text):
    lines = text.strip().split('\n')
    return lines

# Wczytanie danych
data = pd.read_csv('games.csv')

# Przetworzenie ruchów szachowych
moves = data['moves'].to_string(index=False)
moves = re.sub(r'\d+\.', '', moves)
arrayOfMoves = divide_string_into_array(moves)
arrayOfMoves = [move.strip() for move in arrayOfMoves]

# Przygotowanie danych w formacie do modelu
tokenizer = Tokenizer()
tokenizer.fit_on_texts(arrayOfMoves)
moves_seq = tokenizer.texts_to_sequences(arrayOfMoves)
max_sequence_length = max(len(seq) for seq in moves_seq)
moves_padded = pad_sequences(moves_seq, maxlen=max_sequence_length, padding='post')

# Przygotowanie etykiet
winner_numeric = np.array([1 if w == 'white' else 0 for w in data['winner']])

# Podział na zbiory treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(
    moves_padded, winner_numeric, test_size=0.4
)

# Tworzenie modelu
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 32, input_length=max_sequence_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Ocena modelu
model.evaluate(x_test, y_test, verbose=2)

# Wyświetlenie accuracy i val_accuracy
print("Accuracy: ", history.history['accuracy'])
print("Val Accuracy: ", history.history['val_accuracy'])

# Przykładowe sekwencje ruchów szachowych do przewidzenia
sample_moves = [
    "d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5 Bf4"
]

# Tokenizacja i padding dla nowych danych
sample_moves_seq = tokenizer.texts_to_sequences(sample_moves)
sample_moves_padded = pad_sequences(sample_moves_seq, maxlen=max_sequence_length, padding='post')

# Przewidywanie wyniku
predictions = model.predict(sample_moves_padded)

# Wyświetlenie wyników
for i, pred in enumerate(predictions):
    print(f"Przewidywany wynik dla gry {i+1}: {pred[0]}")
    if pred[0] > 0.5:
        print("Białe powinny wygrać.")
    elif pred[0] == 0.5:
        print("remis")
    else:
        print("Czarne powinny wygrać.")
