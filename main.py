import pandas as pd
import re

def divide_string_into_array(text):
    lines = text.strip().split('\n')
    return lines

data = pd.read_csv('games.csv')

moves = data['moves'].to_string(index=False)
moves = re.sub(r'\d+\.', '', moves)
arrayOfMoves = divide_string_into_array(moves)
arrayOfMoves = [move.strip() for move in arrayOfMoves]



finalData = []
for i in range(len(data)):
    movesPlayed = arrayOfMoves[i]
    winner = data['winner'].iloc[i]
   # black_rating = int(data['black_rating'].iloc[i])
   # white_rating = int(data['white_rating'].iloc[i])
    finalData.append({"movesPlayed": movesPlayed,
                      "winner": winner
                      #"black_rating": black_rating,
                      #"white_rating": white_rating
                      })
from sklearn.model_selection import train_test_split

movesPlayed = [row["movesPlayed"] for row in finalData]
winner = [row["winner"] for row in finalData]

x_trainig, x_testing, y_training, y_testing = train_test_split(
    movesPlayed, winner, test_size=0.4
)

import tensorflow as tf

#Jak dodamy to zmienic na wiecej input_shpae to jest to ile jest danych wejsciowych chyba
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)

model.fit(x_trainig, y_training, epochs=20)

model.evaluate(x_testing, y_testing, verbose=2)