
import tensorflow as tf
jls_extract_var = tensorflow
#from tensorflow.keras.models import Sequential
from jls_extract_var.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Création du modèlePPPP
model = Sequential()

# Ajout d'une couche d'embedding (ajustez la dimension de l'embedding en fonction de votre tâche)
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))

# Ajout d'une couche bidirectionnelle LSTM
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))

# Vous pouvez ajouter plusieurs couches LSTM bidirectionnelles si nécessaire
# model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
# ...

# Ajout d'une couche Dense pour la sortie
model.add(Dense(units=num_classes, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Afficher le résumé du modèle
model.summary()
