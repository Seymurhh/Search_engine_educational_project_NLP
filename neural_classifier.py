import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

class TopicClassifier:
    def __init__(self, input_dim=384, num_topics=5, hidden_units=128, dropout_rate=0.3, learning_rate=0.001, l2_rate=0.0):
        self.input_dim = input_dim
        self.num_topics = num_topics
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_rate = l2_rate
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        """
        """
        # Use explicit tf.keras to ensure consistency
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hidden_units, 
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_rate), 
                input_shape=(self.input_dim,)
            ),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(
                self.hidden_units // 2, 
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_rate)
            ),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.num_topics, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, embeddings, labels, epochs=20, batch_size=32):
        """
        Trains the model.
        embeddings: numpy array of shape (num_samples, 384)
        labels: numpy array of shape (num_samples,) containing topic IDs
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
        
        # Train
        print("Training Neural Classifier...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0 # Suppress output for Streamlit
        )
        
        # Evaluate
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy, self.history

    def predict(self, embedding):
        """
        Predicts the topic for a single embedding.
        """
        # Reshape for batch of 1
        embedding = np.array(embedding).reshape(1, -1)
        prediction = self.model.predict(embedding, verbose=0)
        predicted_topic = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_topic, confidence
