import os
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

class FallDetectionModel:
    def __init__(self, model_path, feedback_threshold=5, save_model_path="artifacts/retraining/updated_model.h5"):
        self.model = load_model(model_path)
        self.incorrect_predictions = []
        self.feedback_threshold = feedback_threshold
        self.save_model_path = save_model_path

    def predict_with_feedback(self, image_path):
        """Make a prediction and ask for user feedback."""
        # Load and preprocess the image
        image_data = load_img(image_path, target_size=(224, 224))
        image_data = img_to_array(image_data) / 255.0  # Normalize image data
        image_data = np.expand_dims(image_data, axis=0)

        # Make the prediction
        prediction = self.model.predict(image_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class: {predicted_class} (0: No Fall, 1: Fall)")

        # Ask the user if the prediction is correct
        feedback = input("Is the prediction correct? (yes/no): ").strip().lower()

        if feedback == 'no':
            correct_label = input("What is the correct label (0 for No Fall, 1 for Fall)? ").strip()
            self.save_incorrect_prediction(image_path, correct_label)

        # If enough incorrect predictions are collected, trigger retraining
        if len(self.incorrect_predictions) >= self.feedback_threshold:
            print("Retraining the model with new data...")
            self.retrain_model()

    def save_incorrect_prediction(self, image_path, correct_label):
        """Store incorrect prediction for retraining."""
        self.incorrect_predictions.append({
            "image_path": image_path,
            "correct_label": int(correct_label)
        })
        print(f"Incorrect prediction saved for retraining.")

    def retrain_model(self):
        """Retrain the model using incorrect predictions."""
        images = []
        labels = []

        # Prepare data from incorrect predictions
        for item in self.incorrect_predictions:
            image_data = load_img(item["image_path"], target_size=(224, 224))
            image_data = img_to_array(image_data) / 255.0  # Normalize
            images.append(image_data)
            labels.append(item["correct_label"])

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Train/test split for validation
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Compile the model before retraining
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Retrain the model
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

        # Ensure the directory for saving the updated model exists
        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)

        # Save the retrained model
        self.model.save(self.save_model_path)
        print(f"Model retrained and saved as '{self.save_model_path}'.")

        # Clear incorrect predictions after retraining
        self.incorrect_predictions = []

# Example usage
if __name__ == "__main__":
    # Initialize the model with the current model file path
   model = FallDetectionModel(model_path="artifacts/training/model.h5")

    # Test prediction and feedback loop
   model.predict_with_feedback("path/to/test_image.jpg")
