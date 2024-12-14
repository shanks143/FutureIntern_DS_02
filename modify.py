# Importing necessary libraries for enhanced model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("C:/Users/sanke/OneDrive/Desktop/email_classification.csv")
data.head()

# Dataset overview
print("Dataset Shape:", data.shape)
print(data['label'].value_counts())

# Balancing dataset by downsampling
ham_msg = data[data['label'] == 'ham']
spam_msg = data[data['label'] == 'spam']

ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)
balanced_data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)

# Plotting balanced dataset
sns.countplot(x='label', data=balanced_data)
plt.title('Balanced Dataset Distribution')
plt.show()

# Text Preprocessing
def preprocess_text(text):
    # Remove "Subject" and punctuations
    text = text.replace('Subject', '').translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.lower().split() if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(lemmatized)

balanced_data['cleaned_email'] = balanced_data['email'].apply(preprocess_text)

# Visualizing WordClouds for both classes
def plot_wordcloud(data, title):
    text = ' '.join(data)
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400).generate(text)
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=15)
    plt.show()

plot_wordcloud(balanced_data[balanced_data['label'] == 'ham']['cleaned_email'], 'Non-Spam Emails')
plot_wordcloud(balanced_data[balanced_data['label'] == 'spam']['cleaned_email'], 'Spam Emails')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(balanced_data['cleaned_email']).toarray()
Y = balanced_data['label'].map({'ham': 0, 'spam': 1})

# Train-test split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model Building
def build_model(input_dim):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=64, input_length=X.shape[1]),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Instantiate model
model = build_model(input_dim=X.shape[1])
model.summary()

# Callbacks for early stopping and learning rate reduction
es = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, factor=0.5, monitor='val_loss')

# Training the model
history = model.fit(
    train_X, train_Y,
    validation_data=(test_X, test_Y),
    epochs=10,
    batch_size=64,
    callbacks=[es, lr]
)

# Evaluate the model
loss, accuracy = model.evaluate(test_X, test_Y)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Classification Report and Confusion Matrix
def evaluate_model(model, X, Y):
    predictions = (model.predict(X) > 0.5).astype(int).ravel()
    print("\nClassification Report:\n", classification_report(Y, predictions))
    cm = confusion_matrix(Y, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

evaluate_model(model, test_X, test_Y)

# Plot Training History
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
