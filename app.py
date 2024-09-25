import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize PorterStemmer
ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer1.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))

# Streamlit app with custom styles
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.markdown("""
    <style>
    body {
        background-color: #2c2c2c;  /* Darker background */
        color: #ffffff;  /* Default text color */
    }
    h1 {
        color: black;  /* Title in black */
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: #3a3a3a;  /* Sidebar background */
    }
    textarea {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        width: 100%;
        background-color: #444;  /* Dark textarea */
        color: #ffffff;  /* Text color in textarea */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
    }
    textarea:hover {
        border-color: #45a049;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📩 Email/SMS Spam Classifier")
st.markdown("This application classifies messages as Spam or Not Spam. Enter the message you want to analyze below:")

# Input text area
input_sms = st.text_area("Enter the message", height=150, placeholder="Write your message here...")

# Prediction button
if st.button('Predict'):
    if input_sms:
        # 1. Preprocess the input
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize the preprocessed text
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict using the trained model
        result = model.predict(vector_input)[0]
        
        # 4. Display the result
        if result == 1:
            st.markdown('<p class="result" style="color: red;">🚫This message is Spam</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result" style="color: green;">✅ This message is Not Spam</p>', unsafe_allow_html=True)
        
        # Display original message and prediction result
        st.markdown(f"**Original Message:** {input_sms}")
        st.markdown(f"**Prediction Result:** {'Spam' if result == 1 else 'Not Spam'}")
    else:
        st.warning("Please enter a message to analyze.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
