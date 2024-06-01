import pandas as pd
import streamlit as st
from xgboost.sklearn import XGBClassifier
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import textstat
import spacy
from googleapiclient import discovery
import json
from textblob import TextBlob
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
import os



# Load environment variables from .env file
load_dotenv()

nlp = spacy.load("en_core_web_sm")






import torch.nn as nn
from transformers import RobertaModel
class RobertaWithFeatures(nn.Module):
    def __init__(self, num_features, num_labels):
        super(RobertaWithFeatures, self).__init__()
        # Load the pretrained RoBERTa base model
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # Define a linear layer to process numerical features
        # Assuming roberta-base, the hidden size is 768
        self.feature_processor = nn.Linear(num_features, 768)
        # Final classifier that takes the concatenated output of text + numerical features
        self.classifier = nn.Linear(768 * 2, num_labels)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         print(outputs)
        sequence_output = outputs.pooler_output
        
    
    # Add a debug print to check feature shape
    
        features_processed = self.feature_processor(features)  # features should be [batch_size, 1]
        features_processed = features_processed.squeeze(1)
        #print("Sequence output shape:", sequence_output.shape)
        #print("Features processed shape:", features_processed.shape)
    
        combined_features = torch.cat((sequence_output, features_processed), dim=1)
        #print(combined_features.shape)
        logits = self.classifier(combined_features)
        return logits



# cache the model to optimize performance
@st.cache_resource
def load_roberta():
    rob_model = RobertaWithFeatures(num_features=8, num_labels=2)
    rob_model.load_state_dict(torch.load('./roberta_with_features_v1.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rob_model.to(device)
    return rob_model, device


@st.cache_resource
def load_xgboost():
    xgb_model = XGBClassifier()  # Initialize the model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    xgb_model.load_model('xgb_constructive_model_v3.json') 
    return xgb_model, sbert_model

@st.cache_resource
def load_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return tokenizer


tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Score Descriptions", "XGBoost Model Card", "SoCial-ConStruct-RoBerta Model Card"])


def clean_text(text, remove_numbers=False):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove or retain numbers based on the flag
    if remove_numbers:
        text = re.sub(r'\d+', '', text)  # Removes numbers

    # Remove special characters, retaining only letters, basic punctuation, and spaces
    text = re.sub(r'[^a-zA-Z\s.!?]', '', text)

    # Remove newline and carriage return characters
    text = text.replace('\n', ' ').replace('\r', ' ')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove specific unwanted characters and general non-alphanumeric characters except spaces
    text = re.sub(r'\¬†', '', text)  # Targeting specific artifact


    return text

@st.cache_data
def get_readability(text):
    readability = textstat.flesch_kincaid_grade(text)
    return readability

@st.cache_data
def count_pos(text, pos_tags=['DET', 'ADJ', 'NOUN', 'VERB', 'ADP']):
    # Initialize dictionary with desired POS tags set to zero
    pos_dictionary = {tag: 0 for tag in pos_tags}
    
    # Process the text
    doc = nlp(text)
    for token in doc:
        if token.pos_ in pos_dictionary:
            pos_dictionary[token.pos_] = pos_dictionary.get(token.pos_, 0) + 1

    return pos_dictionary

@st.cache_data
def subjectivity(text):
    blob = TextBlob(text)
    subjectivity_score = blob.sentiment.subjectivity
    return subjectivity_score

API_KEY = os.getenv('API_KEY')



client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey = API_KEY,
  discoveryServiceUrl = "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery = False,
)



def predict_with_roberta(user_input):

    rob_model, device = load_roberta()

    cleaned_comment = clean_text(user_input)
    # word_count = len(cleaned_comment.strip().split())
    # st.write('word_count:', word_count)
    readability = get_readability(cleaned_comment)
    st.write('readability_score:', readability)
    subj_score = subjectivity(cleaned_comment)
    st.write('subjectivity_score:', subj_score)
    pos_counts = count_pos(cleaned_comment)
    st.write('pos_dictionary:', pos_counts)
    analyze_request = {
    'comment': { 'text': cleaned_comment}, # you can chaneg the text here
    'requestedAttributes': {'UNSUBSTANTIAL': {}},
    'doNotStore': True
    }

    response = client.comments().analyze(body=analyze_request).execute()
    unsubstantial_summary_value = response['attributeScores']['UNSUBSTANTIAL']['summaryScore']['value']
    #incoherent_summary_value = response['attributeScores']['TOXICITY']['summaryScore']['value']
    st.write('unsubstantialness', unsubstantial_summary_value)

    # Tokenization
    tokenizer = load_tokenizer()
    encodings = tokenizer(cleaned_comment, truncation=True, padding=True, max_length=512, return_tensors="pt")
    pos_order = ['DET', 'ADJ', 'NOUN', 'VERB', 'ADP']
    pos_features = [pos_counts[tag] for tag in pos_order]
    additional_features = np.array([unsubstantial_summary_value, readability] + pos_features + [subj_score])
    additional_features = torch.tensor(additional_features, dtype=torch.float).unsqueeze(0)
    rob_model.eval()
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    additional_features = additional_features.to(device)

    with torch.no_grad():
        logits = rob_model(input_ids, attention_mask, additional_features)
        probabilitity = F.softmax(logits, dim=1)
        positive_class_probability = probabilitity[:, 1]
        prediction = (positive_class_probability >= 0.5).long()

    return prediction.item()

def predict_with_xgboost(user_input):
    xgb_model, sbert_model = load_xgboost()
    cleaned_comment = clean_text(user_input)
    # word_count = len(cleaned_comment.strip().split())
    # st.write('word_count:', word_count)
    readability = get_readability(cleaned_comment)
    st.write('readability_score:', readability)
    subj_score = subjectivity(cleaned_comment)
    st.write('subjectivity_score:', subj_score)
    pos_counts = count_pos(cleaned_comment)
    st.write('pos_dictionary:', pos_counts)
    analyze_request = {
    'comment': { 'text': cleaned_comment}, # you can chaneg the text here
    'requestedAttributes': {'UNSUBSTANTIAL': {}},
    'doNotStore': True
    }

    response = client.comments().analyze(body=analyze_request).execute()
    unsubstantial_summary_value = response['attributeScores']['UNSUBSTANTIAL']['summaryScore']['value']
    #incoherent_summary_value = response['attributeScores']['TOXICITY']['summaryScore']['value']
    st.write('unsubstantialness', unsubstantial_summary_value)
    #st.write('incoherent', incoherent_summary_value)
    user_embeddings = sbert_model.encode([user_input])
    st.write('user_embeddings:', user_embeddings.shape)

    pos_order = ['DET', 'ADJ', 'NOUN', 'VERB', 'ADP']
    pos_features = [pos_counts[tag] for tag in pos_order]

        # word_count_array = np.array([word_count]).reshape(-1,1)
        # st.write('word_count_array', word_count_array.shape)

    sub_count_array = np.array([subj_score]).reshape(-1,1)
    st.write('subjective_array', sub_count_array.shape)


    pos_count_array = np.array(pos_features).reshape(-1,1)
    st.write('pos_array', pos_count_array.shape)

    read_count_array = np.array([readability]).reshape(-1,1)
    st.write('readability_array', read_count_array.shape)

    un_count_array = np.array([unsubstantial_summary_value]).reshape(-1,1)
    st.write('unsubstantial_array', un_count_array.shape)
    user_embeddings_flat = user_embeddings.flatten()
        #word_count_flat = word_count_array.flatten()
    pos_count_flat = pos_count_array.flatten()
    read_count_flat = read_count_array.flatten()
    un_count_flat = un_count_array.flatten()
    sub_count_flat = sub_count_array.flatten()
        
    input_features = np.concatenate([
    user_embeddings_flat,
    un_count_flat,  # List to array if not already
    read_count_flat,
    pos_count_flat, 
    sub_count_flat])  # Assuming it's already a numpy array

    final_features = input_features.reshape(1, -1)
    st.write('input', final_features.shape)
    prediction = xgb_model.predict(final_features)
    return prediction







with tab1:

    st.image("construct.png", width=400) 
    st.title('Reddit Constructiveness Comment Classifier')

    st.markdown("""
    Welcome to the Reddit Constructiveness Comment Classifier! This tool analyzes comments from Reddit to determine if they are constructive.

    **Features include:**
    - **Machine Learning Analysis:** Utilizes advanced NLP techniques to understand and classify comments.
    - **User-Friendly Interface:** Simply paste your comment in the text box and get instant feedback.
    - **Insights into Discourse Quality:** Ideal for moderators and social media analysts.

    **Please enter a comment below to get started:**
    """)

    model_choice = st.selectbox(
        'Choose a model to use for prediction:',
        ('XGBoost Classifier', 'SoCial-ConStruct-RoBERTa Model')
    )

    if model_choice == 'SoCial-ConStruct-RoBERTa Model':
        st.markdown("""
        **You have selected the SoCial-ConStruct-RoBERTa Model.**
        - **High Precision:** This model is tuned for higher precision, making it more stringent in identifying constructive comments. It's recommended if you're looking for strict criteria in comment constructiveness, useful in contexts where the quality of discourse is critical.
        """)
    elif model_choice == 'XGBoost Classifier':
        st.markdown("""
        **You have selected the XGBoost Classifier.**
        - **Balanced Approach:** This model offers a balance between precision and recall, making it versatile for general purposes. It's recommended for everyday moderation tasks where a balanced perspective on constructiveness is needed.
        """)


    user_input = st.text_area("")

    if st.button('Predict'):
        # Process the input and make predictions based on the selected model
        if model_choice == 'XGBoost Classifier':
            prediction = predict_with_xgboost(user_input)
        else:
            prediction = predict_with_roberta(user_input)

    
    # Convert input features to the appropriate format for prediction
        
        if prediction == 0:
            st.write('Prediction: Non-constructive')
        elif prediction == 1:
            st.write('Prediction: Constructive')


with tab2:
    st.header("Score Descriptions")
    st.markdown("""
    - **Readability Score**: Measures the complexity of the text. Lower scores indicate simpler text, while higher scores indicate more complex text suitable for higher education levels.
    - **Subjectivity Score**: Measures the amount of personal opinion and subjective information in the text. A higher score means more subjectivity.
    - **Part-of-Speech Counts**: Reflects the usage of various parts of speech in the text, such as nouns, verbs, adjectives, etc., which can indicate the nature of the language used.
    - **Unsubstantial Score**: Generated by the Google Perspective API, this score measures how 'light' or unsubstantial the content might be perceived, with higher scores indicating less substantial content.
    """)

    st.header("Understanding SBERT Embeddings")
    st.markdown("""
    - **SBERT Embeddings**: Sentence-BERT (SBERT) is a modification of the pretrained BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared directly with cosine similarity. In our application:
    - **Text Processing**: When you enter text, it is first cleaned and processed. This cleaned text is then fed into the SBERT model.
    - **Embedding Extraction**: SBERT generates a fixed-sized vector (embedding) for each sentence, capturing the essence of its meaning. These embeddings are dense with floating-point numbers, unlike traditional sparse representations like one-hot encoding.
    - **Utility**: The embeddings are used as part of the input features for our machine learning model, specifically the XGBoost classifier. The dense numerical nature of these embeddings allows the model to effectively learn complex patterns that relate text semantics to constructiveness.
    - **Advantages**: Using SBERT allows us to leverage deep learning models trained on large datasets with vast general knowledge. This helps in understanding nuanced differences in text beyond what traditional text features might capture.
    """)    

with tab3:
    st.header("XGBoost Model Information")
    st.markdown("""
    - **SBERT Embeddings**: Our model utilizes Sentence-BERT (SBERT), a state-of-the-art sentence embedding method, to convert text inputs into numerical data that captures semantic meaning. These embeddings are then used as features in our XGBoost classifier.
    - **XGBoost Classifier**: A powerful, efficient machine learning algorithm that handles a range of complex nonlinear classification tasks. It uses these embeddings along with other extracted features to predict whether a comment is constructive. Precision is important since a comment being incorrectly classified as constructive could be harmful to the discussion.
    - **Feature Integration**: The following features are integrated into the model along with SBERT embeddings:
        - **Text-based features**: Such as the readability and subjectivity scores.
        - **Linguistic features**: Including various parts of speech counts.
        - **Contextual scores**: Like the 'unsubstantial' score from external APIs.
    """)

    f1_score = 0.89
    accuracy_score = 0.851
    precision_score = 0.892
    recall_score = 0.888

    st.header("XGBoost Model Performance Metrics")
    st.markdown("""
    - **F1 Score**: `{}`
    - **Accuracy Score**: `{}`
    - **Recall Score**: `{}`
    - **Precision Score**: `{}`
    """.format(f1_score, accuracy_score, recall_score, precision_score))

    st.subheader("ROC AUC Curve")
    st.image("auc.png", caption='ROC AUC Curve')

    st.subheader("Confusion Matrix")
    st.image("confusion.png", caption='Confusion Matrix')



with tab4:
    st.header("SoCial-ConStruct-RoBerta Model Information")
    st.markdown("""
    - RoBerta has a higher precision rate so it has a higher criteria for constructiveness as it minimizes false positives
    """)

    f1_score = 0.886
    accuracy_score = 0.848
    precision_score = 0.90
    recall_score = 0.87

    st.header("RoBerta Model Performance Metrics")
    st.markdown("""
    - **F1 Score**: `{}`
    - **Accuracy Score**: `{}`
    - **Recall Score**: `{}`
    - **Precision Score**: `{}`
    """.format(f1_score, accuracy_score, recall_score, precision_score))

    st.subheader("ROC AUC Curve")
    st.image("robertauc.png", caption='ROC AUC Curve')

    st.subheader("Confusion Matrix")
    st.image("roberta.png", caption='Confusion Matrix')






