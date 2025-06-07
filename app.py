import streamlit as st
import sqlite3
import uuid
import datetime
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from safetensors.torch import load_file
import time

# Custom CSS for colorful UI with Poppins font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    /* Lime-to-teal gradient background */
    .stApp {
        background: linear-gradient(135deg, #76ff03 0%, #26a69a 100%);
        color: #333333;
        font-family: 'Poppins', sans-serif;
    }

    /* Colorful alerts */
    .success-card {
        background: #4caf50;
        color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        animation: slideIn 0.5s ease-out;
        font-size: 18px;
        font-weight: 600;
    }
    .error-card {
        background: #d81b60;
        color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        animation: slideIn 0.5s ease-out;
        font-size: 18px;
        font-weight: 600;
    }

    /* Colorful buttons */
    .stButton > button {
        background: #ff6f61 !important;
        color: #333333 !important; /* Dark gray for Details buttons */
        border: none !important;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #8e44ad !important;
        color: #333333 !important;
        transform: scale(1.02);
    }
    /* Form submit button */
    .stFormSubmitButton button, button[type="submit"] {
        background: #ff6f61 !important;
        color: #ffffff !important; /* White text for Submit button */
        border: none !important;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stFormSubmitButton button:hover, button[type="submit"]:hover {
        background: #8e44ad !important;
        color: #ffffff !important;
        transform: scale(1.02);
    }

    /* Form inputs */
    .stTextInput {
        border-radius: 12px;
        border: 2px solid #ff6f61;
        background: #ffffff;
        padding: 10px;
        font-size: 18px;
        color: #333333;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .stTextArea {
        border-radius: 10px;
        border: 2px solid #ff6f61;
        background: #ffffff;
        padding: 10px;
        font-size: 18px !important;
        color: #333333 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    /* Slider */
    .stSlider {
        background: #fce4ec;
        border-radius: 12px;
        padding: 12px;
    }

    /* DataFrame styles */
    .stDataFrame {
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        font-size: 18px;
    }
    .stDataFrame th {
        background: #ab47bc;
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        padding: 12px;
        text-align: left;
        border: 1px solid #e0e0e0;
    }
    .stDataFrame td {
        padding: 12px;
        text-align: left;
        border: 1px solid #e0e0e0;
        word-wrap: break-word;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stDataFrame tr:nth-child(even) {
        background: #f5f5f5;
    }
    .stDataFrame tr:hover {
        background: #ffebee;
        color: #8e44ad;
    }

    /* Expander styles */
    .stExpander {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 10px 0;
    }
    .stExpanderContent p {
        font-size: 18px;
        color: #333333;
        padding: 10px;
    }

    /* Sidebar styles */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #76ff03 0%, #26a69a 100%);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }
    .sidebar .stExpander {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .stExpanderContent p {
        font-size: 18px;
        color: #333333;
    }

    /* Animations */
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }

    /* Text styles */
    h1 {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 10px;
    }
    h3 {
        font-size: 28px;
        font-weight: 600;
        color: #00695c;
    }
    p {
        font-size: 18px;
        color: #333333;
    }
    .caption {
        font-size: 16px;
        font-style: italic;
        color: #26a69a;
    }
    .subtitle {
        font-size: 20px;
        color: #ffffff;
        text-align: center;
    }
    .sidebar-text {
        font-size: 18px;
        color: #FF0000 !important; /* Black text for high contrast */
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5); /* Subtle white shadow */
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect('reviews.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS reviews
            (id TEXT PRIMARY KEY, product_id TEXT, review_text TEXT, rating INTEGER,
             is_verified INTEGER, review_date TEXT, is_fake INTEGER, confidence REAL)
        ''')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        return False

# Define BertWithMetadata class
class BertWithMetadata(torch.nn.Module):
    def __init__(self, bert_model, metadata_dim=4):
        super(BertWithMetadata, self).__init__()
        self.bert = bert_model
        self.fc = torch.nn.Linear(bert_model.config.hidden_size + metadata_dim, 2)

    def forward(self, input_ids, attention_mask, metadata, labels=None):
        outputs = self.bert.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined = torch.cat((pooled_output, metadata), dim=1)
        logits = self.fc(combined)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits} if loss is not None else logits

# Predict fake review
def predict_fake_review(model, tokenizer, review_text, rating, is_verified):
    try:
        encoding = tokenizer(
            review_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        review_length = len(review_text)
        word_count = len(review_text.split())
        avg_word_length = np.mean([len(word) for word in review_text.split()]) if review_text.split() else 0
        metadata = torch.tensor([[review_length, word_count, avg_word_length, is_verified]], dtype=torch.float)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        metadata = metadata.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                metadata=metadata
            )
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None

# Streamlit app
def main():
    # Sidebar with app description and How to Use
    with st.sidebar:
        st.markdown("<p class='sidebar-text slide-in'>Detect fake product reviews instantly.<br>Powered by AI with a user-friendly interface.</p>", unsafe_allow_html=True)
        with st.expander("How to Use"):
            st.markdown("""
                <p>
                    1. <b>Write a Review</b>: Enter a Product ID, write a review (up to 500 characters), select a rating (1-5 stars), and check "Verified Purchase" if applicable. Click "Submit" to analyze.<br>
                    2. <b>View Reviews</b>: Select a Product ID, choose a sort option (e.g., Date, Rating, Confidence), and view results in the table.<br>
                    3. <b>Check Details</b>: Click the "Details" button in the table to see full review information, including fake detection results.<br>
                    4. <b>Debug</b>: Enable "Show Debug Info" to inspect raw data for troubleshooting.
                </p>
            """, unsafe_allow_html=True)

    # Initialize database
    if not init_db():
        return

    # Load model and tokenizer
    model_path = './'
    alert_placeholder = st.empty()  # Placeholder for auto-hiding alert
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
        bert_model = BertForSequenceClassification(config)
        state_dict = load_file(f'{model_path}/model.safetensors')
        adjusted_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                adjusted_state_dict[key] = value
            elif key.startswith('fc.'):
                adjusted_state_dict[key] = value
            else:
                adjusted_state_dict[key] = value
        model = BertWithMetadata(bert_model, metadata_dim=4)
        model.load_state_dict(adjusted_state_dict, strict=False)
        model.eval()
        # Display success alert for 3 seconds
        with alert_placeholder.container():
            st.markdown("<div class='success-card slide-in'>Model and tokenizer loaded successfully!</div>", unsafe_allow_html=True)
        time.sleep(3)
        alert_placeholder.empty()
    except Exception as e:
        alert_placeholder.markdown(f"<div class='error-card slide-in'>Error loading model: {e}</div>", unsafe_allow_html=True)
        st.info("Ensure './checkpoint-best' contains model.safetensors, vocab.txt, tokenizer_config.json, and special_tokens_map.json")
        return

    st.title("🔍 Fake Review Detector")
    st.markdown("<p class='subtitle slide-in'>Spot fake product reviews with ease!</p>", unsafe_allow_html=True)

    # Submit Review Section
    with st.expander("📝 Write a Review", expanded=True):
        with st.form("review_form", clear_on_submit=True):
            product_id = st.text_input("Product ID", value=str(uuid.uuid4()), help="Unique product identifier")
            review_text = st.text_area("Your Review", max_chars=500, help="Write your review (max 500 characters)")
            char_count = len(review_text)
            st.markdown(f"<p class='caption slide-in'>Characters: {char_count}/500</p>", unsafe_allow_html=True)
            rating = st.slider("Rating (1-5)", min_value=1, max_value=5, value=3)
            is_verified = st.checkbox("Verified Purchase")
            submit_button = st.form_submit_button("Submit")

            if submit_button:
                if review_text and char_count > 0:
                    with st.spinner("Checking review..."):
                        is_fake, confidence = predict_fake_review(model, tokenizer, review_text, rating, is_verified)
                        if is_fake is not None:
                            review_date = datetime.datetime.now().isoformat()
                            conn = sqlite3.connect('reviews.db')
                            c = conn.cursor()
                            review_id = str(uuid.uuid4())
                            c.execute('''
                                INSERT INTO reviews (id, product_id, review_text, rating, is_verified, review_date, is_fake, confidence)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (review_id, product_id, review_text, rating, is_verified, review_date, is_fake, confidence))
                            conn.commit()
                            conn.close()
                            st.markdown(f"""
                                <div class='success-card slide-in'>
                                    <b>Review Submitted!</b><br>
                                    Review ID: {review_id[:8]}...<br>
                                    Fake: {'Yes' if is_fake else 'No'}<br>
                                    Confidence: {confidence * 100:.2f}%
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='error-card slide-in'>Failed to process review.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='error-card slide-in'>Please enter a review.</div>", unsafe_allow_html=True)

    # View Reviews Section
    with st.container():
        st.markdown("<h3 class='slide-in'>🕵️‍♂️ View Reviews</h3>", unsafe_allow_html=True)
        conn = sqlite3.connect('reviews.db')
        product_ids = pd.read_sql_query("SELECT DISTINCT product_id FROM reviews", conn)['product_id'].tolist()
        conn.close()

        search_product_id = st.selectbox("Select Product ID", ["Select a Product"] + product_ids)
        sort_by = st.selectbox("Sort By", ["Date (Newest)", "Date (Oldest)", "Rating (High)", "Rating (Low)", "Confidence (High)", "Confidence (Low)"])

        if search_product_id != "Select a Product":
            conn = sqlite3.connect('reviews.db')
            df = pd.read_sql_query("SELECT * FROM reviews WHERE product_id = ?", conn, params=(search_product_id,))
            conn.close()

            if not df.empty:
                # Convert data types for sorting
                df['review_date'] = pd.to_datetime(df['review_date'])
                df['rating'] = df['rating'].astype(int)
                df['confidence'] = df['confidence'].astype(float)

                # Apply sorting
                if sort_by == "Date (Newest)":
                    df = df.sort_values('review_date', ascending=False)
                elif sort_by == "Date (Oldest)":
                    df = df.sort_values('review_date', ascending=True)
                elif sort_by == "Rating (High)":
                    df = df.sort_values('rating', ascending=False)
                elif sort_by == "Rating (Low)":
                    df = df.sort_values('rating', ascending=True)
                elif sort_by == "Confidence (High)":
                    df = df.sort_values('confidence', ascending=False)
                elif sort_by == "Confidence (Low)":
                    df = df.sort_values('confidence', ascending=True)

                # Limit to 50 reviews
                df = df.head(50)

                # Prepare DataFrame for display
                display_df = pd.DataFrame({
                    'ID': df['id'].str[:8],
                    'Review': df['review_text'].apply(lambda x: x[:50] + ('...' if len(x) > 50 else '')),
                    'Rating': df['rating'].apply(lambda x: '★' * x),
                    'Verified': df['is_verified'].apply(lambda x: 'Yes' if x else 'No'),
                    'Date': df['review_date'].dt.strftime('%Y-%m-%d'),
                    'Fake': df['is_fake'].apply(lambda x: 'Yes' if x else 'No'),
                    'Confidence': df['confidence'].apply(lambda x: f"{x * 100:.2f}%"),
                    'Details': df['id']  # Store full ID for button key
                })

                # Debug: Display raw DataFrame and styles
                if st.checkbox("Show Debug Info"):
                    st.write("Raw DataFrame:", df)
                    st.write("Display DataFrame:", display_df)
                    st.write("DataFrame Types:", df.dtypes)

                # Display DataFrame
                st.dataframe(
                    display_df[['ID', 'Review', 'Rating', 'Verified', 'Date', 'Fake', 'Confidence']],
                    use_container_width=True,
                    height=400,
                    column_config={
                        "ID": st.column_config.TextColumn(width=100),
                        "Review": st.column_config.TextColumn(width=300),
                        "Rating": st.column_config.TextColumn(width=100),
                        "Verified": st.column_config.TextColumn(width=100),
                        "Date": st.column_config.TextColumn(width=100),
                        "Fake": st.column_config.TextColumn(width=100),
                        "Confidence": st.column_config.TextColumn(width=100)
                    }
                )

                # Display details with expanders
                for idx, row in display_df.iterrows():
                    if st.button("Details", key=f"details_{row['Details']}"):
                        with st.expander(f"Review Details for {row['ID']}",expanded=True):
                            st.markdown("""
                                <div style='background: linear-gradient(90deg, #ff6f61 0%, #8e44ad 100%); color: #ffffff; font-size: 24px; font-weight: 700; padding: 12px; border-radius: 12px 12px 0 0; text-align: center;'>
                                    Review Details
                                </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                                <p><b>ID:</b> {row['Details']}</p>
                                <p><b>Product ID:</b> {df.loc[idx, 'product_id']}</p>
                                <p><b>Review:</b> {df.loc[idx, 'review_text']}</p>
                                <p><b>Rating:</b> {row['Rating']}</p>
                                <p><b>Verified:</b> {row['Verified']}</p>
                                <p><b>Date:</b> {df.loc[idx, 'review_date']}</p>
                                <p><b>Fake:</b> {row['Fake']} ({row['Confidence']})</p>
                            """, unsafe_allow_html=True)

            else:
                st.markdown("<div class='error-card slide-in'>No reviews found for this product.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
    
    
# Footer
st.markdown("---")
st.markdown("**Designed and Developed by Khushi Raghuvanshi**")