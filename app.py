import pandas as pd
import nltk
from flask import Flask, request, render_template
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask
app = Flask(__name__)

# Load dataset
file_path = r"D:\Skin_Problem_Solver\skincare_products.xlsx"
df_original = pd.read_excel(file_path)  # keep original for display

# Copy for processing
df = df_original.copy()

# Tokenization + stemming
stemmer = PorterStemmer()
nltk.download('punkt')

def tokenization(txt):
    tokens = nltk.word_tokenize(str(txt).lower())
    stems = [stemmer.stem(w) for w in tokens]
    return " ".join(stems)

# Preprocess for similarity (use separate tokenized columns)
df['skin_concern_tok'] = df['skin_concern'].apply(tokenization)
df['category_tok'] = df['category'].apply(tokenization)
df['concern_cate'] = df['skin_concern_tok'] + " " + df['category_tok']

# TF-IDF + cosine similarity (fit once)
vectorizer = TfidfVectorizer(tokenizer=tokenization, token_pattern=None)
tfidf_matrix = vectorizer.fit_transform(df['concern_cate'])

def cosine_sim(txt1, txt2):
    obj_tfidf = TfidfVectorizer(tokenizer=tokenization, token_pattern=None)
    tfidfmatrix = obj_tfidf.fit_transform([txt1, txt2])
    similarity = cosine_similarity(tfidfmatrix)[0][1]
    return similarity

# Recommendation function
def recommation(query):
    tokenized_query = tokenization(query)
    # compute similarity on tokenized text
    df['similarity'] = df['concern_cate'].apply(lambda x: cosine_sim(tokenized_query, x))
    # pick top 3 indices
    top_idx = df.sort_values(by=['similarity'], ascending=False).head(3).index
    # return original (non-tokenized) columns for display
    final_df = df_original.loc[top_idx, 
        ['product_name', 'skin_concern', 'category', 'description', 'image_url']]
    return final_df.to_dict(orient='records')

# Routes
@app.route('/')
def index():
    # Show original concerns (not tokenized) in dropdown
    skin_concerns = df_original['skin_concern'].unique()
    return render_template("index.html", skin_concern=skin_concerns, df=[])

@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['skin_concern']
    final_df = recommation(query)
    skin_concerns = df_original['skin_concern'].unique()
    return render_template('index.html', skin_concern=skin_concerns, df=final_df)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
