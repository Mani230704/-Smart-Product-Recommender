import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from textblob import TextBlob
from typing import List, Dict, Any
import logging
import time
import uuid
import os
from nltk.corpus import wordnet
import nltk
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import re

# Download NLTK data
nltk.download('wordnet', quiet=True)

# Set up logging for production
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Cache embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(Config.EMBEDDING_MODEL)

# Cache Pinecone index
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key="pcsk_6872y7_Qf4tNLr5T2RYiy8ZiBjTP86rQx6fnwN83udReSZJkpansZpUmkm9bqNBTbw8Mma")
    index_name = Config.VECTOR_DB_NAME
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension of all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

# Configuration class
class Config:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_DB_NAME = "products"
    TOP_K = 6
    AUTHENTICITY_THRESHOLD = 0.4
    CATEGORY_WEIGHTS = {'Electronics': 1.0, 'Wearables': 0.8, 'Accessories': 0.6}
    ITEMS_PER_PAGE = 3
    DEFAULT_IMAGE_URL = "https://via.placeholder.com/400x300?text=Product+Image"

# Evaluation Metrics Module
class EvaluationMetrics:
    def __init__(self):
        self.retrieval_count = 0
        self.total_latency = 0
        self.feedback_scores = []

    def log_retrieval(self, latency: float):
        """Log retrieval metrics."""
        self.retrieval_count += 1
        self.total_latency += latency

    def log_feedback(self, rating: int):
        """Log feedback for accuracy."""
        self.feedback_scores.append(rating)

    def get_metrics(self) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        try:
            avg_latency = self.total_latency / self.retrieval_count if self.retrieval_count > 0 else 0
            avg_feedback = np.mean(self.feedback_scores) if self.feedback_scores else 0
            return {
                'avg_latency': avg_latency,
                'retrieval_count': self.retrieval_count,
                'avg_feedback_score': avg_feedback
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'avg_latency': 0, 'retrieval_count': 0, 'avg_feedback_score': 0}

# Data Loader Module
class DataLoader:
    @staticmethod
    @st.cache_data
    def load_initial_product_data(_data_version: str = "initial") -> pd.DataFrame:
        """Load initial product data with specific image URLs."""
        schema = {
            'id': str,
            'name': str,
            'description': str,
            'specifications': str,
            'category': str,
            'price': float,
            'reviews': object,
            'image_url': str
        }
        products = [
            {
                'id': '1',
                'name': 'Wireless Headphones Pro',
                'description': 'Premium wireless headphones with active noise cancellation and 30-hour battery life.',
                'specifications': 'Bluetooth 5.2, 40mm drivers, USB-C, 20Hz-20kHz',
                'category': 'Electronics',
                'price': 129.99,
                'reviews': [
                    'Excellent sound and comfort!',
                    'Best headphones ever!!!',
                    'Great for long flights.'
                ],
                'image_url': 'https://m.media-amazon.com/images/I/619dbMmvFHL.jpg'
            },
            {
                'id': '2',
                'name': 'Fitness Smartwatch',
                'description': 'Advanced smartwatch with heart rate, GPS, and sleep tracking.',
                'specifications': '1.4-inch AMOLED, IP68, 10-day battery',
                'category': 'Wearables',
                'price': 179.99,
                'reviews': [
                    'Accurate tracking, sleek design.',
                    'Must buy, perfect!!!',
                    'Battery lasts forever.'
                ],
                'image_url': 'https://ptron.in/cdn/shop/products/1_0a5ed607-d327-4630-b0ca-c1e60d7c9ceb.jpg?v=1653567794'
            },
            {
                'id': '3',
                'name': '4K OLED TV',
                'description': '65-inch 4K OLED TV with HDR10 and smart streaming.',
                'specifications': '4K UHD, 120Hz, HDMI 2.1, Dolby Vision',
                'category': 'Electronics',
                'price': 799.99,
                'reviews': [
                    'Vivid colors, amazing clarity!',
                    'Unreal deal, buy now!!!',
                    'Needs better sound.'
                ],
                'image_url': 'https://www.lg.com/content/dam/channel/wcms/in/images/tvs/oled42c3psa_atr_eail_in_c/gallery/OLED42C3PSA-DZ-03.jpg'
            },
            {
                'id': '4',
                'name': 'Wireless Charger',
                'description': 'Fast wireless charger compatible with multiple devices.',
                'specifications': '15W charging, Qi-certified, USB-C',
                'category': 'Accessories',
                'price': 29.99,
                'reviews': [
                    'Charges quickly, very reliable.',
                    'Perfect charger, get it now!!!',
                    'Compact and efficient.'
                ],
                'image_url': 'https://m.media-amazon.com/images/I/614+4XNwpML.jpg'
            }
        ]
        df = pd.DataFrame(products).astype(schema)
        return df

    @staticmethod
    def load_product_data(data_version: str = "initial") -> pd.DataFrame:
        """Load product data, returning session state if available."""
        if 'products_df' in st.session_state and not st.session_state.products_df.empty:
            logger.info(f"Returning updated products_df from session state with {len(st.session_state.products_df)} products")
            return st.session_state.products_df
        df = DataLoader.load_initial_product_data(data_version)
        st.session_state.products_df = df
        logger.info(f"Initialized products_df with {len(df)} initial products")
        return df

# Review Authenticity Module
class ReviewAuthenticity:
    @staticmethod
    def score_review(review: str) -> float:
        """Score review authenticity using sentiment and heuristics."""
        try:
            analysis = TextBlob(review)
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            length = len(review)
            exclamation_count = review.count('!')
            
            if length < 15 or subjectivity > 0.85 or exclamation_count > 2:
                return 0.25
            if 'buy now' in review.lower() or 'perfect' in review.lower():
                return 0.3
            return max(0.5, 1.0 - abs(polarity) * 0.4)
        except Exception as e:
            logger.error(f"Error scoring review: {e}")
            return 0.5

    @staticmethod
    def filter_reviews(reviews: List[str], threshold: float = Config.AUTHENTICITY_THRESHOLD) -> List[str]:
        """Filter out inauthentic reviews."""
        return [r for r in reviews if ReviewAuthenticity.score_review(r) >= threshold]

# Query Expansion Module
class QueryExpander:
    @staticmethod
    def expand_query(query: str) -> str:
        """Expand query with synonyms using WordNet."""
        try:
            words = query.split()
            expanded = []
            for word in words:
                expanded.append(word)
                synonyms = set()
                for syn in wordnet.synsets(word)[:2]:
                    for lemma in syn.lemmas()[:2]:
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != word:
                            synonyms.add(synonym)
                expanded.extend(list(synonyms)[:2])
            return ' '.join(expanded)
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query

# Vector Database Module
class VectorDB:
    def __init__(self):
        self.index = init_pinecone()
        self.embedding_function = load_embedding_model()

    def clear_collection(self) -> None:
        """Clear the vector database collection."""
        try:
            # Delete all vectors in the index
            self.index.delete(delete_all=True)
            logger.info("Cleared Pinecone index")
        except Exception as e:
            logger.error(f"Error clearing Pinecone index: {e}")

    def add_products(self, df: pd.DataFrame) -> None:
        """Process and store product data in Pinecone."""
        vectors = []
        for _, row in df.iterrows():
            combined_text = f"{row['name']}: {row['description']} {row['specifications']}"
            filtered_reviews = ReviewAuthenticity.filter_reviews(row['reviews'])
            review_sentiments = [TextBlob(review).sentiment.polarity for review in filtered_reviews]
            avg_sentiment = np.mean(review_sentiments) if review_sentiments else 0
            embedding = self.embedding_function.encode(combined_text).tolist()
            metadata = {
                'id': row['id'],
                'name': row['name'],
                'category': row['category'],
                'price': float(row['price']),
                'avg_sentiment': float(avg_sentiment),
                'review_count': len(filtered_reviews)
            }
            vectors.append({
                'id': row['id'],
                'values': embedding,
                'metadata': metadata
            })
        if vectors:
            self.index.upsert(vectors=vectors)
        logger.info("Products successfully added to Pinecone index")

# Recommendation Engine
class RecommendationEngine:
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        self.query_expander = QueryExpander()

    @st.cache_data
    def _get_recommendations_cached(_self, query: str, preferences: List[str], min_price: float, max_price: float, sort_by: str, top_k: int, _data_version: str) -> List[Dict[str, Any]]:
        """Cached recommendation function."""
        return _self.get_recommendations(query, preferences, min_price, max_price, sort_by, top_k)

    def get_recommendations(self, query: str, preferences: List[str], min_price: float, max_price: float, sort_by: str, top_k: int = Config.TOP_K) -> tuple[List[Dict[str, Any]], float]:
        """Generate personalized recommendations."""
        try:
            start_time = time.time()
            expanded_query = self.query_expander.expand_query(query)
            query_embedding = self.vector_db.embedding_function.encode(expanded_query).tolist()
            filter_conditions = {}
            if preferences:
                filter_conditions['category'] = {'$in': preferences}
            filter_conditions['price'] = {'$gte': min_price, '$lte': max_price}
            
            results = self.vector_db.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_conditions,
                include_metadata=True
            )

            ranked_results = []
            for match in results['matches']:
                meta = match['metadata']
                dist = match['score']  # Pinecone uses cosine similarity (higher is better)
                category_weight = Config.CATEGORY_WEIGHTS.get(meta['category'], 1.0)
                review_weight = meta['review_count'] / (meta['review_count'] + 2)
                score = dist * (1 + meta['avg_sentiment']) * category_weight * review_weight
                ranked_results.append({
                    'id': meta['id'],
                    'name': meta['name'],
                    'description': meta.get('description', '').split(' ')[:-1],  # Fallback if description not stored
                    'price': meta['price'],
                    'category': meta['category'],
                    'score': score,
                    'review_count': meta['review_count'],
                    'avg_sentiment': meta['avg_sentiment']
                })

            if sort_by == "price":
                ranked_results.sort(key=lambda x: x['price'])
            elif sort_by == "sentiment":
                ranked_results.sort(key=lambda x: x['avg_sentiment'], reverse=True)
            else:
                ranked_results.sort(key=lambda x: x['score'], reverse=True)

            latency = time.time() - start_time
            logger.info(f"Recommendations generated in {latency:.2f} seconds")
            return ranked_results[:top_k], latency
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [], 0.0

    def get_cross_category_recommendations(self, product_id: str, min_price: float, max_price: float, top_k: int = Config.TOP_K) -> List[Dict[str, Any]]:
        """Generate cross-category recommendations."""
        try:
            product = self.vector_db.index.fetch([product_id])
            if not product['vectors']:
                return []
            
            query_embedding = product['vectors'][product_id]['values']
            results = self.vector_db.index.query(
                vector=query_embedding,
                top_k=top_k + 1,
                filter={"price": {"$gte": min_price, "$lte": max_price}},
                include_metadata=True
            )
            
            ranked_results = []
            for match in results['matches']:
                if match['id'] != product_id:
                    meta = match['metadata']
                    score = match['score']
                    ranked_results.append({
                        'id': meta['id'],
                        'name': meta['name'],
                        'description': meta.get('description', '').split(' ')[:-1],
                        'price': meta['price'],
                        'category': meta['category'],
                        'score': score
                    })
            
            return sorted(ranked_results, key=lambda x: x['score'], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Error generating cross-category recommendations: {e}")
            return []

# Comparison Module
class ProductComparator:
    @staticmethod
    def compare_products(product_ids: List[str], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Compare selected products."""
        try:
            comparison = []
            for pid in product_ids:
                product_rows = df[df['id'] == pid]
                if product_rows.empty:
                    logger.warning(f"Product ID {pid} not found in DataFrame")
                    continue
                product = product_rows.iloc[0]
                filtered_reviews = ReviewAuthenticity.filter_reviews(product['reviews'])
                avg_sentiment = np.mean([TextBlob(review).sentiment.polarity for review in filtered_reviews]) if filtered_reviews else 0
                comparison.append({
                    'id': product['id'],
                    'name': product['name'],
                    'price': product['price'],
                    'category': product['category'],
                    'description': product['description'],
                    'specifications': product['specifications'],
                    'avg_sentiment': avg_sentiment,
                    'review_count': len(filtered_reviews),
                    'image_url': product['image_url']
                })
            return comparison
        except Exception as e:
            logger.error(f"Error comparing products: {e}")
            return []

# Visualization Module
class Visualizer:
    @staticmethod
    def plot_comparison(comparison: List[Dict[str, Any]]) -> None:
        """Plot comparison bar chart."""
        try:
            df = pd.DataFrame({
                'Name': [p['name'] for p in comparison],
                'Price': [p['price'] for p in comparison],
                'Sentiment': [p['avg_sentiment'] for p in comparison],
                'Reviews': [p['review_count'] for p in comparison]
            })
            fig = px.bar(
                df,
                x='Name',
                y=['Price', 'Sentiment', 'Reviews'],
                barmode='group',
                title="Product Comparison",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_layout(
                legend_title_text='Metrics',
                title_font=dict(size=20, family="Arial, sans-serif"),
                font=dict(size=14, family="Arial, sans-serif"),
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error plotting comparison: {e}")

    @staticmethod
    def plot_radar_chart(comparison: List[Dict[str, Any]]) -> None:
        """Plot radar chart for feature comparison."""
        try:
            categories = ['Price', 'Sentiment', 'Reviews']
            fig = go.Figure()
            for product in comparison:
                values = [
                    1000 - product['price'],
                    product['avg_sentiment'] * 100,
                    product['review_count'] * 10
                ]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=product['name'],
                    line=dict(width=2)
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1000]),
                    angularaxis=dict(showline=True)
                ),
                showlegend=True,
                title="Feature Comparison",
                title_font=dict(size=20, family="Arial, sans-serif"),
                font=dict(size=14, family="Arial, sans-serif"),
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error plotting radar chart: {e}")

# User Behavior Tracker
class UserBehaviorTracker:
    def __init__(self):
        self.user_history = st.session_state.get('user_history', {})
        self.feedback = st.session_state.get('feedback', {})
        self.favorites = st.session_state.get('favorites', set())

    def log_interaction(self, user_id: str, action: str, product_id: str) -> None:
        """Log user interactions."""
        try:
            if user_id not in self.user_history:
                self.user_history[user_id] = []
            self.user_history[user_id].append({
                'action': action,
                'product_id': product_id,
                'timestamp': time.time()
            })
            st.session_state.user_history = self.user_history
            logger.info(f"Logged {action} for user {user_id} on product {product_id}")
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")

    def log_feedback(self, user_id: str, product_id: str, rating: int) -> None:
        """Log user feedback."""
        try:
            if user_id not in self.feedback:
                self.feedback[user_id] = {}
            self.feedback[user_id][product_id] = rating
            st.session_state.feedback = self.feedback
            logger.info(f"Logged feedback {rating} for user {user_id} on product {product_id}")
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")

    def toggle_favorite(self, user_id: str, product_id: str) -> bool:
        """Toggle product in favorites."""
        try:
            if product_id in self.favorites:
                self.favorites.remove(product_id)
                st.session_state.favorites = self.favorites
                logger.info(f"Removed {product_id} from favorites for user {user_id}")
                return False
            else:
                self.favorites.add(product_id)
                st.session_state.favorites = self.favorites
                logger.info(f"Added {product_id} to favorites for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error toggling favorite: {e}")
            return False

    def get_user_preferences(self, user_id: str, df: pd.DataFrame) -> List[str]:
        """Infer user preferences."""
        try:
            categories = set()
            if user_id in self.user_history:
                product_ids = [entry['product_id'] for entry in self.user_history[user_id]]
                categories.update(df[df['id'].isin(product_ids)]['category'].unique())
            
            if user_id in self.feedback:
                high_rated = [pid for pid, rating in self.feedback[user_id].items() if rating >= 4]
                categories.update(df[df['id'].isin(high_rated)]['category'].unique())
            
            if self.favorites:
                categories.update(df[df['id'].isin(self.favorites)]['category'].unique())
            
            return list(categories)
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return []

# Product Addition Module
class ProductAdder:
    @staticmethod
    def add_product(name: str, description: str, specifications: str, category: str, price: float, reviews: List[str], image_url: str, vector_db: VectorDB) -> bool:
        """Add a new product to the database."""
        try:
            # Validate inputs
            if not name.strip() or not description.strip() or not specifications.strip() or not category.strip():
                st.error("All text fields must be non-empty.")
                return False
            if price < 0:
                st.error("Price must be non-negative.")
                return False
            # Use default image if image_url is invalid or empty
            final_image_url = image_url.strip() if image_url.strip() and re.match(r'^https?://', image_url.strip()) else Config.DEFAULT_IMAGE_URL

            # Initialize products_df if it doesn't exist
            if 'products_df' not in st.session_state or st.session_state.products_df.empty:
                schema = {
                    'id': str,
                    'name': str,
                    'description': str,
                    'specifications': str,
                    'category': str,
                    'price': float,
                    'reviews': object,
                    'image_url': str
                }
                st.session_state.products_df = pd.DataFrame(columns=schema.keys()).astype(schema)

            # Create new product DataFrame
            new_id = str(uuid.uuid4())
            new_product = {
                'id': new_id,
                'name': name.strip(),
                'description': description.strip(),
                'specifications': specifications.strip(),
                'category': category.strip(),
                'price': float(price),
                'reviews': [r.strip() for r in reviews if r.strip()],
                'image_url': final_image_url
            }
            new_df = pd.DataFrame([new_product])

            # Concatenate with existing DataFrame
            df = st.session_state.products_df
            st.session_state.products_df = pd.concat([df, new_df], ignore_index=True, sort=False)
            logger.info(f"Updated products_df with new product: {name}, ID: {new_id}")

            # Clear and rebuild Pinecone index
            vector_db.clear_collection()
            vector_db.add_products(st.session_state.products_df)
            st.success(f"Product '{name}' added successfully!")
            logger.info(f"Added product {name} with ID {new_id} to Pinecone index")
            return True
        except Exception as e:
            st.error(f"Error adding product: {e}")
            logger.error(f"Error adding product: {e}")
            return False

# Streamlit UI
def main():
    try:
        st.set_page_config(page_title="Smart Product Recommender", layout="wide")
        st.markdown("""
            <style>
                .main { background: linear-gradient(to bottom, #f6f9fc, #e8ecef); }
                .stButton>button { 
                    background: linear-gradient(135deg, #007bff, #00d4ff);
                    color: white; 
                    border-radius: 12px; 
                    padding: 12px 24px; 
                    font-size: 16px; 
                    font-weight: 600;
                    border: none;
                    transition: all 0.3s ease;
                    box-shadow: 0 3px 8px rgba(0,0,0,0.15);
                }
                .stButton>button:hover { 
                    background: linear-gradient(135deg, #00d4ff, #007bff); 
                    transform: translateY(-2px);
                    box-shadow: 0 5px 12px rgba(0,0,0,0.2);
                }
                .card { 
                    background: rgba(255, 255, 255, 0); ; 
                    padding: 20px; 
                    border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                    margin: 10px auto; 
                    width: 100%;
                    max-width: 400px;
                    text-align: left;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                .card:hover { 
                    transform: translateY(-5px); 
                    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
                }
                .stSlider { 
                    background: #f1f3f5; 
                    padding: 15px; 
                    border-radius: 10px; 
                    box-shadow: inset 0 1px 4px rgba(0,0,0,0.1);
                }
                .stTextInput>div>input, .stTextArea>div>textarea { 
                    border: 2px solid #dee2e6; 
                    border-radius: 8px; 
                    padding: 12px; 
                    font-size: 15px;
                    font-family: 'Arial', sans-serif;
                    transition: border-color 0.3s ease, box-shadow 0.3s ease;
                }
                .stTextInput>div>input:focus, .stTextArea>div>textarea:focus { 
                    border-color: #007bff; 
                    box-shadow: 0 0 8px rgba(0,123,255,0.3);
                }
                .favorite-btn { 
                    background: linear-gradient(135deg, #ff6b6b, #ff3b3b); 
                }
                .favorite-btn-selected { 
                    background: linear-gradient(135deg, #2ecc71, #27ae60); 
                }
                .product-image { 
                    width: 100%; 
                    max-width: 250px; 
                    height: auto; 
                    border-radius: 12px; 
                    margin-bottom: 15px;
                    border: 1px solid #e9ecef;
                    object-fit: cover;
                    aspect-ratio: 4/3;
                }
                .sidebar .stSelectbox, .sidebar .stMultiSelect {
                    background: #ffffff;
                    border-radius: 8px;
                    padding: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .stMetric {
                    background: rgba(255, 255, 255, 0); ;
                    border-radius: 12px;
                    padding: 15px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                    font-size: 16px;
                    font-family: 'Arial', sans-serif;
                }
                .stTabs [data-baseweb="tab"] {
                    font-size: 16px;
                    font-weight: 600;
                    padding: 12px 24px;
                    border-radius: 10px;
                    transition: all 0.3s ease;
                    font-family: 'Arial', sans-serif;
                }
                .stTabs [data-baseweb="tab"]:hover {
                    background: #f1f3f5;
                }
                .stTabs [data-baseweb="tab"][aria-selected="true"] {
                    background: linear-gradient(135deg, #007bff, #00d4ff);
                    color: white;
                }
                .card-container {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 20px;
                    width: 100%;
                    margin: 0 auto;
                }
                @media (max-width: 600px) {
                    .card { 
                        padding: 15px; 
                        max-width: 100%;
                    }
                    .stButton>button { 
                        font-size: 14px; 
                        padding: 10px 20px; 
                    }
                    .product-image { 
                        max-width: 200px; 
                    }
                    .stTextInput>div>input, .stTextArea>div>textarea { 
                        font-size: 13px; 
                    }
                    .stMetric {
                        font-size: 14px;
                    }
                    .stTabs [data-baseweb="tab"] {
                        font-size: 14px;
                        padding: 8px 16px;
                    }
                    .card-container {
                        flex-direction: column;
                        align-items: center;
                    }
                }
                @media (min-width: 601px) and (max-width: 1024px) {
                    .card { 
                        padding: 18px; 
                        max-width: 350px;
                    }
                    .product-image { 
                        max-width: 220px; 
                    }
                    .stButton>button { 
                        font-size: 15px; 
                    }
                    .card-container {
                        justify-content: space-around;
                    }
                }
                @media (min-width: 1025px) {
                    .card-container {
                        max-width: 1400px;
                    }
                }
            </style>
        """, unsafe_allow_html=True)

        # Theme toggle
        theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], help="Switch between light and dark themes")
        if theme == "Dark":
            st.markdown("""
                <style>
                    .main { background: linear-gradient(to bottom, #1e1e1e, #343a40); color: #e9ecef; }
                    .card { background: #2c2c2c; color: #e9ecef; }
                    .stTextInput>div>input, .stTextArea>div>textarea { 
                        background: #3a3a3a; 
                        color: #e9ecef; 
                        border-color: #555; 
                    }
                    .stSelectbox>div>select { 
                        background: #3a3a3a; 
                        color: #e9ecef; 
                    }
                    .sidebar .stSelectbox, .sidebar .stMultiSelect {
                        background: #2c2c2c;
                    }
                    .stMetric {
                        background: #2c2c2c;
                        color: #e9ecef;
                    }
                    .stTabs [data-baseweb="tab"][aria-selected="true"] {
                        background: linear-gradient(135deg, #00d4ff, #007bff);
                        color: white;
                    }
                </style>
            """, unsafe_allow_html=True)

        st.title("🛒 Smart Product Recommender")

        # Initialize session state
        if 'user_id' not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        if 'user_history' not in st.session_state:
            st.session_state.user_history = {}
        if 'feedback' not in st.session_state:
            st.session_state.feedback = {}
        if 'favorites' not in st.session_state:
            st.session_state.favorites = set()
        if 'page' not in st.session_state:
            st.session_state.page = 1
        if 'data_version' not in st.session_state:
            st.session_state.data_version = str(uuid.uuid4())
        if 'reset_search' not in st.session_state:
            st.session_state.reset_search = False
        if 'reset_compare' not in st.session_state:
            st.session_state.reset_compare = False

        # Initialize modules
        with st.spinner("Initializing recommendation engine..."):
            df = DataLoader.load_product_data(st.session_state.data_version)
            if df.empty:
                logger.warning("Products DataFrame is empty")
                st.warning("No products available. Please add products in the 'Add Product' tab.")
            vector_db = VectorDB()
            vector_db.add_products(df)
            recommender = RecommendationEngine(vector_db)
            comparator = ProductComparator()
            tracker = UserBehaviorTracker()
            metrics = EvaluationMetrics()
            visualizer = Visualizer()
            product_adder = ProductAdder()

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Search & Recommend", "Compare Products", "Add Product", "Favorites", "Performance Metrics"])

        with tab1:
            st.header("Search for Products")
            with st.form("search_form"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    query = st.text_input("Search for products", placeholder="e.g., wireless headphones", key="search_query")
                with col2:
                    sort_by = st.selectbox("Sort by", ["Relevance", "Price", "Sentiment"], key="sort_by")
                col3, col4 = st.columns([1, 1])
                with col3:
                    submitted = st.form_submit_button("Search")
                with col4:
                    reset = st.form_submit_button("Clear Search")

                if reset:
                    st.session_state.reset_search = True
                    st.session_state.search_query = ""
                    st.session_state.sort_by = "Relevance"
                    st.session_state.page = 1
                    st.rerun()

            st.sidebar.header("Filter Preferences")
            categories = df['category'].unique().tolist() if not df.empty else []
            inferred_preferences = tracker.get_user_preferences(st.session_state.user_id, df)
            selected_categories = st.sidebar.multiselect(
                "Select categories",
                categories,
                default=inferred_preferences,
                help="Choose preferred product categories"
            )
            min_price, max_price = st.sidebar.slider(
                "Price Range",
                float(df['price'].min()) if not df.empty else 0.0,
                float(df['price'].max()) if not df.empty else 1000.0,
                (0.0, 1000.0),
                help="Select your budget range"
            )

            if query and submitted or (query and not st.session_state.reset_search):
                with st.spinner("Finding the best products for you..."):
                    recommendations, latency = recommender._get_recommendations_cached(
                        query, selected_categories, min_price, max_price, sort_by, Config.TOP_K, st.session_state.data_version
                    )
                    metrics.log_retrieval(latency)
                
                if recommendations:
                    st.subheader("Recommended Products")
                    st.markdown(f"**Filters Applied**: Categories: {', '.join(selected_categories) or 'All'}, Price: ${min_price:.2f} - ${max_price:.2f}, Sort by: {sort_by}")

                    # Pagination
                    total_pages = (len(recommendations) + Config.ITEMS_PER_PAGE - 1) // Config.ITEMS_PER_PAGE
                    st.session_state.page = st.number_input("Page", min_value=1, max_value=max(total_pages, 1), value=st.session_state.page, step=1)
                    start_idx = (st.session_state.page - 1) * Config.ITEMS_PER_PAGE
                    end_idx = start_idx + Config.ITEMS_PER_PAGE
                    paginated_recs = recommendations[start_idx:end_idx]

                    # Responsive card grid
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    for rec in paginated_recs:
                        product_rows = df[df['id'] == rec['id']]
                        if product_rows.empty:
                            logger.warning(f"Product ID {rec['id']} not found in DataFrame for recommendation")
                            continue
                        product = product_rows.iloc[0]
                        st.markdown(f"""
                            <div class="card">
                                <img src="{product['image_url']}" class="product-image" alt="{rec['name']}">
                                <h3 style="font-size: 18px; margin: 10px 0; text-align: center;">{rec['name']} - ${rec['price']:.2f}</h3>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Category</b>: {rec['category']}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Description</b>: {' '.join(rec['description'])}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Score</b>: {rec['score']:.2f}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Sentiment</b>: {rec['avg_sentiment']:.2f}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Reviews</b>: {rec['review_count']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if st.button("View Details", key=f"view_{rec['id']}"):
                                tracker.log_interaction(st.session_state.user_id, "view", rec['id'])
                        with col2:
                            feedback = st.slider("Rate (1-5)", 1, 5, 3, key=f"feedback_{rec['id']}")
                            if st.button("Submit Feedback", key=f"submit_{rec['id']}"):
                                tracker.log_feedback(st.session_state.user_id, rec['id'], feedback)
                                metrics.log_feedback(feedback)
                                st.success("Feedback submitted!")
                        with col3:
                            is_favorite = rec['id'] in st.session_state.favorites
                            btn_class = "favorite-btn-selected" if is_favorite else "favorite-btn"
                            if st.button(
                                f"{'Remove from' if is_favorite else 'Add to'} Favorites",
                                key=f"fav_{rec['id']}",
                                help="Add or remove from your favorites list"
                            ):
                                is_favorite = tracker.toggle_favorite(st.session_state.user_id, rec['id'])
                                st.rerun()
                        
                        cross_recs = recommender.get_cross_category_recommendations(rec['id'], min_price, max_price, 3)
                        if cross_recs:
                            with st.expander("You might also like...", expanded=False):
                                for cross_rec in cross_recs:
                                    st.write(f"- {cross_rec['name']} ({cross_rec['category']}, ${cross_rec['price']:.2f})")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Download recommendations
                    with st.spinner("Preparing download..."):
                        csv_buffer = io.StringIO()
                        pd.DataFrame(recommendations).to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Download Recommendations as CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_recs"
                        )
                else:
                    st.info("No recommendations found. Try adjusting your search or filters.")

            if st.session_state.reset_search:
                st.session_state.reset_search = False

        with tab2:
            st.header("Compare Products")
            # Filter valid product IDs
            valid_product_ids = df['id'].tolist() if not df.empty else []
            default_selection = [pid for pid in st.session_state.get('compare_select', []) if pid in valid_product_ids] if not st.session_state.reset_compare else []
            product_ids = st.multiselect(
                "Select products to compare",
                valid_product_ids,
                default=default_selection,
                format_func=lambda x: df[df['id'] == x]['name'].iloc[0] if not df[df['id'] == x].empty else x,
                help="Select at least 2 products to compare",
                key="compare_select"
            )
            # Clear reset signal after applying
            if st.session_state.reset_compare:
                st.session_state.reset_compare = False
            
            if product_ids and len(product_ids) >= 2:
                comparison = comparator.compare_products(product_ids, df)
                if not comparison:
                    st.warning("No valid products selected for comparison.")
                else:
                    st.subheader("Product Comparison")
                    
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    for product in comparison:
                        st.markdown(f"""
                            <div class="card">
                                <img src="{product['image_url']}" class="product-image" alt="{product['name']}">
                                <h3 style="font-size: 18px; margin: 10px 0; text-align: center;">{product['name']}</h3>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Price</b>: ${product['price']:.2f}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Category</b>: {product['category']}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Description</b>: {product['description']}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Specs</b>: {product['specifications']}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Sentiment</b>: {product['avg_sentiment']:.2f}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Reviews</b>: {product['review_count']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        if st.button("Select", key=f"select_{product['id']}"):
                            tracker.log_interaction(st.session_state.user_id, "select", product['id'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    visualizer.plot_comparison(comparison)
                    visualizer.plot_radar_chart(comparison)
            elif product_ids:
                st.info("Please select at least 2 products to compare.")

        with tab3:
            st.header("Add New Product")
            with st.form("add_product_form"):
                name = st.text_input("Product Name", placeholder="e.g., Wireless Earbuds")
                description = st.text_area("Description", placeholder="e.g., High-quality earbuds with noise cancellation")
                specifications = st.text_area("Specifications", placeholder="e.g., Bluetooth 5.0, 10-hour battery")
                category = st.selectbox("Category", categories + ["Other"] if categories else ["Other"], help="Select or choose 'Other' for a new category")
                new_category = st.text_input("New Category (if 'Other')", placeholder="Enter new category", disabled=category != "Other")
                price = st.number_input("Price ($)", min_value=0.0, max_value=10000.0, step=0.01)
                reviews = st.text_area("Reviews (one per line)", placeholder="Enter reviews, one per line").splitlines()
                image_url = st.text_input("Image URL (optional)", placeholder="e.g., https://images.unsplash.com/photo-1234567890")
                submitted = st.form_submit_button("Add Product")
                
                if submitted:
                    final_category = new_category if category == "Other" and new_category else category
                    success = product_adder.add_product(name, description, specifications, final_category, price, reviews, image_url, vector_db)
                    if success:
                        st.session_state.data_version = str(uuid.uuid4())
                        st.session_state.reset_compare = True
                        st.rerun()

        with tab4:
            st.header("Your Favorites")
            if st.session_state.favorites:
                favorite_ids = list(st.session_state.favorites)
                favorite_df = df[df['id'].isin(favorite_ids)]
                if favorite_df.empty:
                    st.info("No valid favorites found. Products may have been removed.")
                else:
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    for product in favorite_df.itertuples():
                        st.markdown(f"""
                            <div class="card">
                                <img src="{product.image_url}" class="product-image" alt="{product.name}">
                                <h3 style="font-size: 18px; margin: 10px 0; text-align: center;">{product.name} - ${product.price:.2f}</h3>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Category</b>: {product.category}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Description</b>: {product.description}</p>
                                <p style="font-size: 14px; margin: 5px 0; text-align: center;"><b>Specs</b>: {product.specifications}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        if st.button("Remove from Favorites", key=f"remove_fav_{product.id}"):
                            tracker.toggle_favorite(st.session_state.user_id, product.id)
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No favorites added yet. Add products from the Search & Recommend tab.")

        with tab5:
            st.header("System Performance")
            metrics_data = metrics.get_metrics()
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Latency (s)", f"{metrics_data['avg_latency']:.2f}")
            col2.metric("Total Queries", metrics_data['retrieval_count'])
            col3.metric("Average Feedback Score", f"{metrics_data['avg_feedback_score']:.2f}")
            
            with st.spinner("Preparing download..."):
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Product Database as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_db"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Main loop error: {e}")

if __name__ == "__main__":
    main()


