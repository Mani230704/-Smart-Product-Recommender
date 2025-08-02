# Smart Product Recommender 🎉

Welcome to the **Smart Product Recommender**, a cutting-edge, web-based e-commerce solution built with **Streamlit** and **ChromaDB**. Discover personalized product recommendations, compare options, and manage your favorites with a sleek, user-friendly interface!

---

## 🚀 Setup and Run Instructions

Get started with the Smart Product Recommender in just a few simple steps:

### 1. Clone the Repository
Clone the project to your local machine:
git clone <repository-url>
Replace <repository-url> with your GitHub repository URL.
### 2. Install Dependencies
Navigate to the project directory:
bashcd <repository-folder>
Install the required Python packages:
bashpip install -r requirements.txt
Ensure requirements.txt includes the following dependencies:
textstreamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
sentence-transformers==3.1.1
chromadb==0.5.5
textblob==0.18.0.post0
plotly==5.24.1
nltk==3.8.1
### 3. Run the Application
Launch the Streamlit app:
bashstreamlit run app.py
Open your browser and visit the local URL (e.g., http://localhost:8501) to start exploring!
### 4. Optional Deployment 🌐
Deploy on Streamlit Community Cloud:

Push your code to GitHub.
Create a new app and link it to your repository.
Access your live app via the generated URL.


## 🎨 Summary of My Approach
The Smart Product Recommender leverages a Retrieval-Augmented Generation (RAG) system to deliver personalized experiences:

Data Management: Loads initial product data into a Pandas DataFrame, supports dynamic additions, and indexes data in a ChromaDB vector database for efficient searches.
Recommendation Engine: Uses SentenceTransformer embeddings and WordNet synonyms to match queries, with sorting by relevance, price, or sentiment, enhanced by category and review weights.
User Interaction: Tracks views, feedback, and favorites via session state, offering features like search, comparison, product addition, favorites, and performance metrics.
UI Design: Features a responsive, card-based interface with custom CSS, light/dark theme toggling, and pagination for a seamless experience.
Extensibility: Enables real-time updates by refreshing the vector database and provides CSV exports for recommendations and the product database.


## 📋 Assumptions Made

Data Availability: Assumes sufficient initial data with valid image URLs and consistent schema for testing.
User Behavior: Assumes consistent user interactions to build meaningful preference patterns; limited historical data may reduce accuracy.
Network Access: Requires internet for loading external images and embedding models; offline use is not supported.
Performance: Assumes local machine handles ChromaDB and embedding computations; no high-load optimizations.
Input Validation: Assumes reasonable user inputs (e.g., non-negative prices) with basic error handling.
Review Authenticity: Relies on heuristic rules (e.g., short length, excessive exclamation marks) to detect inauthentic reviews.


## 🧪 Basic Test Coverage for Key Endpoints
Test the core functionalities with this script. Save as test_app.py:
pythonimport unittest
import pandas as pd
from app import DataLoader, ReviewAuthenticity, QueryExpander, VectorDB, RecommendationEngine

class TestProductRecommender(unittest.TestCase):
    def setUp(self):
        self.df = DataLoader.load_initial_product_data()
        self.vector_db = VectorDB()
        self.vector_db.add_products(self.df)
        self.recommender = RecommendationEngine(self.vector_db)

    def test_data_loading(self):
        df = DataLoader.load_product_data()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 4)

    def test_review_authenticity(self):
        review = "Great product!!! Buy now!!!"
        score = ReviewAuthenticity.score_review(review)
        self.assertLess(score, 0.4)
        
        review = "Good quality, reliable."
        score = ReviewAuthenticity.score_review(review)
        self.assertGreaterEqual(score, 0.5)

    def test_query_expansion(self):
        query = "wireless headphones"
        expanded = QueryExpander.expand_query(query)
        self.assertTrue(len(expanded.split()) > 2)  # Should include synonyms

    def test_recommendations(self):
        recommendations, _ = self.recommender.get_recommendations(
            "headphones", ["Electronics"], 0.0, 200.0, "relevance", 2
        )
        self.assertGreaterEqual(len(recommendations), 1)
        self.assertIn("Wireless Headphones Pro", [r['name'] for r in recommendations])

    def test_cross_category_recommendations(self):
        cross_recs = self.recommender.get_cross_category_recommendations("1", 0.0, 200.0, 2)
        self.assertGreaterEqual(len(cross_recs), 1)
        self.assertNotIn("Electronics", [r['category'] for r in cross_recs])

if __name__ == '__main__':
    unittest.main()
Run Tests
bashpython -m unittest test_app.py
Coverage

Tests data loading, review authenticity scoring, query expansion, standard recommendations, and cross-category recommendations.



A sleek, web-based e-commerce product recommendation system built with **Streamlit** and **ChromaDB**. Discover personalized product suggestions, compare options, and manage your favorites with ease!

---

## ✨ Features

- **Smart Search**: Find products by relevance, price, or sentiment.
- **Product Comparison**: Visualize differences with interactive bar and radar charts.
- **Dynamic Additions**: Add new products with real-time updates.
- **User Preferences**: Track favorites and infer preferences based on your behavior.
- **Data Export**: Download recommendations and the full product database as CSV.
- **Stylish UI**: Enjoy a responsive design with light/dark theme support.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>

Navigate to the project directory:
bashcd <repository-folder>

Install dependencies:
bashpip install -r requirements.txt

Launch the app:
bashstreamlit run app.py

Open your browser at http://localhost:8501 and start exploring!

Deployment
Deploy on Streamlit Community Cloud:

Push your code to GitHub.
Create a new app and link it to this repository.
Access your live app via the generated URL.


## 🧪 Testing
Ensure everything works smoothly with our basic test suite:
bashpython -m unittest test_app.py

## 🎨 Design Highlights

Responsive Layout: Optimized for all devices with a card-based grid.
Theme Toggle: Switch between light and dark modes.
Custom Styling: Enhanced with Tailwind-inspired CSS for a modern look.


## 📋 Assumptions

Internet access is required for image loading and model downloads.
Users provide reasonable inputs; minimal historical data may limit preference accuracy.
Local machine handles computation and storage efficiently.


## 📝 License
[Insert license here, e.g., MIT]

Feel free to contribute or fork this project!

## 🤝 Contributing
Contributions are welcome! Open an issue or submit a pull request to help improve this recommender system.
