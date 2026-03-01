import json
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

class QueryClassifier:
    """
    User Query Classifier that takes in a user input and classifies it as a 'conceptual' question,
    a 'homework' question, a 'check my work' request, a 'final answer request', a 'study strategy' 
    request, or 'other'. Uses a 384 dimensional natural language embedder and a Logistic Regression
    classifier. 
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.embedder = SentenceTransformer(embedding_model)
        self.LR = LogisticRegression(max_iter = 1000)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False


    def _load_data(self, path):
        """
        Private method that extracts the queries and labels from dataset
        
        :param str: path: Path for the query dataset
        """
        queries = []
        labels = []

        with open(path, "r", encoding = "utf-8") as f:
            for line in f:
                obj = json.loads(line)
                queries.append(obj["text"])
                labels.append(obj["label"])

        return queries, labels
    

    def fit(self, path):
        """
        Method that fits the queries dataset with labels to a Logistic Regression Model
        
        :param str: path: Path for the query dataset
        """
        queries, labels = self._load_data(path)

        y = self.label_encoder.fit_transform(labels)
        X = self.embedder.encode(queries, normalize_embeddings = True)

        self.LR.fit(X, y)
        self.is_fitted = True


    def predict(self, user_query):
        """
        Method that takes in a new user query and returns the predicted label 
        and label confidence
        
        :param user_query: new user query that needs to be classified 
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been trained.")
        
        emb = self.embedder.encode([user_query], normalize_embeddings = True)

        probs = self.LR.predict_proba(emb)[0]
        idx = np.argmax(probs)

        label = self.label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])

        return label, confidence