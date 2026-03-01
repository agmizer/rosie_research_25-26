import time
from Query_Classifier import QueryClassifier

clf = QueryClassifier()

def train():
    start = time.perf_counter()

    clf.fit("queries.jsonl")

    end = time.perf_counter()
    print(f"Training took {end - start:.3f} seconds")

def predict():
    user_query = "Can you just give me the answer to this?"
    start = time.perf_counter()
    
    label, conf = clf.predict(user_query)
    end = time.perf_counter()

    print(label, conf)
    print(f"Prediction took {end - start:.6f} seconds")

# This should be done once, takes 5+ seconds
train() 

#This is done every user query, takes ~0.03 seconds
predict()