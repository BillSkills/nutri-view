import pandas as pd
import numpy as np
import os
import openai
from ast import literal_eval
import ast

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# search through the reviews for a specific product
def search_reviews(df, product_description, n=1, pprint=True):
    product_embedding = get_embedding(
        product_description,
        model="text-embedding-ada-002"
    )
    #compare the product embedding with the embeddings of all the food names in the dataframe
    def cosine_similarity (a, b):
        #takes in two lists and computes the cosine similarity between them
        #a and b are both lists
        #returns a float
        a= ast.literal_eval(a)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    df["similarity"] = df["embedding"].apply(
        lambda x: cosine_similarity(x, np.array(product_embedding))
    )
    results = (df.sort_values("similarity", ascending=False).head(n))
    return results

def getindexofmaxofnparray(x):
    with open ("class_list.txt", "r") as myfile:
        class_list = myfile.readlines()
    
    class_list = [i.split(" ")[1] for i in class_list]

    # print(class_list)

    return class_list[np.where(x == np.amax(x))[0][0]]


if __name__ == "__main__":
    getindexofmaxofnparray(np.array([1,2,3,4,5,6,7,8,9,10]))
