# Visualizing Embeddings using PCA and Kmeans
This repository contains code for visualizing embeddings using PCA (Principal Component Analysis) and Kmeans clustering. The goal is to capture the meaning of text data by reducing high-dimensional embeddings to a lower-dimensional space.

## Verrtex AI environment setup
Note: I have used this template from DeepLearning course. You need to create your own credentials to initialize the project

Load credentials and relevant Python Libraries
```
from utils import authenticate
credentials, PROJECT_ID = authenticate() #Get credentials and project ID
REGION = 'us-central1'
# Import and initialize the Vertex AI Python SDK

import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
              credentials = credentials)
```
# Create Embeddings
Read sentences from a text file (you can uncomment the code to read from a file if needed):
```
# with open('sentence_doc.txt') as f:
#     lines = f.readlines()
```
```
## I have retained the same example from DeepLearning cource
in_1 = "Missing flamingo discovered at swimming pool"
in_2 = "Sea otter spotted on surfboard by beach"
in_3 = "Baby panda enjoys boat ride"
in_4 = "Breakfast themed food truck beloved by all!"
in_5 = "New curry restaurant aims to please!"
in_6 = "Python developers are wonderful people"
in_7 = "TypeScript, C++ or Java? All are great!" 

#create a list of sentences
input_text_lst_news = [in_1, in_2, in_3, in_4, in_5, in_6, in_7]
```

#### The textembedding-gecko model in Google Vertex AI Embeddings provides 768 dimensions. This high-dimensional representation allows capturing intricate semantic relationships between words and sentences.
```
import numpy as np
from vertexai.language_models import TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")
```
Get embeddings for all pieces of text.
Store them in a 2D NumPy array (one row for each embedding).

```
embeddings = []
for input_text in input_text_lst_news:
    emb = embedding_model.get_embeddings(
        [input_text])[0].values
    embeddings.append(emb)
    
embeddings_array = np.array(embeddings)
print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)
```
#### Reduce embeddings from 768 to 2 dimensions for visualization
- We'll use principal component analysis (PCA).
- You can learn more about PCA in [this video](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/73zWO/reducing-the-number-of-features-optional) from the Machine Learning Specialization.

```
from sklearn.decomposition import PCA

# Perform PCA for 2D visualization
PCA_model = PCA(n_components = 2)
PCA_model.fit(embeddings_array)
new_values = PCA_model.transform(embeddings_array)
print("Shape: " + str(new_values.shape))
print(new_values)
import matplotlib.pyplot as plt
import mplcursors
%matplotlib ipympl

from utils import plot_2D
plot_2D(new_values[:,0], new_values[:,1], input_text_lst_news)
```

#### Visualize sentence Clusters with Kmeans
Classify the scatterplot above using Kmeans

```
from sklearn.cluster import KMeans
import pandas as pd
df = pd.DataFrame(new_values)
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(df)
plt.scatter(df[0], df[1], c=kmeans.labels_)
plt.show()
```
#### Compute cosine similarity¶
The cosine_similarity function expects a 2D array, which is why we'll wrap each embedding list inside another list.
You can verify that sentence 1 and 2 have a higher similarity compared to sentence 1 and 4, even though sentence 1 and 4 both have the words "desert" and "plant".

```
from sklearn.metrics.pairwise import cosine_similarity
def compare(embeddings,idx1,idx2):
    return cosine_similarity([embeddings[idx1]],[embeddings[idx2]])
##example
print(in_1)
print(in_2)
print(compare(embeddings,0,1))
```

