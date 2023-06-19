#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[4]:


# Reading data
df = pd.read_csv('C:/Users/Samatha/Downloads/Reviews.csv')
df


# In[6]:


df.shape


# In[7]:


df = df.head(500)


# In[8]:


print(df.shape)


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df['Score'].value_counts()


# In[12]:


df['Score'].value_counts().sort_index().plot(kind='bar',title='Count of Reviews by Stars',figsize=(12,7))


# In[13]:


ax = df['Score'].value_counts().sort_index().plot(kind='bar',title='Count of Reviews by Stars',figsize=(12,7))

ax.set_xlabel('Review Stars')
plt.show()


# In[14]:


example = df['Text'][120]
print(example)


# In[15]:


nltk.word_tokenize(example)


# In[16]:


tokens = nltk.word_tokenize(example)
tokens[:12]


# In[17]:


nltk.pos_tag(tokens)


# In[18]:


tagged = nltk.pos_tag(tokens)
tagged[:12]


# In[19]:


entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[22]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia


# In[23]:


sia.polarity_scores('I am very poor')


# In[24]:


sia.polarity_scores('Best thing in the world')


# In[25]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[26]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
vaders.head()


# In[27]:


df.tail()


# In[28]:


fig, axs = plt.subplots(1, 3, figsize=(14, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[31]:


ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

