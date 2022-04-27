
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')


# In[3]:


import nltk.corpus


# In[3]:


# tokenization


# In[4]:


from nltk.tokenize import word_tokenize


# In[5]:


chess = "Samay Raina is the best chess streamer in the world"


# In[6]:


word_tokenize(chess)


# In[7]:


#sentence tokenizer 
from nltk.tokenize import sent_tokenize


# In[8]:


chess2 = "Samay Raina is the best chess streamer in the world . Sagar Shah is the best chess coach in the world "


# In[9]:


sent_tokenize(chess2)


# In[10]:


#checking the number of token
len(word_tokenize(chess))


# In[11]:


len(sent_tokenize(chess2))


# In[ ]:


#bigram and n-gram => in an email write "hello how " sugestion will come "are you", in iPhone  how siri  responding 


# In[12]:


astronaut = "Can anybody hear me or am I talking to myself ? My mind is running empty in the search for someone else"


# In[13]:


astronaut_token = (word_tokenize(astronaut))


# In[16]:


astronaut_token


# In[14]:


list(nltk.bigrams(astronaut_token)) # if it is bigram feeds 2 token at a time to machine


# In[15]:


list(nltk.trigrams(astronaut_token))


# In[17]:


list(nltk.ngrams(astronaut_token,5))


# In[ ]:


#stemming


# In[19]:


from nltk.stem import PorterStemmer


# In[20]:


my_stem = PorterStemmer()


# In[21]:


my_stem.stem("eating")


# In[22]:


my_stem.stem("going")


# In[23]:


my_stem.stem("shopping")


# In[ ]:


#pos-tagging => tags particular part of speetch like noun, adj , determinor

nltk.download('averaged_perceptron_tagger')
# In[28]:


tom = "Tom Hanks is the best actor in the world"


# In[32]:


tom_token = word_tokenize(tom)


# In[33]:


nltk.pos_tag(tom_token)


# In[ ]:


# Named entity recognize  => identifying person, location of that word 


# In[40]:


from nltk import ne_chunk


# In[45]:


nltk.download('maxent_ne_chunker')


# In[51]:


nltk.download('words')


# In[52]:


president = "Barack obama was the 44th President of America"


# In[48]:


president_token = word_tokenize(president)


# In[53]:


president_pos = nltk.pos_tag(president_token)


# In[54]:


print(ne_chunk(president_pos))


# In[55]:


get_ipython().system('pip install gTTS')


# In[65]:


from gtts import gTTS


# In[66]:


import IPython.display


# In[67]:


from IPython.display import Audio


# In[78]:


tts  = gTTS('Magnus Carlsen is the number one chess player in the world')


# In[79]:


tts.save('1.wav')


# In[80]:


sound_file = '1.wav'


# In[81]:


Audio(sound_file,autoplay=True)

