# বাংলায় ছোট্ট সেন্টিমেন্ট অ্যানালাইসিস

{% hint style="info" %}
আমরা বই পড়ছি, নোটবুক কেন পড়বো? 

নোটবুক সবসময় আপডেটেড। এই বই থেকেও। 

যেহেতু গিটবুকে নোটবুক ঠিকমতো রেন্ডার হয়না, সেকারণে গুগল কোলাব এবং গিটহাবে দেখা উচিৎ। গিটহাব লিংক: 

নিজে নিজে প্র্যাকটিস করুন: [https://github.com/raqueeb/TensorFlow2/blob/master/small\_sent\_analysis\_v2.ipynb](https://github.com/raqueeb/TensorFlow2/blob/master/small_sent_analysis_v2.ipynb) এবং [https://nbviewer.jupyter.org/github/raqueeb/TensorFlow2/blob/master/small\_sent\_analysis\_v2.ipynb](https://nbviewer.jupyter.org/github/raqueeb/TensorFlow2/blob/master/small_sent_analysis_v2.ipynb)

কোলাব লিংক: [https://colab.research.google.com/github/raqueeb/TensorFlow2/blob/master/small\_sent\_analysis.ipynb](https://colab.research.google.com/github/raqueeb/TensorFlow2/blob/master/small_sent_analysis.ipynb)
{% endhint %}

### ছোট্ট একটা সেন্টিমেন্ট  অ্যানালাইসিস - ১

আমরা প্রায় বইয়ের শেষ পর্যায়ে চলে এসেছি। বাকি আছে দুটো মডেল সামনে। দুটোই সেন্টিমেন্ট অ্যানালাইসিসের ওপর। প্রথমটা একটা বেসিক মডেল, যেটা যে কেউ বুঝতে পারবেন। খুব সহজভাবে উপস্থাপন করা হয়েছে যেটা আগের সব শেখা বিষয়গুলোর কিছুটা কম্পাইলেশন। একদম ছোট একটা ডাটাসেট দিয়ে তৈরি করা, যা সহজেই ভেতরের কম্প্লেক্সিটি বুঝতে সাহায্য করবে। তবে শুরুতেই সেন্টিমেন্ট অ্যানালাইসিস কি আর কেনই বা এটা নিয়ে এতো তোলপাড়?

কিছুদিন আগ পর্যন্ত একটা ব্র্যান্ডের ভ্যালুয়েশন নির্ভর করত তার ব্যাপারে মানুষজন একে অপরকে কি বলছে? আমরা যাকে বলছি ‘ওয়ার্ড অফ মাউথ’। আমরা একটা জিনিস কেনার আগে বন্ধুবান্ধবকে জিজ্ঞাসা করে কিনতাম। অথবা ওই জিনিসটার ওপর পেপারে যদি কেউ লেখালিখি করে সেটার ওপর ভিত্তি করে একটা সিদ্ধান্ত নিয়ে ফেলতাম। কোম্পানিগুলো বড় বড় সেলিব্রেটিকে ব্যবহার করত তাদের প্রোডাক্টের অ্যাডভার্টাইজমেন্ট এ। খেয়াল আছে পিয়ারসনের কথা? ছোটবেলার টিভিতে ওই ব্র্যান্ডগুলোর জিংগেল এখনো মনে আছে।

এখন কি করি? একটা প্রোডাক্ট কেনার আগে ইন্টারনেটে দেখি তার ব্যাপারে ‘রিভিউ’ কেমন? বিশেষ করে প্রতিটা প্রোডাক্টের সাথে ‘ইউজার রিভিউ’ একটা বিশাল জিনিস। এতে বেড়েছে ট্রান্সপারেন্সি। একটা বাজে জিনিস গুছিয়ে দিয়ে পার পাবেনা কোম্পানি। একজন ব্যবহারকারী হিসেবে যেভাবে আমরা একটা প্রোডাক্ট নিয়ে ইন্টারনেটে রিসার্চ করি, সে ধরনের কাছাকাছি রিসার্চ করে থাকে বর্তমান কোম্পানিগুলো। তার প্রোডাক্টগুলো বাজারে কেমন চলছে, পাশাপাশি সেগুলোর ব্যাপারে ব্যবহারকারীরা কি বলছেন অথবা সামনের প্রোডাক্টে কি ধরনের ‘রিভিশন’ বা মডিফিকেশন আসতে পারে সেগুলোর ইনপুট আসবে ইন্টারনেটের বিভিন্ন রিভিউ থেকে। তবে সেটা সমস্যা হয়ে দাঁড়ায় যখন কোম্পানিটি তার ব্র্যান্ডের সোশ্যাল সেন্টিমেন্ট নিয়ে কাজ শুরু করেন। এত মানুষ, হাজারো কমেন্ট, অনেক ধরনের রিঅ্যাকশন এই সবকিছুকে প্রসেস করতে গেলে সেই রিসোর্স ম্যানেজ করবে কে? সেখানে চলে এসেছে ডিপ লার্নিং। মানুষের মনের খবর জানতে সেন্টিমেন্ট অ্যানালাইসিস এর নতুন জগৎ।

আমরা যখন একটা ব্যাপারে প্রচুর তথ্য পাই, তখন সেগুলোকে অ্যানালাইসিস করার জন্য আমাদেরকে ‘লাইন বাই লাইন’ পড়তে হবে। আর এখন যেভাবে সোশ্যাল মিডিয়া, ব্লগ, হাজারো ইলেকট্রনিক পাবলিকেশন মিনিটে অগুনিত ইউজার জেনারেটেড কনটেন্ট তৈরি করছে, সেখানে ডিপ লার্নিং ছাড়া এই ধরনের ডাটা প্রসেস করা দুষ্কর। শুধুমাত্র ডাটা প্রসেসিং নয় - এর মধ্যে টেক্সটগুলোর কন্টেক্সটুয়াল মাইনিং থেকে কোন তথ্যটি দরকার আর কোন তথ্যটি নয়, সেটা জানতে ডিপ লার্নিংকে বুঝতে হবে কিভাবে মানুষ তার মনের ভাব প্রকাশ করে। কে কি শব্দ বলল, শব্দটা বাক্যের কোথায় আছে, আগের এবং পরের বাক্যের সাথে এর সংযোগ/সিমিলারিটি কতটুকু সেটা বের করতে সেন্টিমেন্ট অ্যানালাইসিস বিশাল কাজ করে।

শুধুমাত্র কোম্পানিগুলো নয়, এখন অনেক দেশ তাদের জনগণের যেকোন বিষয়ে মনোভাব বোঝার জন্য এই সেন্টিমেন্ট অ্যানালাইসিস ব্যবহার করে থাকে। সরকারের প্রচুর সেবা যেহেতু জনগণের জন্য টার্গেট করে তৈরি করা, সেখানে সরকার তো জানতে চাইতেই পারেন - তাদের সার্ভিস ডেলিভারী প্লাটফর্মগুলো কিভাবে কাজ করছে, কেমন পারফর্ম করছে? সেন্টিমেন্ট অ্যানালাইসিস কিছু ধারনা আগে পেয়েছি আগের চ্যাপ্টারে। তবে সেটার আরো ভালো ধারণা নেবার চেষ্টা করব নিচের কোড থেকে।

সাধারণ ‘টেক্সট’ থেকে মানুষের মতো করে বোঝার সিস্টেম চলে এসেছে এখন। সেখানে সেন্টিমেন্ট অ্যানালাইসিসে যেকোনো একটা টেক্সট থেকে সেই ব্যাপারটা ‘পজিটিভ’ না ‘নেগেটিভ’ নাকি একেবারে ‘নিউট্রাল’ - সেটার একটা ধারণা দিতে পারে এই জিনিস। এর পাশাপাশি চলে এসেছে ‘ইনটেন্ট অ্যানালাইসিস’ যেটা আসলে সেন্টিমেন্ট অ্যানালাইসিসের আরেক ধাপ ওপরে - যা ওই টেক্সট থেকে ব্যবহারকারীর ‘ইনটেনশন’ মনোভাব নিয়ে একটা প্রেডিকশন দিতে পারে। ওই টেক্সট থেকে বলে দিতে পারে উনি এরপর কি করতে পারেন।

শুরুতেই ডেটা প্রি-প্রসেসিং। আর, একটা ছোট্ট ডাটাসেট। অ্যাক্যুরেসি ইম্পর্ট্যান্ট কিছু নয়, ফ্রেমওয়ার্কটা বুঝেলেই হবে। 

```python
try:
  # শুধুমাত্র টেন্সর-ফ্লো ২.x ব্যবহার করবো 
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
keras = tf.keras
```

```text
TensorFlow 2.x selected.
```

```python
# বাড়তি ওয়ার্নিং ফেলে দিচ্ছি, আপনাদের কাজের সময় লাগবে না 
import warnings
warnings.filterwarnings("ignore")
```

### ডেটা প্রি-প্রসেসিং

```python
# আমরা কিছু লাইব্রেরি যোগ করে নিচ্ছি 
import gensim, re
import numpy as np
import pandas as pd

# একটা ডেটা ডিকশনারি বানাই, আমাদের পছন্দ মতো বাক্য 

data = ['আমি মেশিন লার্নিং শিখতে পছন্দ করি',
        'আমার বই পড়তে ভালো লাগে না',
        'পাইথন শিখতে কষ্ট',
        'এই বইটা বেশ ভালো লাগছে',
        'আমার ন্যাচারাল ল্যাঙ্গুয়েজ প্রসেসিং পছন্দ']

labels = ['positive', 'negative', 'negative', 'positive', 'positive']

# আমাদের ডেটাকে কিছুটা প্রি-প্রসেস করি, বাংলা ইউনিকোড রেঞ্জের বাইরে সবকিছু ফেলে দিচ্ছি 

text = [re.sub(r'[^\u0980-\u09FF ]+', '', sentence) for sentence in data]
```

```python
# দেখি সে কি দেখায়?
text
```

```text
['আমি মেশিন লার্নিং শিখতে পছন্দ করি',
 'আমার বই পড়তে ভালো লাগে না',
 'পাইথন শিখতে কষ্ট',
 'এই বইটা বেশ ভালো লাগছে',
 'আমার ন্যাচারাল ল্যাঙ্গুয়েজ প্রসেসিং পছন্দ']
```

### টোকেনাইজার

```python
# আরো কিছু লাইব্রেরি যোগ করি 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# আমাদের টোকেনাইজার কতগুলো ফিচার এক্সট্র্যাক্ট করবে?
features = 500
tokenizer = Tokenizer(num_words = features)

# আমাদের টোকেনাইজারকে পুরো টেক্সটে ফিট করতে হবে 
tokenizer.fit_on_texts(text)

# আমাদের টোকেনাইজার চেনে সেরকম শব্দ নিয়ে আসতে হবে 
word_index = tokenizer.word_index

# ম্যাট্রিক্স এর মধ্যে টোকেনকে ফেলি 
X = tokenizer.texts_to_sequences(text)
X = pad_sequences(X)

# লেবেল তৈরি করি 
y = np.asarray(pd.get_dummies(labels))
```

```python
# লেবেল দেখি কি  আছে ভেতরে?

pd.get_dummies(labels)
```

  
    .dataframe tbody tr th:only-of-type {  
        vertical-align: middle;  
    }  
  
    .dataframe tbody tr th {  
        vertical-align: top;  
    }  
  
    .dataframe thead th {  
        text-align: right;  
    }  


|  | negative | positive |
| :--- | :--- | :--- |
| 0 | 0 | 1 |
| 1 | 1 | 0 |
| 2 | 1 | 0 |
| 3 | 0 | 1 |
| 4 | 0 | 1 |

```python
# ট্রেইন এবং টেস্ট ডেটাসেটকে আলাদা করি 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
```

### ওয়ার্ড২ভেক

এখানে আমরা দুটো এপ্রোচ নিয়েছি, একটা ছোট ডেটাসেট, আরেকটা একটু বড় বাংলা উইকি থেকে নেয়া।

```python
# আগেও ব্যবহার করেছিলাম 

!wget https://media.githubusercontent.com/media/raqueeb/datasets/master/bnwiki-texts.zip
!unzip bnwiki-texts.zip
preprocessed_text_file_path = 'bnwiki-texts-preprocessed.txt'

lines_from_file = []
with open(preprocessed_text_file_path, encoding='utf8') as text_file:
    for line in text_file:
        lines_from_file.append(line)

tokenized_lines = []
for single_line in lines_from_file:
    tokenized_lines.append(single_line.split())
```

```text
--2019-11-07 02:17:34--  https://media.githubusercontent.com/media/raqueeb/datasets/master/bnwiki-texts.zip
Resolving media.githubusercontent.com (media.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
Connecting to media.githubusercontent.com (media.githubusercontent.com)|151.101.0.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 61696513 (59M) [application/zip]
Saving to: ‘bnwiki-texts.zip’

bnwiki-texts.zip    100%[===================>]  58.84M   136MB/s    in 0.4s    

2019-11-07 02:17:35 (136 MB/s) - ‘bnwiki-texts.zip’ saved [61696513/61696513]

Archive:  bnwiki-texts.zip
  inflating: bnwiki-texts-preprocessed.txt  
```

```python
# ওয়ার্ড২ভেককে ট্রেইন করি আমাদের দু ধরণের ডেটা দিয়ে, টেস্ট করি দুটো দিয়ে দুবার - আলাদা করে 
# একটু সময় লাগবে 
word_model = gensim.models.Word2Vec(tokenized_lines, size=300, min_count=1, iter=10)
# word_model = gensim.models.Word2Vec(text, size=300, min_count=1, iter=10)
```

```python
# টেস্ট করি ওয়ার্ড২ভেক কাজ করে কিনা 
word_model.wv.most_similar(positive='আমি')
```

```text
/usr/local/lib/python3.6/dist-packages/gensim/matutils.py:737: FutureWarning: Conversion of the second argument of issubdtype from `int` to `np.signedinteger` is deprecated. In future, it will be treated as `np.int64 == np.dtype(int).type`.
  if np.issubdtype(vec.dtype, np.int):





[('আমরা', 0.7259500026702881),
 ('তুমি', 0.6985121965408325),
 ('আপনি', 0.6706202030181885),
 ('আমার', 0.669513463973999),
 ('তোমরা', 0.66762375831604),
 ('তোমার', 0.6606091856956482),
 ('আমাকে', 0.6351409554481506),
 ('তোমাকে', 0.6340588331222534),
 ('আপনাকে', 0.6301310658454895),
 ('আপনারা', 0.5874826312065125)]
```

```python
# নতুন ম্যাট্রিক্সে সেভ করি ভেক্টরগুলোকে 
embedding_matrix = np.zeros((len(word_model.wv.vocab) + 1, 300))
for i, vec in enumerate(word_model.wv.vectors):
  embedding_matrix[i] = vec
```

### তৈরি করি মডেল

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# লেয়ারকে ইনিশিয়ালাইজ করি, অবশ্যই সিকোয়েন্সিয়াল মডেল  
model = Sequential()
# শব্দ ভেক্টরকে এমবেড করি 
model.add(Embedding(len(word_model.wv.vocab)+1, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
# ডেটার মধ্যে কো-রিলেশন বোঝার চেষ্টা করতে দেই মডেলকে 
model.add(LSTM(300,return_sequences=False))
model.add(Dense(y.shape[1],activation="softmax"))
# মডেলের কাঠামো দেখি 
model.summary()
```

```text
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 6, 300)            200881800 
_________________________________________________________________
lstm (LSTM)                  (None, 300)               721200    
_________________________________________________________________
dense (Dense)                (None, 2)                 602       
=================================================================
Total params: 201,603,602
Trainable params: 721,802
Non-trainable params: 200,881,800
_________________________________________________________________
```

```python
# মডেলকে কম্পাইল করি 

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])
```

```python
# মডেলকে ট্রেইন করি ব্যাচ দিয়ে, দেখানোর জন্য

batch = 32
epochs = 6
model.fit(X_train, y_train, batch, epochs)
```

```text
Train on 4 samples
Epoch 1/6
4/4 [==============================] - 5s 1s/sample - loss: 0.7121 - acc: 0.2500
Epoch 2/6
4/4 [==============================] - 0s 29ms/sample - loss: 0.0903 - acc: 1.0000
Epoch 3/6
4/4 [==============================] - 0s 29ms/sample - loss: 0.0146 - acc: 1.0000
Epoch 4/6
4/4 [==============================] - 0s 29ms/sample - loss: 0.0039 - acc: 1.0000
Epoch 5/6
4/4 [==============================] - 0s 29ms/sample - loss: 0.0015 - acc: 1.0000
Epoch 6/6
4/4 [==============================] - 0s 31ms/sample - loss: 7.6183e-04 - acc: 1.0000





<tensorflow.python.keras.callbacks.History at 0x7f6780296668>
```

```python
# ইভ্যালুয়েট করে দেখি কি অবস্থা 

model.evaluate(X_test, y_test)
```

```text
1/1 [==============================] - 1s 545ms/sample - loss: 0.4236 - acc: 1.0000





[0.42364874482154846, 1.0]
```

## অথবা ফাস্টটেক্সট

```python
from urllib.request import urlopen
import gzip

# ফাস্টটেক্সটের প্রি-ট্রেইন ভেক্টরগুলোকে নিয়ে আসি, বাংলায় - cc.bn.300.vec.gz ৯০০ মেগাবাইটের মতো 
file = gzip.open(urlopen('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz'))
```

```python
vocab_and_vectors = {}
# শব্দগুলোকে ডিকশনারি ইনডেক্সে রাখি, আর ভেক্টরগুলোকে ওয়ার্ড ভ্যালুতে, একটু সময় নেবে  
for line in file:
  values = line.split()
  word = values[0].decode('utf-8')
  vector = np.asarray(values[1:], dtype='float32')
  # print(word)
  vocab_and_vectors[word] = vector
```

```python
# দেখি কি আছে এখানে?

list(vocab_and_vectors.keys())[2]
```

```text
'</s>'
```

```python
# এমবেডিং ম্যাট্রিক্স 

embedding_matrix = np.zeros((len(vocab_and_vectors) + 1, 300))
for i, word, in enumerate(vocab_and_vectors.keys()):
  # print(i)
  # print(word)
  embedding_vector = vocab_and_vectors.get(word)
  # যে শব্দগুলো পাওয়া যাবে না, সেগুলোকে '০'তে রাখবো 
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
```

```python
# দেখি কি আছে?

embedding_matrix
```

```text
array([[ 3.00000000e+02,  3.00000000e+02,  3.00000000e+02, ...,
         3.00000000e+02,  3.00000000e+02,  3.00000000e+02],
       [ 5.64000010e-02,  3.53000015e-02,  3.46000008e-02, ...,
         4.83999997e-02, -4.99999989e-03,  8.00000038e-03],
       [-1.15999999e-02, -5.90000022e-03, -7.60000013e-03, ...,
        -1.83000006e-02,  2.96999998e-02, -1.77999996e-02],
       ...,
       [ 5.29999984e-03,  7.60000013e-03,  9.80000012e-03, ...,
         2.12999992e-02,  8.99999985e-04, -9.99999975e-05],
       [-2.89999996e-03,  1.77999996e-02,  3.99999991e-02, ...,
        -3.53000015e-02,  2.55999994e-02, -5.49999997e-03],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])
```

```python
# এখানেও

vocab_and_vectors
```

```text
{'1468578': array([300.], dtype=float32),
 'একটি': array([-3.560e-02, -1.420e-02,  1.280e-02, -6.040e-02, -1.060e-02,
         2.900e-02,  3.180e-02,  1.510e-02, -2.640e-02, -2.710e-02,
         2.440e-02, -9.800e-03, -7.000e-03, -2.620e-02, -9.400e-03,
        -2.450e-02, -3.400e-02,  1.200e-02,  2.360e-02, -4.000e-04,
        -7.370e-02,  1.480e-02, -1.480e-02,  1.760e-02,  2.890e-02,
         5.570e-02, -1.490e-01,  1.170e-02,  6.600e-03, -2.050e-02,
        -1.280e-02,  1.250e-02, -1.790e-02, -7.300e-03, -9.700e-03,
         7.600e-03, -2.800e-03,  6.300e-03, -1.260e-02, -5.460e-02,
         1.940e-02,  1.340e-02, -5.200e-02,  1.400e-02, -9.700e-03,
         1.900e-03, -2.300e-03,  4.090e-02,  2.710e-02, -4.760e-02],
       dtype=float32),
 ...}
```

### মডেল

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# লেয়ারকে ইনিশিয়ালাইজ করি, অবশ্যই সিকোয়েন্সিয়াল মডেল  
model = Sequential()
# শব্দ ভেক্টরকে এমবেড করি 
model.add(Embedding(len(vocab_and_vectors)+1, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
# ডেটার মধ্যে কো-রিলেশন বোঝার চেষ্টা করতে দেই মডেলকে 
model.add(LSTM(300, return_sequences=False))
model.add(Dense(y.shape[1], activation="softmax"))
# মডেলের কাঠামো দেখি 
model.summary()
```

```text
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 6, 300)            440574000 
_________________________________________________________________
lstm_1 (LSTM)                (None, 300)               721200    
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 602       
=================================================================
Total params: 441,295,802
Trainable params: 721,802
Non-trainable params: 440,574,000
_________________________________________________________________
```

```python
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['acc'])
```

```python
batch = 32
epochs = 12
model.fit(X_train,y_train,batch,epochs)
```

```text
Train on 4 samples
Epoch 1/12
4/4 [==============================] - 3s 670ms/sample - loss: 0.6542 - acc: 0.7500
Epoch 2/12
4/4 [==============================] - 0s 106ms/sample - loss: 0.5248 - acc: 1.0000
Epoch 3/12
4/4 [==============================] - 0s 108ms/sample - loss: 0.4373 - acc: 1.0000
Epoch 4/12
4/4 [==============================] - 0s 107ms/sample - loss: 0.3568 - acc: 1.0000
Epoch 5/12
4/4 [==============================] - 0s 107ms/sample - loss: 0.2827 - acc: 1.0000
Epoch 6/12
4/4 [==============================] - 0s 108ms/sample - loss: 0.2169 - acc: 1.0000
Epoch 7/12
4/4 [==============================] - 0s 106ms/sample - loss: 0.1599 - acc: 1.0000
Epoch 8/12
4/4 [==============================] - 0s 106ms/sample - loss: 0.1127 - acc: 1.0000
Epoch 9/12
4/4 [==============================] - 0s 108ms/sample - loss: 0.0758 - acc: 1.0000
Epoch 10/12
4/4 [==============================] - 0s 109ms/sample - loss: 0.0487 - acc: 1.0000
Epoch 11/12
4/4 [==============================] - 0s 106ms/sample - loss: 0.0294 - acc: 1.0000
Epoch 12/12
4/4 [==============================] - 0s 106ms/sample - loss: 0.0166 - acc: 1.0000





<tensorflow.python.keras.callbacks.History at 0x7f65115c6630>
```

```python
model.evaluate(X_test,y_test)
```

```text
1/1 [==============================] - 1s 863ms/sample - loss: 4.4300e-04 - acc: 1.0000





[0.0004430027911439538, 1.0]
```

## প্রেডিকশন

```python
# sents = ["আমি ভালো কাজ জানিনা",
#          "এই ক্লাসটা বড় হতে পারতো",
#          "আমি লেখাটা পছন্দ করছি না",
#          "ভালো কথা না",
#          "সুন্দর একটা ফুল",
#          "এই ক্লাস ভালো লেগেছে"]

# ইচ্ছেমতো বাক্য লিখে টেস্ট করুন 

sents = ['আমি মেশিন লার্নিং শিখতে পছন্দ করি', 'পছন্দ করি', 'করি না', 'আমি ভাল', 'বেশ কষ্ট', 'এই বইটা বেশ ভালো লাগছে']
# sent_n = [[word_index[w]+3 for w in s.split()] for s in sents]
sent_n = tokenizer.texts_to_sequences(sents)
X_new = pad_sequences(sent_n)
prediction = model.predict(X_new)
```

```python
# '১' মানে পজিটিভ, '০' মানে নেগেটিভ, কারণ pd.get_dummy() দেখিয়েছে কলাম '০' মানে নেগেটিভ আর কলাম '১' মানে পজিটিভ  

np.argmax(prediction, axis=1)
```

```text
array([1, 0, 0, 0, 0, 1])
```

