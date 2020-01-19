## আমাদের কর্পাস

বাংলা নাচারাল ল্যাঙ্গুয়েজ প্রসেসিং

মেশিন লার্নিং বা ডিপ লার্নিং মডেলের যেকোন ভাষাকে বুঝতে হলে তাকে শুরুতে মানুষের মতো হাটি হাটি পা পা করে বুঝতে হয়। আমি এখানে শুরুতে অক্ষরের কথা বলবো না কারণ মেশিন সংখ্যা ছাড়া বোঝেনা। তবে, এর জন্য শুরুতে একটা বড় বাক্যকে ভেঙ্গে ছোট ছোট মিনিংফুল শব্দে ভাগ করে নিতে হয়, যাতে একেকটা শব্দের সাথে আরেকটা শব্দের সম্পর্ক বুঝতে পারে। এই যে বড় বড় বাক্যকে ভেঙ্গে ছোট ছোট মিনিংফুল ইউনিটে ভাগ করাকে আমরা বলছি টোকেনাইজেশন।

যেহেতু মেশিন সংখ্যা ছাড়া বোঝেনা, সেখানে এই ছোট ছোট ইউনিটে ভাগ করা শব্দগুলোকে আমরা পাল্টে ফেলব অঙ্কের সংখ্যাতে। এটাকে আমরা বলছি ভেক্টরাইজেশন।

১. টোকেনাইজেশন

২. ভেক্টরাইজেশন


```python
sentences = ['আমি মাঝে মধ্যেই ফিরে যাই পুরানো কিছু ক্লাসিক বইয়ে', 'বিশেষ করে বেসিক ঝালাই করার জন্য']
```


```python
import nltk
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True




```python
from sklearn.feature_extraction.text import CountVectorizer

# ট্রান্সফরমেশন তৈরি করি
vectorizer = CountVectorizer()

# টোকেনাইজ এবং ভোকাবুলারি তৈরি করি
vectorizer.fit(sentences)

# সামারি দেখি
vectorizer.vocabulary_
```




    {'আম': 0, 'কর': 1, 'জন': 2, 'বইয়': 3, 'মধ': 4}




```python
import warnings
warnings.filterwarnings("ignore")
```


```python
# ইউনিকোডে দেখুন নিচের লিঙ্কে
# https://jrgraphix.net/r/Unicode/0980-09FF

from nltk import word_tokenize

vectorizer = CountVectorizer(encoding='utf-8', tokenizer=word_tokenize)

vectorizer.fit(sentences)
vectorizer.vocabulary_
```




    {'আমি': 0,
     'করার': 1,
     'করে': 2,
     'কিছু': 3,
     'ক্লাসিক': 4,
     'জন্য': 5,
     'ঝালাই': 6,
     'পুরানো': 7,
     'ফিরে': 8,
     'বইয়ে': 9,
     'বিশেষ': 10,
     'বেসিক': 11,
     'মধ্যেই': 12,
     'মাঝে': 13,
     'যাই': 14}




```python
print(vectorizer.vocabulary_)
```

    {'আমি': 0, 'মাঝে': 13, 'মধ্যেই': 12, 'ফিরে': 8, 'যাই': 14, 'পুরানো': 7, 'কিছু': 3, 'ক্লাসিক': 4, 'বইয়ে': 9, 'বিশেষ': 10, 'করে': 2, 'বেসিক': 11, 'ঝালাই': 6, 'করার': 1, 'জন্য': 5}



```python
vectorizer.transform(sentences).toarray()
```




    array([[1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
           [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]])




```python
vec = CountVectorizer()
x = vec.fit_transform(sentences).toarray()
print(x.shape)
print(vec.get_feature_names())
```

    (2, 5)
    ['আম', 'কর', 'জন', 'বইয়', 'মধ']



```python
# ইউনিকোডে দেখুন নিচের লিঙ্কে
# https://jrgraphix.net/r/Unicode/0980-09FF

vectorizer = CountVectorizer(encoding='utf-8', token_pattern=r'[\u0980-\u09ff]+')
vectorizer.fit(sentences)
vectorizer.vocabulary_
```




    {'আমি': 0,
     'করার': 1,
     'করে': 2,
     'কিছু': 3,
     'ক্লাসিক': 4,
     'জন্য': 5,
     'ঝালাই': 6,
     'পুরানো': 7,
     'ফিরে': 8,
     'বইয়ে': 9,
     'বিশেষ': 10,
     'বেসিক': 11,
     'মধ্যেই': 12,
     'মাঝে': 13,
     'যাই': 14}




```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer =TfidfVectorizer(encoding='utf-8', tokenizer=word_tokenize)

vectorizer.fit(sentences)
vectorizer.vocabulary_
```




    {'আমি': 0,
     'করার': 1,
     'করে': 2,
     'কিছু': 3,
     'ক্লাসিক': 4,
     'জন্য': 5,
     'ঝালাই': 6,
     'পুরানো': 7,
     'ফিরে': 8,
     'বইয়ে': 9,
     'বিশেষ': 10,
     'বেসিক': 11,
     'মধ্যেই': 12,
     'মাঝে': 13,
     'যাই': 14}




```python
vectorizer.transform(sentences).toarray()
```




    array([[0.33333333, 0.        , 0.        , 0.33333333, 0.33333333,
            0.        , 0.        , 0.33333333, 0.33333333, 0.33333333,
            0.        , 0.        , 0.33333333, 0.33333333, 0.33333333],
           [0.        , 0.40824829, 0.40824829, 0.        , 0.        ,
            0.40824829, 0.40824829, 0.        , 0.        , 0.        ,
            0.40824829, 0.40824829, 0.        , 0.        , 0.        ]])




```python
print(vectorizer.idf_)
```

    [1.40546511 1.40546511 1.40546511 1.40546511 1.40546511 1.40546511
     1.40546511 1.40546511 1.40546511 1.40546511 1.40546511 1.40546511
     1.40546511 1.40546511 1.40546511]



```python
vector = vectorizer.transform([sentences[0]])
# এনকোডেড ভেক্টরকে সামারাইজ করি
print(vector.shape)
print(vector.toarray())
```

    (1, 15)
    [[0.33333333 0.         0.         0.33333333 0.33333333 0.
      0.         0.33333333 0.33333333 0.33333333 0.         0.
      0.33333333 0.33333333 0.33333333]]



```python
cities = ['ঢাকা', 'বার্লিন', 'কুমিল্লা', 'শিকাগো', 'সিঙ্গাপুর']
cities
```




    ['ঢাকা', 'বার্লিন', 'কুমিল্লা', 'শিকাগো', 'সিঙ্গাপুর']




```python
from sklearn.preprocessing import LabelEncoder
```


```python
encoder = LabelEncoder()
city_labels = encoder.fit_transform(cities)
city_labels
```




    array([1, 2, 0, 3, 4])




```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
city_labels = city_labels.reshape((5, 1))
encoder.fit_transform(city_labels)
```




    array([[0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])


