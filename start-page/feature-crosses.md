## ডিপ লার্নিং কেন? নন-লিনিয়ার সমস্যা, ফিচার ক্রস

এই প্রশ্নটা প্রায় অনেকেই করেন, মেশিন লার্নিং থাকতে ডিপ লার্নিং কেন দরকার পড়লো? এর উত্তর সবার জানা, তবে যে জন্য আমি ডিপ লার্নিংয়ে এসেছি সেটা আলাপ করি বরং। আমার একটা সমস্যা হচ্ছে, যে কোন ডাটা পেলেই সেটাকে আগে কাগজে প্লট করে ফেলি। তাহলে সেটা বুঝতে সুবিধা হয়। 

মনে আছে আমাদের আইরিশ ডাটা সেটের কথা? সেখানে সবগুলো প্রজাতির ডাটাকে প্লট করলে কিছুটা এরকম দেখা যেত। তিন প্রজাতিকে ছবির মধ্যে আলাদা করা খুব একটা সমস্যা ছিল না। কারণ তিনটা ছবির মধ্যে দুটো লাইন বা সরলরেখা টানলেই কিন্তু তিনটা প্রজাতিকে আলাদা করে ফেলা যেত। এক পিকচার থেকে আরেক পিকচারের ডিসিশন সারফেস এবং ডিসিশন বাউন্ডারি সরলরেখার। 

<img src="https://raw.githubusercontent.com/raqueeb/deep_learning_book/master/assets/iris.png" alt="আইরিশ ডেটাসেটের ডিসিশন বাউন্ডারি"> চিত্রঃ আইরিশ ডেটাসেটের ডিসিশন বাউন্ডারি

বয়সের সাথে ওজন বাড়বে, বাড়ি স্কয়ার ফিট এর সাথে দাম বাড়বে, এধরনের লিনিয়ার সম্পর্কগুলোতে ডাটাকে প্লট করলে সেগুলোকে সোজা লাইন দিয়ে আলাদা করে দেখা যায় সহজে। কিন্তু ডাটা যদি এমন হয়? কিভাবে একটা লাইন দিয়ে দুটো ফিচারকে ভাগ করবেন?

<img src="https://raw.githubusercontent.com/raqueeb/deep_learning_book/master/assets/fc1.png" alt="নন-লিনিয়ার ক্লাসিফিকেশন সমস্যা"> চিত্রঃ নন-লিনিয়ার ক্লাসিফিকেশন সমস্যা 

এটা নন-লিনিয়ার ক্লাসিফিকেশন সমস্যা। নন-লিনিয়ার এর সমস্যা হচ্ছে তাদের 'ডিসিশন সারফেস' সরলরেখা নয়। এই ছবিতে যদি দুটো ফিচার থাকে তাহলে সেটা দিয়ে এই সমস্যার সমাধান করা সম্ভব নয়। সেটার জন্য প্রয়োজন ফিচার ক্রস। মানে এই নতুন ফিচার কয়েকটা ফিচার স্পেসের গুণফলের আউটকাম নন-লিনিয়ারিটিকে এনকোড করে মানে আরেকটা সিনথেটিক ফিচার তৈরি করে সেটার মাধ্যমে এই নন-লিনিয়ারিটিকে কিছুটা ডিল করা যায়। ক্রস এসেছে ক্রস প্রোডাক্ট থেকে। এখানে x1 = x2x3 (সাবস্ক্রিপ্ট হবে)

## একটা নন-লিনিয়ারিটির উদাহরণ দেখি 


```python
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf

%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

## ভার্সন চেক করি


```python
sklearn.__version__
```




    '0.21.3'




```python
tf.__version__
```




    '2.0.0'



## ১. ডেটাকে আমরা লোড করে সেটাকে ট্রেইন এবং টেস্টসেটে ভাগ করি 

ধরুন আপনি ডাটা প্লট করে দেখলেন এই অবস্থা। কি করবেন? নিচের ছবি দেখুন। এই ডাটাসেটে অক্ষাংশ, দ্রাঘিমাংশ, ভূমির উচ্চতা ইত্যাদি আছে। আমরা সবগুলোর মধ্যে শুধুমাত্র অক্ষাংশ, দ্রাঘিমাংশ প্লট করছি। দেখুন কি অবস্থা। একটা ফিচার ভেতরে আরেকটা ঘিরে রয়েছে সেটাকে চারপাশ দিয়ে। এখন কিভাবে এদুটোকে আলাদা করবেন? সোজা লাইন টেনে সম্ভব না। চেষ্টা করি নিউরাল নেটওয়ার্ক দিয়ে।  


```python
# df = pd.read_csv('geoloc_elev.csv')
df = pd.read_csv('https://raw.githubusercontent.com/raqueeb/TensorFlow2/master/datasets/geoloc_elev.csv')

# আমাদের দুটো ফিচার হলেই যথেষ্ট 
X = df[['lat', 'lon']].values
y = df['target'].values
```


```python
df.plot(kind='scatter',
        x='lat',
        y='lon',
        c='target',
        cmap='bwr');
```


![png](output_10_0.png)


মেশিন লার্নিং এ ফিচার ক্রসের সুবিধা থাকলেও আমার পছন্দের ব্যাপার হচ্ছে মডেলে নন লিনিয়ারিটি ঢুকানো। মডেলে নন লিনিয়ারিটি ঢুকাতে গেলে নিউরাল নেটওয়ার্ক ভালো একটা উপায়। আমরা একটা ছবি আঁকি লিনিয়ার মডেলের। তিনটে ইনপুট ফিচার। ইনপুট ফিচারের সাথে ওয়েটকে যোগ করে নিয়ে এলাম আউটপুটে। আপনার কি মনে হয় এভাবে মডেলে নন-লিনিয়ারিটি ঢুকানোর সম্ভব? আপনি বলুন।

<img src="https://raw.githubusercontent.com/raqueeb/deep_learning_book/master/start-page/linear_net.png"> 

চিত্রঃ তিনটা ইনপুট যাচ্ছে একটা নিউরাল নেটওয়ার্কে  

পরের ছবিতে আমরা একটা হিডেন লেয়ার যোগ করি। হিডেন লেয়ারের অর্থ হচ্ছে এর মধ্যে কিছু মাঝামাঝি ভ্যালু যোগ করা। আগের ইনপুট লেয়ার থেকে এই লেয়ারে তাদের ওয়েটগুলোর যোগফল পাঠিয়ে দিচ্ছে সামনের লেয়ারে। এখানে সামনে লেয়ার হচ্ছে আউটপুট। ইনপুট থেকে ওয়েট যোগ করে সেগুলোকে পাঠিয়ে দিচ্ছে আউটপুট লেয়ারে। ইংরেজিতে আমরা বলি 'ওয়েটেড সাম অফ প্রিভিয়াস নোডস'। এখনো কি মডেলটা লিনিয়ার? মডেল অবশ্যই লিনিয়ার হবে কারণ আমরা এ পর্যন্ত যা করেছি তা সব লিনিয়ার ইনপুটগুলোকেই একসাথে করেছি। নন-লিনিয়ারিটি যোগ করার মতো এখনো কিছু করিনি। 

<img src="https://raw.githubusercontent.com/raqueeb/deep_learning_book/master/start-page/1hidden.png"> 
চিত্রঃ যোগ করলাম প্রথম হিডেন লেয়ার, লিনিয়ারিটি বজায় থাকবে?

এরকম করে আমরা যদি আরেকটা হিডেন লেয়ার যোগ করি তাহলে কি হবে? নন লিনিয়ার কিছু হতে পারে? না। আমরা যতই লেয়ার বাড়াই না কেন এই আউটপুট হচ্ছে আসলে ইনপুটের একটা ফাংশন। মানে হচ্ছে ইনপুটের ওয়েট গুলোর একটা যোগফল। যাই যোগফল হোকনা কেন সবই লিনিয়ার। এই যোগফল আসলে আমাদের নন লিনিয়ার সমস্যা মেটাবে না। 

<img src="https://raw.githubusercontent.com/raqueeb/deep_learning_book/master/start-page/2hidden.png"> 
চিত্রঃ যোগ করলাম দ্বিতীয় হিডেন লেয়ার, লিনিয়ারিটি বজায় থাকবে?

একটা নন লিনিয়ার সমস্যাকে মডেল করতে গেলে আমাদের মডেলে যোগ করতে হবে নন লিনিয়ার কিছু ফাংশন। ব্যাপারটা আমাদেরকে নিজেদেরকেই ঢোকাতে হবে। সবচেয়ে মজার কথা হচ্ছে আমরা এই ইনপুটগুলোকে পাইপ করে হিডেন লেয়ারের শেষে একটা করে নন লিনিয়ার ফাংশন যোগ করে দিতে পারি। এই ছবিটা দেখুন। আমরা এক নাম্বার হিডেন লেয়ার এর পর একটা করে নন লিনিয়ার ফাংশন যোগ করে দিয়েছি যাতে সেটার আউটপুট সে পাঠাতে পারে দ্বিতীয় হিডেন লেয়ারে। এই ধরনের নন লিনিয়ার ফাংশনকে আমরা এর আগেও বলেছি অ্যাক্টিভেশন ফাংশন। 

আমাদের পছন্দের অ্যাক্টিভেশন ফাংশন হচ্ছে রেল্যু, রেকটিফাইড লিনিয়ার ইউনিট অ্যাক্টিভেশন ফাংশন। কাজে এটা স্মার্ট, অনেকের থেকে ভালো আর সে কারণে এর ব্যবহার অনেক বেশি। ভুল হবার চান্স কম। ডায়াগ্রাম দেখলেই বুঝতে পারবেন - যদি ইনপুট শূন্য হয় তাহলে আউটপুট ০ আর ইনপুটের মান ০ থেকে বেশি হয় তাহলে সেটার আউটপুটে যাবে পরের লেয়ারে যাওয়ার জন্য। 
<img src="https://raw.githubusercontent.com/raqueeb/deep_learning_book/master/start-page/relu.png"> 
চিত্রঃ ইনপুটের সবকিছুর 'ওয়েটেড সাম' থেকে ০ আসলে সেটাই থাকবে বেশি হলে ১, মানে পরের লেয়ারে পার  

আমরা যখন অ্যাক্টিভেশন ফাংশন যোগ করব তার সঙ্গে বেশি বেশি লেয়ার মডেলে ভালো কাজ করে। একটা নন লিনিয়ারিটি আরেকটা নন লিনিয়ারিটির উপর থাকাতে মডেল অনেক কমপ্লেক্স সম্পর্ক ধরতে পারে ইনপুট থেকে আউটপুট পর্যন্ত। মডেলের প্রতিটা লেয়ার তার অংশে কমপ্লেক্স জিনিসগুলো এক্সট্রাক্ট করতে পারে সেই কারণেই। অ্যাক্টিভেশন ফাংশন ছাড়া একটা নিউরাল নেটওয়াক আসলে আরেকটা লিনিয়ার রিগ্রেশন মডেল। এদিকে অ্যাক্টিভেশন ফাংশনে ব্যাক-প্রপাগেশন সম্ভব করে কারণ এর গ্রেডিয়েন্ট, তার এরর, ওয়েট এবং বায়াসকে আপডেট পাঠায়। শুরুর দিকের অ্যাক্টিভেশন ফাংশন হচ্ছে সিগময়েড। যার কাজ হচ্ছে যাই পাক না কেন সেটাকে ০ অথবা ১ এ পাঠিয়ে দেবে।   
<img src="https://raw.githubusercontent.com/raqueeb/deep_learning_book/master/start-page/sigmoid.png"> 
চিত্রঃ ইনপুটের সবকিছুর 'ওয়েটেড সাম' পাল্টে দেবে ০ থেকে ১ এর মধ্যে এই সিগময়েড 

আবারো বলছি - সিগময়েড অ্যাক্টিভেশন ফাংশন লেয়ারগুলোর ইনপুটের/আউটপুটের যোগফলকে ০ অথবা ১ এর মধ্যে ফেলে দেয়। হয় এসপার না হলে ওসপার। লিনিয়ারিটির কোন স্কোপ থাকবে না। একটা ছবি দেখুন। ইকুয়েশন সহ। 

আমার আরেকটা পছন্দের অ্যাক্টিভেশন ফাংশন হচ্ছে সফটম্যাক্স। এটা সাধারণত আমরা ব্যবহার করি দুইয়ের বেশি ক্লাসিফিকেশন সমস্যা হ্যান্ডেল করতে। সিগময়েড ভালো যখন আমরা দুটো ক্লাসিফিকেশন করি, তবে মাল্টিপল ক্লাসিফিকেশন এর জন্য সফটম্যাক্স অসাধারণ। যখন আউটপুট লেয়ার একটার সাথে আরেকটা 'মিউচুয়ালি এক্সক্লুসিভ' হয়, মানে কোন আউটপুট একটার বেশী আরেকটার ঘরে পড়বে না তাহলে সেটা 'সফটম্যাক্'স হ্যান্ডেল করবে। আমাদের যেকোনো শার্ট অথবা হাতে লেখা MNIST ইমেজগুলো যেকোন একটা ক্লাসেই পড়বে তার বাইরে নয়।  

## ট্রেনিং/টেস্ট স্প্লিট করা 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size = 0.3, random_state=0)
```

## একটা লজিস্টিক রিগ্রেশন করি 


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)



## একটা প্লটিং দেখি 


```python
hticks = np.linspace(-2, 2, 101)
vticks = np.linspace(-2, 2, 101)
aa, bb = np.meshgrid(hticks, vticks)
ab = np.c_[aa.ravel(), bb.ravel()]

c = lr.predict(ab)
cc = c.reshape(aa.shape)

ax = df.plot(kind='scatter', c='target', x='lat', y='lon', cmap='bwr')
ax.contourf(aa, bb, cc, cmap='bwr', alpha=0.5)
```




    <matplotlib.contour.QuadContourSet at 0x7f618e7f2f28>




![png](output_22_1.png)


## ২. একদম বেসিক নিউরাল নেটওয়ার্কে দেখি
১টা ইনপুট লেয়ার, ১টা নিউরন, ১টা আউটপুট


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_dim=2, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(tf.keras.optimizers.SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
result = model.fit(X_train, y_train, epochs=20, validation_split=0.1)
```

    Train on 945 samples, validate on 105 samples
    Epoch 1/20
    945/945 [==============================] - 1s 1ms/sample - loss: 0.6600 - accuracy: 0.5524 - val_loss: 0.6323 - val_accuracy: 0.6381
    Epoch 2/20
    945/945 [==============================] - 0s 107us/sample - loss: 0.6104 - accuracy: 0.6582 - val_loss: 0.6159 - val_accuracy: 0.6381
    Epoch 3/20
    945/945 [==============================] - 0s 107us/sample - loss: 0.5916 - accuracy: 0.6582 - val_loss: 0.6003 - val_accuracy: 0.6381
    Epoch 4/20
    945/945 [==============================] - 0s 100us/sample - loss: 0.5757 - accuracy: 0.6476 - val_loss: 0.5988 - val_accuracy: 0.6381
    Epoch 5/20
    945/945 [==============================] - 0s 109us/sample - loss: 0.5677 - accuracy: 0.6434 - val_loss: 0.5819 - val_accuracy: 0.6381
    Epoch 6/20
    945/945 [==============================] - 0s 100us/sample - loss: 0.5596 - accuracy: 0.6169 - val_loss: 0.5766 - val_accuracy: 0.6381
    Epoch 7/20
    945/945 [==============================] - 0s 98us/sample - loss: 0.5547 - accuracy: 0.6487 - val_loss: 0.5731 - val_accuracy: 0.6381
    Epoch 8/20
    945/945 [==============================] - 0s 122us/sample - loss: 0.5511 - accuracy: 0.6582 - val_loss: 0.5688 - val_accuracy: 0.6381
    Epoch 9/20
    945/945 [==============================] - 0s 174us/sample - loss: 0.5450 - accuracy: 0.6307 - val_loss: 0.5659 - val_accuracy: 0.6381
    Epoch 10/20
    945/945 [==============================] - 0s 149us/sample - loss: 0.5433 - accuracy: 0.6275 - val_loss: 0.5637 - val_accuracy: 0.6381
    Epoch 11/20
    945/945 [==============================] - 0s 110us/sample - loss: 0.5395 - accuracy: 0.6307 - val_loss: 0.5613 - val_accuracy: 0.6381
    Epoch 12/20
    945/945 [==============================] - 0s 130us/sample - loss: 0.5382 - accuracy: 0.6497 - val_loss: 0.5597 - val_accuracy: 0.6381
    Epoch 13/20
    945/945 [==============================] - 0s 121us/sample - loss: 0.5369 - accuracy: 0.6402 - val_loss: 0.5581 - val_accuracy: 0.6381
    Epoch 14/20
    945/945 [==============================] - 0s 103us/sample - loss: 0.5352 - accuracy: 0.6233 - val_loss: 0.5563 - val_accuracy: 0.6381
    Epoch 15/20
    945/945 [==============================] - 0s 112us/sample - loss: 0.5335 - accuracy: 0.6381 - val_loss: 0.5559 - val_accuracy: 0.6381
    Epoch 16/20
    945/945 [==============================] - 0s 178us/sample - loss: 0.5323 - accuracy: 0.6476 - val_loss: 0.5545 - val_accuracy: 0.3905
    Epoch 17/20
    945/945 [==============================] - 0s 130us/sample - loss: 0.5309 - accuracy: 0.5937 - val_loss: 0.5549 - val_accuracy: 0.6381
    Epoch 18/20
    945/945 [==============================] - 0s 108us/sample - loss: 0.5307 - accuracy: 0.5989 - val_loss: 0.5605 - val_accuracy: 0.6381
    Epoch 19/20
    945/945 [==============================] - 0s 135us/sample - loss: 0.5309 - accuracy: 0.6360 - val_loss: 0.5523 - val_accuracy: 0.6381
    Epoch 20/20
    945/945 [==============================] - 0s 129us/sample - loss: 0.5299 - accuracy: 0.6497 - val_loss: 0.5499 - val_accuracy: 0.6381


## অ্যাক্যুরেসি প্লটিং দেখি 


```python
pd.DataFrame(result.history).plot(ylim=(-0.05, 1.05))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f618e57ce10>




![png](output_26_1.png)


## ডিসিশন বাউন্ডারি কি ঠিক হলো?


```python
hticks = np.linspace(-2, 2, 101)
vticks = np.linspace(-2, 2, 101)
aa, bb = np.meshgrid(hticks, vticks)
ab = np.c_[aa.ravel(), bb.ravel()]

c = model.predict(ab)
cc = c.reshape(aa.shape)

ax = df.plot(kind='scatter', c='target', x='lat', y='lon', cmap='bwr')
ax.contourf(aa, bb, cc, cmap='bwr', alpha=0.5)
```




    <matplotlib.contour.QuadContourSet at 0x7f617c71a048>




![png](output_28_1.png)



```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(tf.keras.optimizers.SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(X_train, y_train, epochs=20, validation_split=0.1)
```

    Train on 945 samples, validate on 105 samples
    Epoch 1/20
    945/945 [==============================] - 1s 1ms/sample - loss: 0.6587 - accuracy: 0.6296 - val_loss: 0.6551 - val_accuracy: 0.6381
    Epoch 2/20
    945/945 [==============================] - 0s 134us/sample - loss: 0.6396 - accuracy: 0.6582 - val_loss: 0.6385 - val_accuracy: 0.6381
    Epoch 3/20
    945/945 [==============================] - 0s 104us/sample - loss: 0.6146 - accuracy: 0.6550 - val_loss: 0.6101 - val_accuracy: 0.6381
    Epoch 4/20
    945/945 [==============================] - 0s 104us/sample - loss: 0.5638 - accuracy: 0.6519 - val_loss: 0.5562 - val_accuracy: 0.6381
    Epoch 5/20
    945/945 [==============================] - 0s 131us/sample - loss: 0.4867 - accuracy: 0.7852 - val_loss: 0.4737 - val_accuracy: 0.6476
    Epoch 6/20
    945/945 [==============================] - 0s 178us/sample - loss: 0.3971 - accuracy: 0.8561 - val_loss: 0.4102 - val_accuracy: 0.8571
    Epoch 7/20
    945/945 [==============================] - 0s 132us/sample - loss: 0.3504 - accuracy: 0.8815 - val_loss: 0.3727 - val_accuracy: 0.8476
    Epoch 8/20
    945/945 [==============================] - 0s 101us/sample - loss: 0.3158 - accuracy: 0.8910 - val_loss: 0.3412 - val_accuracy: 0.8762
    Epoch 9/20
    945/945 [==============================] - 0s 109us/sample - loss: 0.2875 - accuracy: 0.8984 - val_loss: 0.3143 - val_accuracy: 0.8667
    Epoch 10/20
    945/945 [==============================] - 0s 170us/sample - loss: 0.2523 - accuracy: 0.9164 - val_loss: 0.2566 - val_accuracy: 0.9238
    Epoch 11/20
    945/945 [==============================] - 0s 142us/sample - loss: 0.1984 - accuracy: 0.9439 - val_loss: 0.1927 - val_accuracy: 0.9333
    Epoch 12/20
    945/945 [==============================] - 0s 109us/sample - loss: 0.1465 - accuracy: 0.9746 - val_loss: 0.1356 - val_accuracy: 1.0000
    Epoch 13/20
    945/945 [==============================] - 0s 193us/sample - loss: 0.1094 - accuracy: 0.9947 - val_loss: 0.1040 - val_accuracy: 1.0000
    Epoch 14/20
    945/945 [==============================] - 0s 133us/sample - loss: 0.0865 - accuracy: 0.9968 - val_loss: 0.0850 - val_accuracy: 1.0000
    Epoch 15/20
    945/945 [==============================] - 0s 120us/sample - loss: 0.0712 - accuracy: 0.9979 - val_loss: 0.0702 - val_accuracy: 1.0000
    Epoch 16/20
    945/945 [==============================] - 0s 111us/sample - loss: 0.0605 - accuracy: 0.9989 - val_loss: 0.0609 - val_accuracy: 1.0000
    Epoch 17/20
    945/945 [==============================] - 0s 103us/sample - loss: 0.0527 - accuracy: 0.9989 - val_loss: 0.0533 - val_accuracy: 1.0000
    Epoch 18/20
    945/945 [==============================] - 0s 183us/sample - loss: 0.0468 - accuracy: 0.9989 - val_loss: 0.0477 - val_accuracy: 1.0000
    Epoch 19/20
    945/945 [==============================] - 0s 155us/sample - loss: 0.0422 - accuracy: 0.9989 - val_loss: 0.0434 - val_accuracy: 1.0000
    Epoch 20/20
    945/945 [==============================] - 0s 107us/sample - loss: 0.0384 - accuracy: 0.9989 - val_loss: 0.0403 - val_accuracy: 1.0000


## একটা কনফিউশন ম্যাট্রিক্স তৈরি করি 


```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict_classes(X_test)
```


```python
cm = confusion_matrix(y_test, y_pred)

pd.DataFrame(cm,
             index=["Miss", "Hit"],
             columns=['pred_Miss', 'pred_Hit'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_Miss</th>
      <th>pred_Hit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Miss</td>
      <td>311</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Hit</td>
      <td>0</td>
      <td>139</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_score = model.evaluate(X_train, y_train, verbose=0)[1]
test_score = model.evaluate(X_test, y_test,  verbose=0)[1]

print("""Accuracy scores:
   Train:\t{:0.3}
   Test:\t{:0.3}""".format(train_score, test_score))
```

    Accuracy scores:
       Train:	0.999
       Test:	1.0


## নতুন ডিসিশন বাউন্ডারি, অল্প লেয়ারেই 


```python
hticks = np.linspace(-2, 2, 101)
vticks = np.linspace(-2, 2, 101)
aa, bb = np.meshgrid(hticks, vticks)
ab = np.c_[aa.ravel(), bb.ravel()]

c = model.predict(ab)
cc = c.reshape(aa.shape)

ax = df.plot(kind='scatter', c='target', x='lat', y='lon', cmap='bwr')
ax.contourf(aa, bb, cc, cmap='bwr', alpha=0.5)
```




    <matplotlib.contour.QuadContourSet at 0x7f615cf83da0>




![png](output_35_1.png)

