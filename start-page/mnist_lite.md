---
description: (পুরো চ্যাপ্টার একটা নোটবুক)
---

# মোবাইল অ্যাপের জন্য সাধারণ "এমনিস্ট" মডেল ট্রেনিং, টেন্সরফ্লো লাইট দিয়ে \(২\)

{% hint style="info" %}
আমরা বই পড়ছি, নোটবুক কেন পড়বো? 

নোটবুক সবসময় আপডেটেড। এই বই থেকেও। 

যেহেতু গিটবুকে নোটবুক ঠিকমতো রেন্ডার হয়না, সেকারণে গুগল কোলাব এবং গিটহাবে দেখা উচিৎ। গিটহাব লিংক: 

নিজে নিজে প্র্যাকটিস করুন: [https://github.com/raqueeb/tf\_lite\_android/blob/master/digitclassifier.ipynb](https://github.com/raqueeb/tf_lite_android/blob/master/digitclassifier.ipynb) এবং [https://nbviewer.jupyter.org/github/raqueeb/TensorFlow2/blob/master/digitclassifier.ipynb](https://nbviewer.jupyter.org/github/raqueeb/TensorFlow2/blob/master/digitclassifier.ipynb)

কোলাব লিংক: [https://colab.research.google.com/github/raqueeb/tf\_lite\_android/blob/master/digitclassifier.ipynb](https://colab.research.google.com/github/raqueeb/tf_lite_android/blob/master/digitclassifier.ipynb)
{% endhint %}

## মোবাইল অ্যাপের জন্য সাধারণ "এমনিস্ট" মডেল ট্রেনিং, টেন্সরফ্লো লাইট দিয়ে \(২\)

হালের ফ্যাশন 'এমনিস্ট' ডেটাসেট নিয়ে কাজ করলেও আমার মন পড়েছিল কিভাবে আসল 'এমনিস্ট' নিয়ে কাজ করা যায়। সেই সুযোগটা নিলাম এই মোবাইল অ্যাপ এর জন্য মডেল তৈরি করতে গিয়ে। একই কোড কাজ করবে এখানে, ফ্যাশন এমনিস্টএর মতো। শুধুমাত্র অ্যাডঅন, মডেলটাকে কনভার্ট করবো টেন্সরফ্লো লাইটে - মোবাইল ডিভাইসে চালানোর জন্য

```python
# শুধুমাত্র টেন্সরফ্লো ২.x এর জন্য
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

from __future__ import absolute_import, division, print_function, unicode_literals

# টেন্সরফ্লো ২.x এবং tf.keras
import tensorflow as tf
from tensorflow import keras

# সাহায্যকারী লাইব্রেরি
import numpy as np
import matplotlib.pyplot as plt
import random

print(tf.__version__)
```

```text
TensorFlow 2.x selected.
2.1.0-rc1
```

## ডাউনলোড করি "এমনিস্ট" ডেটাসেট

একই ৬০,০০০ এবং ১০,০০০ ট্রেনিং এবং টেস্ট ইমেজ তবে এবার হাতেলেখা ০ থেকে ৯ পর্যন্ত ইংরেজি সংখ্যা। ০ থেকে ৯ সংখ্যা মানে আগের মতো ১০টা ক্লাস।

আগের মতো ইমেজগুলো ২৮ x ২৮ পিক্সেলের গ্রেস্কেল ইমেজ। ছবি দেখুন। ![MNIST sample](https://github.com/khanhlvg/DigitClassifier/raw/master/images/mnist.png)

```python
# কেরাস থেকে এপিআই দিয়ে ডাউনলোড করে দেখাচ্ছি, টিএফডিএস নয়, ভ্যারিয়েশন
# ট্রেনিং এবং টেস্ট ডেটাসেট
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

```text
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
```

```python
# আগের মতো নর্মালাইজ করে ০ থেকে ১ এর মধ্যে রাখছি
train_images = train_images / 255.0
test_images = test_images / 255.0
print('Pixels are normalized')
```

```text
Pixels are normalized
```

## টেন্সরফ্লো মডেল তৈরি ডিজিট ক্লাসিফাই করার জন্য

একদম আগের মতো। কোন পার্থক্য় নেই।

```python
# শুরুতে মডেল আর্কিটেকচার ডিফাইন করছি
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

# মডেলকে ট্রেইন করতে হবে, তার আগে কম্পাইল
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ডিজিট ক্লাসিফিকেশন মডেল ট্রেনিং
model.fit(train_images, train_labels, epochs=5)
```

```text
Train on 60000 samples
Epoch 1/5
60000/60000 [==============================] - 11s 177us/sample - loss: 0.2655 - accuracy: 0.9264
Epoch 2/5
60000/60000 [==============================] - 4s 67us/sample - loss: 0.1035 - accuracy: 0.9708
Epoch 3/5
60000/60000 [==============================] - 4s 72us/sample - loss: 0.0759 - accuracy: 0.9785
Epoch 4/5
60000/60000 [==============================] - 4s 68us/sample - loss: 0.0626 - accuracy: 0.9818
Epoch 5/5
60000/60000 [==============================] - 4s 68us/sample - loss: 0.0540 - accuracy: 0.9839

<tensorflow.python.keras.callbacks.History at 0x7fc3f04e26a0>
```

মডেলের সামারি

```python
model.summary()
```

```text
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
reshape (Reshape)            (None, 28, 28, 1)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 12)        120       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 12)        0         
_________________________________________________________________
flatten (Flatten)            (None, 2028)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                20290     
=================================================================
Total params: 20,410
Trainable params: 20,410
Non-trainable params: 0
_________________________________________________________________
```

এখানে যে বাড়তি ডাইমেনশন আছে **None** শেপ দিয়ে প্রতিটা লেয়ারে, সেটা আসলে ব্যাচ ডাইমেনশন। যেহেতু ডেটা ব্যাচে প্রসেস হয়, সেকারণে টেন্সরফ্লো নিজে থেকে যোগ করে দেয়।

## মডেলের ইভালুয়েশন

টেস্ট ডেটাসেট দিয়ে কেমন কাজ করে?

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
```

```text
10000/10000 [==============================] - 1s 68us/sample - loss: 0.0647 - accuracy: 0.9801
Test accuracy: 0.9801
```

## কেরাস মডেল থেকে টেন্সরফ্লো লাইটে কনভার্ট করছি এখানে

ডিজিট ক্লাসিফায়ার মডেলকে টিএফ লাইট ফরম্যাটে আনছি।

```python
# কনভার্টারের ব্যবহার
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()

# মডেলের সাইজ কিলোবাইটে
float_model_size = len(tflite_float_model) / 1024
print('Float model size = %dKBs.' % float_model_size)
```

```text
Float model size = 81KBs.
```

## টেন্সরফ্লো লাইট মডেল ডাউনলোড করছি এখানে

এই মডেলটাকে ডাউনলোড করছি এখানে যাতে এটাকে অ্যান্ড্রয়েড অ্যাপে ঢুকিয়ে কাজ করতে পারি। এটাকে কপি করে রাখতে হবে দরকারী ফোল্ডারে।

mnist.tflite ফাইলটা একবারে ডাউনলোড না হলে শুধুমাত্র এই সেলটাকে আবার চালাতে হবে।

```python
# ফাইলটাকে সেভ করে রাখি এখানে mnist.tflite হিসেবে
f = open('mnist.tflite', "wb")
f.write(tflite_float_model)
f.close()

# ডাউনলোড করি mnist.tflite হিসেবে
from google.colab import files
files.download('mnist.tflite')

print('`mnist.tflite` has been downloaded')
```

```text
`mnist.tflite` has been downloaded
```

