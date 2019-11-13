# টেন্সর-ফ্লো মডেল থেকে প্রোডাকশন, টেন্সর-ফ্লো সার্ভিং

আপনি সঞ্চয়পত্র কিনতে গেলেন ব্যাংকে। সঙ্গে নিয়ে গেলেন ন্যাশনাল আই-ডি, ট্যাক্স আইডেন্টিফিকেশন নম্বর এর ফটোকপি। আগে কি হতো? আর কাগজের সত্যতা কি? সেরকম ভেরিফিকেশনের ব্যবস্থা ছিলো না। ফটোকপিই সই। অনেক অফিস ১০/১২টা ডকুমেন্ট জমা দিতে বলতেন - যেখানে আসল কোন ভেরিফিকেশন এর ব্যবস্থা নেই। 

এখন? ফিরে আসি আগের গল্পে।  

ব্যাংকের কর্মকর্তা আপনার ফটোকপি থেকে এনআইডি টুকে এক্সেস করলেন সঞ্চয়পত্র কেনার সফটওয়্যারে। সফটওয়্যার আছে বাংলাদেশ ব্যাংকে। এন্ট্রি দিতে হবে সেখানে। সেই সফটওয়্যার আবার কানেক্টেড ন্যাশনাল আই-ডি সিস্টেমের সাথে। ব্যাপারটা এমন নয় যে ব্যাংক কর্মকর্তা ওই বাংলাদেশ ব্যাংকের সফটওয়্যার থেকে ন্যাশনাল আই-ডি সিস্টেমে ঢুকে আপনাকে সনাক্ত করছেন। বরং, বাংলাদেশ ব্যাংকের সফটওয়্যার একটা সিস্টেমে ন্যাশনাল আই-ডি সিস্টেমের সাথে যুক্ত থাকায় ব্যাংকের সফটওয়্যার সনাক্ত করতে পারছে আপনাকে। 

এর পাশাপাশি এটা ভেরিফিকেশন নিয়ে আসছে ট্যাক্স ইডেন্টিফিকেশন নাম্বার \(টিন\) থেকে। একটা সফটওয়্যার যোগাযোগ করছে তিনটা আলাদা সফটওয়্যারের সাথে। এই ধরনের সিস্টেম থেকে সিস্টেমের যুক্ত থাকার ব্যবস্থা হচ্ছে অ্যাপ্লিকেশন প্রোগ্রামিং ইন্টারফেস \(এপিআই\) দিয়ে। এই যোগাযোগ হচ্ছে মেশিন থেকে মেশিনে। মানুষের মুখ দেখাদেখি মানে মুখ কালাকালি নেই। সেই জন্য সিঙ্গাপুরের এয়ারপোর্টে ভিসা বুথে চকলেট থাকে। সব চেক হচ্ছে সিস্টেম থেকে। আমরা চকলেট খাবো। 

পুরো পৃথিবী চলছে এই এপিআই দিয়ে। এখন সবাই চলে গেছে সিঙ্গেল সাইন-ইনে। পেছনে "ইউনিফাইড এলড্যাপ"। হয় গুগল, ফেইসবুক অথবা আমাদের মতো মানুষদের জন্য গিটহাব। এই এক্সেসের জন্য সবাই ব্যবহার করছে এপিআই। প্রতিটা সার্ভিস একে ওপরের সাথে কথা বলছে এই জিনিস দিয়ে। সবাই জানে একে অপরকে কিভাবে এক্সেস করতে হবে আগে থেকেই। বেশিরভাগ এপিআই ডকুমেন্ট পাবলিশ করা আছে ওপেন ওয়েবে। 

আপনার অ্যাপ্লিকেশনে লাগবে গুগল ম্যাপ। গুগলই বা কেন আপনাকে এক্সেস দেবে পুরো সিস্টেমে? সেকারণে যতোটুকু দরকার ততোটুকু এক্সেস ডিফাইন করা আছে এপিআইতে। আমি নিজেও দেশের সবচেয়ে বড় কয়েকটা সিস্টেমের সাথে কাজ করি বলে বলতে পারি সিস্টেমগুলোর মধ্যে কথা বলতে দুটো সিস্টেমের মধ্যে 'এগ্রিড' প্রটোকলের এপিআই ছাড়া গতি নেই। আমাদেরও একই অবস্থা। 

মেশিন লার্নিং নিয়ে অনেক পরীক্ষা নিরীক্ষা করলাম। এখন প্রোডাকশনে যাবার পালা। পরীক্ষা নিরীক্ষা করা সোজা, কিন্তু প্রোডাকশন এনভায়রনমেন্ট কঠিন। তবে সেটা সোজা করার জন্য টেন্সর ফ্লো নিয়ে এসেছে "টেন্সর-ফ্লো সার্ভিং"। সার্ভিং এর কাজ কি? এটা একটা প্রসেস যার মাধ্যমে 'ট্রেইনড' মেশিন লার্নিং মডেলকে অন্য সিস্টেমের কাছে খুলে দেয়া যাতে তারা প্রেডিকশন 'কোয়েরি' করতে পারে। মডেলের প্রেডিকশনকে সার্ভ করতে পারা, প্রোডাকশন লেভেলে। 

আমরা যখন এই সার্ভিংকে প্রোডাকশন নিয়ে যাবো, তখন আমরা সেটাকে খুব তাড়াতাড়ি তৈরি করা যায় সেটা খেয়াল রাখতে হবে। এই যুগে সিস্টেমটা অবশ্যই ভার্চুয়াল হবে, তবে সেটা কতোটা আলাদা, এবং সিকিউরড সেটা মাথায় রাখলে কাজ সহজ হয়। তবে, মেশিন লার্নিং মডেলকে সহজে সার্ভ করতে হবে আমাদের লাগবে "টেন্সর-ফ্লো সার্ভিং" এবং "ডকার"। ডকার নিয়ে আলাপ হয়েছে আগে। এখন দরকার "টেন্সর-ফ্লো সার্ভিং"এর ডকার ইমেজ। 

###  "টেন্সর-ফ্লো সার্ভিং" এবং ডকার

আমরা দুটো ছোট্ট উদাহরণ দেখবো। "টেন্সর-ফ্লো সার্ভিং" এ দুটো উদাহরণ দেয়া আছে আমাদের প্র্যাকটিসের জন্য। একটা "Half Plus Two", মানে আমাদের প্রেডিক্ট করতে হবে  x এর ভ্যালু যখন এর ইকুয়েশন 0.5 \* x + 2 ডেটা দেয়া আছে। এর অর্থ হচ্ছে ডেটা হিসেবে x এর ভ্যালু ২ দিয়ে আমাদের প্রেডিকশন ৩ আসার কথা। আরেকটা মডেল আছে "Half Plus Three", সেটার কাহিনী একই। ইকুয়েশন 0.5 \* x + 3 দিয়ে ডেটা জেনারেট করা আছে। কতো মজার জিনিস আছে এখানে!   

১. আমাদের যেহেতু ডকার ইনস্টল করা আছে, এখন দরকার লেটেস্ট "টেন্সর-ফ্লো সার্ভিং"এর ডকার ইমেজ। চালু করি উইন্ডোজে পাওয়ার শেল। 

```text
PS C:\Users\Test> docker pull tensorflow/serving

Using default tag: latest
latest: Pulling from tensorflow/serving
22e816666fd6: Pull complete
Digest: sha256:091c1d0440815e250114a6d0232ad3cb1d320c64b1ebb75ed8a80184fc25482d   
Status: Downloaded newer image for tensorflow/serving:latest
docker.io/tensorflow/serving:latest
```

২. এই কমান্ডটা একটা ছোটোখাটো "টেন্সর-ফ্লো সার্ভিং" এর ডকার ইমেজ ইনস্টল করবে এখানে। আমাদের সার্ভিং এর ইমেজ কিভাবে যোগাযোগ করবে? কি কি লাগবে?

1. আমাদের হোস্টে একটা পোর্ট খোলা থাকতে হবে সার্ভিস দেবার জন্য। আমরা দুধরণের সার্ভিস খুলতে পারি। একটা gRPC রিমোট প্রসিডিউর কল এপিআই - পোর্ট ৮৫০০ \(অথবা আপনার ইচ্ছেমতো\), নতুন এসেছে। ডেভেলপ করেছিল গুগল। আরেকটা পুরানো REST এপিআই, এখনো অসাধারণ। সবাই ব্যবহার করে। খোলা থাকবে পোর্ট ৮৫০১, অথবা আপনার পছন্দের পোর্টে। আমরা উদাহরণে দুটো এপিআই খোলা রাখবো। 
2. আমাদের ট্রেইনড মডেল, যা থাকবে SavedModel হিসেবে। 
3. মডেলের একটা নাম যাকে ক্লায়েন্ট ধরে ডাকবে। 

৩. মডেলকে নিয়ে আসি "টেন্সর-ফ্লো সার্ভিং" এর গিটহাব রিপোজিটরি থেকে। গিট থেকে ক্লোন করি। গিট না থাকলে একটা গিট পোর্টেবল ডাউনলোড করে চালাই কাজ।   

```text
PS C:\Users\Test> mkdir -p tfserving

    Directory: C:\Users\Test

Mode                LastWriteTime         Length Name
----                -------------         ------ ----
d-----       11/13/2019   8:47 PM                tfserving

PS C:\Users\Test> cd .\tfserving\
PS C:\Users\Test\tfserving>
```

৪. গিট ব্যাশ চালাই। তবে পাওয়ার শেল থেকেও চালানো যায়। তবে আজকে Git Bash দিয়ে শুরু। 

```text
$ git clone https://github.com/tensorflow/serving

Cloning into 'serving'...
remote: Enumerating objects: 54, done.
remote: Counting objects: 100% (54/54), done.
remote: Compressing objects: 100% (52/52), done.
Receiving objects:  48% (9475/19738), 3.61 MiB | 195.00 KiB/s
remote: Total 19738 (delta 36), reused 20 (delta 2), pack-reused 19684
Receiving objects: 100% (19738/19738), 5.22 MiB | 178.00 KiB/s, done.
Resolving deltas: 100% (14964/14964), done.
```

৫. এখন পাওয়ার শেলে ডকার চালাবো। একটু সিনট্যাক্স বুঝে নেই। আমাদের কন্টেইনারে কি চালাবো? যখন আমাদের সার্ভিং ইমেজ tensorflow\_model\_server, তখন কি হবে? দুটো পোর্ট খুললাম, দুই এপিআই এর জন্য। সাথে মডেলের নাম, আর মডেলটা আছে কোথায়?

```text
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=my_model --model_base_path=/models/my_model
```

আর ডকার দিয়ে চালাতে গেলে? এখানে মাউন্টিং এর প্রশ্ন আসবে। বাইন্ডিং কোথায় করবে, আর মডেলের নাম দিতে হবে। এখন আসল ডিরেক্টরি নাম দেই। পাওয়ার শেলে। আমরা gRPC পোর্ট খুলে দিয়েছি ৮৫০০তে। 

```text
PS E:\git_portable> docker run -t --rm -p 8501:8501 -v "E:\git_portable\serving\tensorflow_serving\servables\tensorflow\testdata\saved_model_half_plus_two_cpu:/models/half_plus_two" -e MODEL_NAME=half_plus_two tensorflow/serving

2019-11-10 07:11:17.037045: I tensorflow_serving/model_servers/server.cc:85] Building single TensorFlow model file config:  model_name: half_plus_two model_base_path: /models/half_plus_two
2019-11-10 07:11:17.037797: I tensorflow_serving/model_servers/server_core.cc:462] Adding/updating models.
2019-11-10 07:11:17.037861: I tensorflow_serving/model_servers/server_core.cc:573]  (Re-)adding model: half_plus_two
2019-11-10 07:11:17.158245: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: half_plus_two version: 123}
2019-11-10 07:11:17.158435: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: half_plus_two version: 123}
2019-11-10 07:11:17.158496: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: half_plus_two version: 123}
2019-11-10 07:11:17.158573: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /models/half_plus_two/00000123
2019-11-10 07:11:17.170610: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2019-11-10 07:11:17.172642: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-10 07:11:17.212202: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:202] Restoring SavedModel bundle.
2019-11-10 07:11:17.230431: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:151] Running initialization op on SavedModel bundle at path: /models/half_plus_two/00000123
2019-11-10 07:11:17.236016: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:311] SavedModel load for tags { serve }; Status: success. Took 77445 microseconds.
2019-11-10 07:11:17.237262: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:105] No warmup data file found at /models/half_plus_two/00000123/assets.extra/tf_serving_warmup_requests
2019-11-10 07:11:17.247605: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: half_plus_two version: 123}
2019-11-10 07:11:17.250931: I tensorflow_serving/model_servers/server.cc:353] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2019-11-10 07:11:17.252948: I tensorflow_serving/model_servers/server.cc:373] Exporting HTTP/REST API at:localhost:8501 ... 
```

ডকারের ইমেজ দেখি। 

```text
PS E:\git_portable> docker ps

CONTAINER ID        IMAGE                                                    COMMAND                  CREATED             STATUS              PORTS                              NAMES
c6e0b94c10b9        tensorflow/tensorflow:2.0.0-py3-jupyter-pandas-sklearn   "bash -c 'source /et…"   2 hours ago         Up 2 hours          0.0.0.0:8888->8888/tcp             tensorflow2
c0a477f441fb        tensorflow/serving                                       "/usr/bin/tf_serving…"   2 hours ago         Up 2 hours          8500/tcp, 0.0.0.0:8501->8501/tcp   hungry_robinson
```

৬. এখন উইন্ডোজ কমান্ড প্রম্পটে যাই। একটা প্রেডিকশন কোয়েরি করি। শুরুতে দেখি মডেল ঠিকমতো আছে কিনা দেখি কার্ল দিয়ে। x এর ভ্যালু ১, ৪, ৫ এবং ৯ হলে প্রেডিকশন কি হবে? উইন্ডোজে এস্কেপ ক্যারেক্টার নিয়ে সমস্যা হতে পারে। প্রেডিকশন রেজাল্ট দেখুন। ২.৫, ৪, ৪.৫ এবং ৬.৫। আপনি কার্ল দিয়ে REST এপিআই কল করেছেন। 

```text
(মডেলের স্ট্যাটাস দেখি REST এপিআই দিয়ে)

C:\Users\Test>curl http://127.0.0.1:8501/v1/models/half_plus_two
{
 "model_version_status": [
  {
   "version": "123",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}

(এবার প্রেডিকশন, ৪টা সংখ্যা দিয়ে)

C:\Users\Test>curl -d "{\"instances\": [1.0, 4.0, 5.0, 9.0]}" -X POST http://127.0.0.1:8501/v1/models/half_plus_two:predict
{
    "predictions": [2.5, 4.0, 4.5, 6.5
    ]
}
```

৭. এবার আপনার জন্য প্রশ্ন।  "Half Plus Three" এর জন্য কি হবে? চলে যান E:\git\_portable\serving\tensorflow\_serving\servables\tensorflow\testdata\ অংশে  \(আপনার জন্য ভিন্ন ডিরেক্টরি হতে পারে\), দেখুন এই নামে ডিরেক্টরি আছে কিনা? অবশ্যই আছে। কারণ আমরা "টেন্সর-ফ্লো সার্ভিং" এর গিটকে ক্লোন করেছিলাম। 

```text
PS E:\git_portable> docker run -t --rm -p 8507:8507 -v "E:\git_portable\serving\tensorflow_serving\servables\tensorflow\testdata\saved_model_half_plus_three:/models/half_plus_three" -e MODEL_NAME=half_plus_three tensorflow/serving
2019-11-10 16:32:41.006983: I tensorflow_serving/model_servers/server.cc:85] Building single TensorFlow model file config:  model_name: half_plus_three model_base_path: /models/half_plus_three
2019-11-10 16:32:41.007173: I tensorflow_serving/model_servers/server_core.cc:462] Adding/updating models.
2019-11-10 16:32:41.007190: I tensorflow_serving/model_servers/server_core.cc:573]  (Re-)adding model: half_plus_three
2019-11-10 16:32:41.112369: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: half_plus_three version: 123}
2019-11-10 16:32:41.112428: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: half_plus_three version: 123}
2019-11-10 16:32:41.112460: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: half_plus_three version: 123}
2019-11-10 16:32:41.112493: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /models/half_plus_three/00000123
2019-11-10 16:32:41.190027: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2019-11-10 16:32:41.190727: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-10 16:32:41.210597: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:202] Restoring SavedModel bundle.
2019-11-10 16:32:41.241783: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:151] Running initialization op on SavedModel bundle at path: /models/half_plus_three/00000123
2019-11-10 16:32:41.247985: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:311] SavedModel load for tags { serve }; Status: success. Took 135489 microseconds.
2019-11-10 16:32:41.249075: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:105] No warmup data file found at /models/half_plus_three/00000123/assets.extra/tf_serving_warmup_requests
2019-11-10 16:32:41.257380: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: half_plus_three version: 123}
2019-11-10 16:32:41.261126: I tensorflow_serving/model_servers/server.cc:353] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2019-11-10 16:32:41.262500: I tensorflow_serving/model_servers/server.cc:373] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...
```

সব ঠিক আছে। তবে ছোট্ট একটা সমস্যা আছে। কি বলুনতো?



