# নিউরাল নেটওয়ার্কের নিউরাল, ওয়েট, বায়াস, অ্যাক্টিভেশন ফাংশন

> What people call \#AI is no more than finding answers to questions we know to ask. Real \#AI is answering questions we haven't dreamed of yet.
>
> ― Tom Golway, Hybrid IT and Intelligent Edge Solutions

![&#x99A;&#x9BF;&#x9A4;&#x9CD;&#x9B0;: &#x9A1;&#x9BF;&#x9AA; &#x9A8;&#x9BF;&#x989;&#x9B0;&#x9BE;&#x9B2; &#x9A8;&#x9C7;&#x99F;&#x993;&#x9DF;&#x9BE;&#x9B0;&#x9CD;&#x995; ](../.gitbook/assets/11.jpeg)

ডিপ লার্নিং নিয়ে যখন কথা বলি তখন আমাদের চোখের সামনে ভাসে 'ভীতিকর' একটার সাথে আরেকটার লেগে থাকা নিউরাল নেটওয়ার্কের স্ট্যাক। এই ছবিটার মতো। মানে একটার উপর আরেকটা নিউরাল নেটওয়ার্ক। একদম ভয় পাবেন না। এই জিনিস আপনাকে আমাকে করতে হবে না। এটা করবে মেশিন। তবে, সেটা কিভাবে হচ্ছে সেটা নিয়ে আলাপ করবো আজ। 

![&#x99A;&#x9BF;&#x9A4;&#x9CD;&#x9B0;: &#x9AE;&#x9B8;&#x9CD;&#x9A4;&#x9BF;&#x9B7;&#x9CD;&#x995;&#x9C7;&#x9B0; &#x9A8;&#x9BF;&#x989;&#x9B0;&#x9A8; &#x9B8;&#x999;&#x9CD;&#x997;&#x9C7; &#x98F;&#x995;&#x99F;&#x9BE; &#x9A1;&#x9BF;&#x9AA; &#x9A8;&#x9BF;&#x989;&#x9B0;&#x9BE;&#x9B2; &#x9A8;&#x9C7;&#x99F;&#x993;&#x9DF;&#x9BE;&#x9B0;&#x9CD;&#x995;&#x9C7;&#x9B0; &#x98F;&#x995;&#x99F;&#x9BE; &#x9A8;&#x9BF;&#x989;&#x9B0;&#x9A8; ](../.gitbook/assets/32.png)

সত্যি বলতে এই নেটওয়ার্কগুলো বেশ কয়েকটা লেয়ারের হয়ে থাকে। সবচেয়ে বড় কথা হচ্ছে এই লেয়ারগুলো তৈরি হয় 'নোড' দিয়ে। গোল গোল অংশগুলোকে আপাততঃ ধরে নিচ্ছি আমরা। নোডগুলো হচ্ছে এমন একটা জায়গা যেখানে আসল কম্পিউটেশন হয়। ঠিক আমাদের মস্তিষ্কের মত যেখানে পর্যাপ্ত পরিমাণে স্টিমুলি হলেই পরবর্তী লেয়ারে ফায়ার করে। আমাদের এখানে একটা নোড ইনপুট থেকে ডাটার সাথে তার 'করেসপন্ডিং' 'কোএফিশিয়েন্ট' যেটাকে আমরা বলি ওয়েট, যা আমাদের ইনপুটকে দরকার মতো হয় বড় বা ছোট করে। একটা নিউরনের সব এলিমেন্টকে নিয়ে তৈরি হয় একটা পারসেপ্ট্রন। 

আমাদের অ্যালগরিদমগুলো ট্রেনিং ডেটা থেকে যে কাজটা শিখতে চাচ্ছে, সেখানে এই ওয়েটগুলো ইনপুটে দরকারি 'সিগনিফিকেন্স' তৈরি করে। 'সিগনিফিকেন্স' হচ্ছে ইনপুটের সাথে এক ধরণের মাল্টিপ্লায়ার - যে ইনপুটকে বেশি বা কম করে বের করে দেয় আউটপুটে। এই আউটপুট হতে পারে পরের লেয়ারের ইনপুট। আমরা বলতে পারি এখানে কোন ইনপুটগুলো সবচেয়ে বেশি সাহায্য করছে একটা ডাটাকে ক্লাসিফাই করতে - সবচেয়ে কম 'এরর' দিয়ে। 

আমাদের এই ইনপুট এবং ওয়েটের প্রডাক্টকে যোগ করে সেটাকে পাঠানো হয় নোডের সামনে - যাকে আমরা বলছি 'অ্যাক্টিভেশন ফাংশন'। এর মধ্যে বায়াস থাকলে সেটাও হবে যোগ অ্যাক্টিভেশনের আগে। এই অ্যাক্টিভেশন ফাংশন এর কাজ হচ্ছে আমাদের পেছনের ইনপুট ডাটা আর সামনে বাড়বে কিনা সেটার একটা সিগন্যাল দেয় যা শেষ আউটকাম নির্ধারণ করে। ধরা যাক, আমরা একটা ক্লাসিফিকেশন করতে চাচ্ছি, তাহলে এই অ্যাক্টিভেশন ফাংশন এমন একটা সিগন্যাল পাঠাবে যা উত্তরের হ্যাঁ অথবা না সিগন্যাল বের করার জন্য সাহায্য করবে। যদি পেছন থেকে সিগন্যালটা সামনে এগিয়ে যায় তাহলে আমরা বলতে পারি নিউরনটা 'অ্যাক্টিভেটেড'।

![&#x99A;&#x9BF;&#x9A4;&#x9CD;&#x9B0;&#x983; &#x9B8;&#x9AC; &#x9AF;&#x9CB;&#x997;&#x9AB;&#x9B2; \(&#x9AC;&#x9BE;&#x9DF;&#x9BE;&#x9B8; &#x9B8;&#x9B9;\) + &#x985;&#x9CD;&#x9AF;&#x9BE;&#x995;&#x9CD;&#x99F;&#x9BF;&#x9AD;&#x9C7;&#x9B6;&#x9A8; &#x9B9;&#x99A;&#x9CD;&#x99B;&#x9C7; &#x98F;&#x995;&#x99F;&#x9BE; &#x9A8;&#x9BF;&#x989;&#x9B0;&#x9A8;&#x9C7; ](../.gitbook/assets/211.jpeg)

অনেক গল্প হল, এখন দুটো ছবি দেখি। আমরা যদি একটা 'নোড' এর লেয়ার দেখি, সেখানে একটা নিউরনের মত সুইচ থাকবে যা 'অন' অথবা 'অফ' হয় নেটওয়ার্ক থেকে ইনপুট পেলে। প্রতিটা লেয়ারের আউটপুটগুলো কিন্তু পাশাপাশি পরের লেয়ারের ইনপুট হিসেবে কাজ করে। ইনপুটের ডাটার সাথে তার ওয়েট সেই ইনপুটের জন্য 'সিগনিফিকেন্স' তৈরি করে। ইনপুট এবং ওয়েটের ডট প্রোডাক্টের যোগফল পাঠানো হয় অ্যাক্টিভেশন ফাংশনে।

![&#x99A;&#x9BF;&#x9A4;&#x9CD;&#x9B0;&#x983; &#x995;&#x9BF;&#x9AD;&#x9BE;&#x9AC;&#x9C7; &#x98F;&#x995;&#x9C7;&#x995;&#x99F;&#x9BE; &#x9A8;&#x9BF;&#x989;&#x9B0;&#x9A8; &#x9AC;&#x9BF;&#x9AD;&#x9BF;&#x9A8;&#x9CD;&#x9A8; &#x9B2;&#x9C7;&#x9DF;&#x9BE;&#x9B0;&#x9C7; &#x995;&#x9BE;&#x99C; &#x995;&#x9B0;&#x99B;&#x9C7; ](../.gitbook/assets/61.PNG)

{% hint style="info" %}
আপনি কি জানেন, এই নিউরাল নেটওয়াক কিন্তু ১৯৫০ সালের জিনিস? তখন, প্রসেসিং ক্ষমতা, আর ডেটা অতো না থাকায় এই জিনিস আর এগোইনি। এখন কম্পিউটিং প্রসেসিং আর বিলিয়ন বিলিয়ন ডেটার কারণে এই ম্যাজিক!
{% endhint %}

