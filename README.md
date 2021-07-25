# 👶 Hate-Speech-Guard for Children
<img src = "https://img.shields.io/badge/ProjectType-TeamProject-orange?style=flat-square"> <img src = "https://img.shields.io/badge/Language-Javascript-critical?style=flat-square&logo=Javascript"> <img src = "https://img.shields.io/badge/Language-Python-critical?style=flat-square&logo=Python"> <img src = "https://img.shields.io/badge/Tools-VScode-brightgreen?style=flat-square&logo=VisualStudioCode"> <img src = "https://img.shields.io/badge/Tools-Pycharm-brightgreen?style=flat-square&logo=Pycharm"><br>
> HSG is an **Crome/Edge extension** that detects hate speech in real time on web pages. 
<br>
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/125822586-a78468bb-f313-43c8-b67e-f742925eee0e.png"> </p> <br>

### <p align="center"> 🔥 2021 MicroSoft Imagine Cup Semifinals 🔥  </p>
### <p align="center">  🔥 Category Feature - Lifestyle, Winner, World Finalists 🔥  </p> <br><br><br>


##  👨‍👧‍👧  Team Info 
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/125805272-d2b8e4e6-051e-42b4-bdde-acd1327582aa.png" width = "80%" > </p>
We are Team En#Plane. We met at the university programming club En#.<br>We hope HSG will have a good influence on children, parents and global. <br><br>

##  💡  Background

### 1. What is ‘Hate Speech’?
**Hate Speech** refers to statements that contain prejudices against certain groups, deliberate slander, threats, and incitement aimed at inciting violence.
### 2. More exposed children to the hate speech since COVID-19.
Exposure to hate speech is an important factor that can affect the emotions of growing children. Children live in numerous hateful expressions 	even though they know themselves or without their knowledge. This exposure to hate speech has become even more serious as online activities have 		increased in the current COVID-19 era.
### 3. The problem of hate speech
Exposure to hate speech adversely affects 	children's emotional development and value formation.<br><br>

HSG seeks to present with the functions of the next chapter a new 		method of education that can recognize, prevent, and cope with discrimination in freedom 	of expression and opinion. When children are defenselessly exposed to hate speech, they recognize it and help them 	cope with it. HSG filters and removes web text and video and provides them to users 		through an extension program. By using HSG, it will be possible to minimize the exposure 	of children's hate speech and help children to form correct values and emotions.<br><br>

##  📝  Features

#### HSG is the Web Extension Program. When you download the program to the web browser you use, it will run automatically detect hate speech.

### 1. Hate Speech Detection for Text
HSG inspects the text in the web site in real time. If it is determined that some text 		contains hate speech, HSG closely analyzes the text. The analysis results show the words 	that contributed greatly to the judgment of hate speech. Using this result, text masking 		is performed win the web site.
### 2. Hate Speech Detection for Video 
HSG checks the Hate Speech of the video in real time. First, HSG finds the text in the 		video. If the text is determined to be hate speech, store the found time and duration. 		Then, video masking conversion within the web site is done using the stored time and 		duration.
### 3. Option Selection
HSG considers the degree of user freedom. The option page helps parents to discuss with their child how to recognize and deal with the hate speech through the selection of users (mute - not mute / display - no notification). This increases the effectiveness of education on hate speech recognition.<br><br>


##  📚  Stack & Library
+ Edge/Crome Extenstions
+ Azure
+ TensorFlow <br><br>

##  🖥️  Preview
### 1. Hate Speech Detection for Text
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/125811475-d91dcbd9-09db-4d9e-9863-25d994b77b5c.png"> </p>

### 2. Hate Speech Detection for Video 
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/125811574-6d54f72d-9e07-4b6b-9511-669c2ed33a99.png"width = "75%"> </p>

### 3. Option Selection
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/125812562-7628185d-d17d-4d21-9759-1f64040c307d.png" width = "73%"> </p><br>

##  🛠️  Architecture
<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/125813977-cd1cf9f7-c20a-48f2-95df-eb0d5f285863.png"> </p><br>

When a user searches the 		web, it brings text. The text enters Azure's virtual machine server and determines 		whether it is a hate speech in a trained Python model. If the sentence is determined to 		be "hate speech", masking the sentence according to the option selected by the user. 		Filtered text/video is shown on the user's web browser.<br><br>

<p align="center"> <img src = "https://user-images.githubusercontent.com/64072741/125815205-2d7076e3-1987-4115-808f-297a54049ee6.png" width = "80%"> </p>

The deep learning model we're trying to design is a simple task defined by a binary classification(hate or not hate), and had to ensure minimal response speed and stability in 	order to avoid compromising user experience. Therefore, rather than using heavy SOTA models such as Bert, 
**We tried to design a neural net structure that could maintain a relatively light level of scale and computational complexity.** As a result, **we adopted the Bi-LSTM model** because the Bi-LSTM model showed the highest accuracy and moderate processing speed. Using the Bi-LSTM Model, a hate speech judgment is made for some sentences. 
<br><br>
For sentences that are judged to be hate speech, We used **the Lime(Local Interpretable Model-Agnostic Explanation) technique** to extract sound masking time and sentences 		that hate speech filtered. The Lime technique enables precise analysis of the evaluated 		sentences. Through the Lime technique, we can see why the sentence was judged as a hate speech 	and how much some words contributed to the result. Using this, we were able to identify 	the word portion to filter in sentence, and extract the time to filter from the video.<br><br>

## 🎓 I Learned
+ AI Model Development
+ Crome/Edge Extension Development
+ Web Front Development<br><br>

## 🔍 More
https://techcommunity.microsoft.com/t5/student-developer-blog/2021-imagine-cup-category-feature-lifestyle/ba-p/2200906<br>
https://news.joins.com/article/24017473<br>
https://www.donga.com/news/article/all/20210322/106017354/1
