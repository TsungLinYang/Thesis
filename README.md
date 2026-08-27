# Thesis

---

## Overview

本 Repository 為碩士論文的完整程式碼，主要研究在多模態反諷偵測中，取得分類精準度與運算效率的最佳折衷，並提升模型可解釋性。

---

## DataSet

本實驗資料集取自於https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

---

## BERT
在本資料夾所使用的方法為文字替換方法，亦即將表情符號轉換成對應文字描述。使用Tweets公開資料集，並依序執行extract_csv.py、data_augmentation.py、preprocessing.py、emoji_replacement.py，並將最後輸出的.csv檔讀入至BERT.py

---

## CLEN_BERT
在本資料夾所使用的方法為本研究提出的多模態混合架構。使用Tweets公開資料集，並依序執行extract_csv.py、separate_emoji.py、data_augmentation.py、preprocessing.py，並將最後輸出的.csv檔讀入至CLN_BERT_vader.py

---

### CLN_BERT_vader.py
在本程式碼中，需在程式中決定是否開啟vader提前攔截機制，並且須選擇語言模型如下:

- bert 
- albert 
- distilbert 
- electra 
- tinybert

並選擇train模式、analysis模式與external test模式，train模式將讀入資料，並且訓練得到訓練權重。Analysis模式則將train模式得到的權重，在testing data進行推論速度與準確率分析，而external test模式則可利用現有權重，將外部資料進行分析。
