import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re

# 下載必要的 NLTK 資料包
def download_nltk_resources():
    resources = [
        'punkt', 'stopwords', 'wordnet', 'omw-1.4', 
        'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'punkt_tab'
    ]
    for res in resources:
        try:
            if res == 'wordnet':
                nltk.data.find('corpora/wordnet')
            elif 'tagger' in res:
                nltk.data.find(f'taggers/{res}')
            else:
                nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res)

download_nltk_resources()

def get_wordnet_pos(treebank_tag):
    """將 NLTK 的詞性標籤轉換為 WordNet 格式"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    """
    執行完整的文字預處理：
    1. Regex 清理 -> 2. 標記化 -> 3. POS Tagging -> 4. 停用詞移除 -> 5. 詞形還原
    6. [新功能] 移除第一個 Token
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 移除數字與特殊符號 (只留英文與空白)
    text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    text_cleaned = text_cleaned.lower()
    
    # 2. 標記化
    tokens = word_tokenize(text_cleaned)
    
    # 3. 詞性標註
    pos_tags = pos_tag(tokens)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = []
    
    for word, tag in pos_tags:
        # 4. 移除停用詞
        if word not in stop_words:
            wn_pos = get_wordnet_pos(tag)
            # 5. 詞形還原
            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            processed_tokens.append(lemma)
    
    # ==========================================
    # 6. [新功能] 移除第一個 Token
    # ==========================================
    if len(processed_tokens) > 0:
        # 使用切片 (slicing) 去掉第 0 個元素，保留剩下的
        processed_tokens = processed_tokens[1:]
            
    return " ".join(processed_tokens)

def process_csv(input_file, output_file):
    try:
        print(f"正在讀取檔案：{input_file} ...")
        df = pd.read_csv(input_file)
        
        '''
        if len(df.columns) < 3:
            print("錯誤：欄位不足 3 欄")
            return
        '''

        print("正在處理文本 (包含移除首字)...")
        # 直接覆蓋第 B 欄
        df.iloc[:, 1] = df.iloc[:, 1].apply(preprocess_text)
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"處理完成！已輸出至：{output_file}")
        
    except Exception as e:
        print(f"錯誤：{e}")

if __name__ == "__main__":
    # 請在此修改您的輸入與輸出檔名
    input_csv = "augmented_backtrans_with_id.csv"    # 原始檔案
    output_csv = "augmented_preprocessedtextemojireplacement.csv"  # 處理後的檔案
    
    process_csv(input_csv, output_csv)