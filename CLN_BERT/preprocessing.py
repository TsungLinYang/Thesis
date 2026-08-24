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

def preprocess_text(text, remove_sym=True, remove_sw=True, apply_lem=True, remove_first_token=False):
    """
    參數化預處理函式，用於控制消融實驗的開關：
    - remove_sym: 移除符號與數字 (Configuration C 控制)
    - remove_sw: 移除停用詞 (Configuration B 控制)
    - apply_lem: 詞形還原 (Configuration A 控制)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 移除數字與特殊符號 (-SYM 開關)
    if remove_sym:
        text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    else:
        text_cleaned = text  # 保留標點符號與數字
        
    # 一律轉小寫
    text_cleaned = text_cleaned.lower()
    
    # 2. 標記化
    tokens = word_tokenize(text_cleaned)
    
    # 3. 詞性標註 (如果是為了詞形還原才需要)
    if apply_lem:
        pos_tags = pos_tag(tokens)
    else:
        # 不做詞形還原時，為了迴圈格式統一，手動補上 None
        pos_tags = [(word, None) for word in tokens]
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = []
    
    for word, tag in pos_tags:
        # 4. 移除停用詞 (-RSW 開關)
        if remove_sw and (word in stop_words):
            continue  # 跳過這個字
            
        # 5. 詞形還原 (-LEM 開關)
        if apply_lem:
            wn_pos = get_wordnet_pos(tag)
            final_word = lemmatizer.lemmatize(word, pos=wn_pos)
        else:
            final_word = word
            
        processed_tokens.append(final_word)
    
    # 6. 移除第一個 Token (依照你原本的程式碼邏輯，可透過參數控制)
    if remove_first_token and len(processed_tokens) > 0:
        processed_tokens = processed_tokens[1:]
            
    return " ".join(processed_tokens)

def process_csv(input_file, output_file, target_col_idx, config):
    """
    傳入 config 字典來控制 preprocess_text 的行為
    """
    try:
        print(f"正在讀取檔案：{input_file} ...")
        df = pd.read_csv(input_file)
        
        if target_col_idx >= len(df.columns):
            print(f"錯誤：指定的欄位索引 {target_col_idx} 超出範圍。")
            return

        target_col_name = df.columns[target_col_idx]
        print(f"處理欄位: {target_col_name} | 實驗設定: {config}")

        # 使用 lambda 將 config 參數傳入 preprocess_text
        df.iloc[:, target_col_idx] = df.iloc[:, target_col_idx].apply(
            lambda x: preprocess_text(
                x, 
                remove_sym=config['remove_sym'],
                remove_sw=config['remove_sw'],
                apply_lem=config['apply_lem']
            )
        )
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"處理完成！已輸出至：{output_file}\n")
        
    except Exception as e:
        print(f"發生未預期的錯誤：{e}\n")

if __name__ == "__main__":
    input_csv = "augmented_emoji_backtrans_with_id_emoji.csv"
    TARGET_COLUMN_INDEX = 2  
    
    # ================= 消融實驗配置字典 =================
    # 定義不同的實驗組別，True 代表執行該步驟，False 代表關閉該步驟
    ablation_configs = {
        "Full_Preprocessing": {
            "remove_sym": True, "remove_sw": True, "apply_lem": True, "output_suffix": "full"
        },
        "Config_A_minus_LEM": {
            "remove_sym": True, "remove_sw": True, "apply_lem": False, "output_suffix": "minus_LEM"
        },
        "Config_B_minus_RSW": {
            "remove_sym": True, "remove_sw": False, "apply_lem": True, "output_suffix": "minus_RSW"
        },
        "Config_C_minus_SYM": {
            "remove_sym": False, "remove_sw": True, "apply_lem": True, "output_suffix": "minus_SYM"
        },
        "Raw_Text": {
            "remove_sym": False, "remove_sw": False, "apply_lem": False, "output_suffix": "raw"
        }
    }
    # ====================================================
    
    # 自動遍歷所有配置，產生多份不同的 CSV 供模型訓練對比
    for experiment_name, config in ablation_configs.items():
        print(f"--- 開始執行實驗組：{experiment_name} ---")
        output_csv = f"augmented_emoji_{config['output_suffix']}.csv"
        
        process_csv(input_csv, output_csv, TARGET_COLUMN_INDEX, config)