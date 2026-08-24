import pandas as pd
import pickle
import re
import os
#將表情符號根據字典轉換成文字描述


def load_emoji_dictionary(dict_path):
    """
    讀取 .p 檔並反轉字典，使其變成 {Emoji符號: 文字描述} 的格式
    """
    if not os.path.exists(dict_path):
        print(f"錯誤：找不到字典檔案 '{dict_path}'")
        return None

    try:
        with open(dict_path, 'rb') as fp:
            emoji_dict = pickle.load(fp)
        
        # 關鍵步驟：反轉字典
        # 原始資料可能是 { '文字': 'Emoji' }，我們需要反轉成 { 'Emoji': '文字' }
        emoji_dict = {v: k for k, v in emoji_dict.items()}
        print(f"成功載入 Emoji 字典，共 {len(emoji_dict)} 個表情符號。")
        return emoji_dict
    except Exception as e:
        print(f"讀取字典時發生錯誤：{e}")
        return None

def convert_emojis_to_word(text, emoji_dict):
    """
    將文字中的表情符號轉換為文字標籤
    邏輯參考自 Kaggle Emoji Dictionary 的說明
    """
    # 如果輸入不是字串 (例如空白欄位是 float nan)，直接回傳空字串
    if not isinstance(text, str):
        return ""
    
    # 針對每一個表情符號進行檢查與替換
    for emot, description in emoji_dict.items():
        if emot in text:
            # 格式化描述文字：
            # 1. replace(",", ""): 去除逗號
            # 2. replace(":", ""): 去除冒號
            # 3. split() & join("_"): 將單字用底線連接 (如 "smiling face" -> "smiling_face")
            clean_desc = "_".join(description.replace(",", "").replace(":", "").split())
            
            # 使用正規表達式替換
            # re.escape(emot) 很重要，因為 emoji 是特殊字元
            # 前後加上空格 " " 是為了避免跟前後的單字黏在一起
            text = re.sub(re.escape(emot), " " + clean_desc + " ", text)
            
    # 移除多餘的連續空白 (Optional)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_csv(input_file, output_file, dict_path):
    # 1. 載入字典
    emoji_dict = load_emoji_dictionary(dict_path)
    if emoji_dict is None:
        return

    try:
        print(f"正在讀取 CSV 檔案：{input_file} ...")
        # 讀取 CSV
        df = pd.read_csv(input_file)

        # 檢查欄位數量是否足夠
        if len(df.columns) < 2:
            print("錯誤：CSV 檔案欄位不足 2 欄，無法處理第 B 欄。")
            return

        # 取得第 B 欄名稱 (索引 1)
        target_col_name = df.columns[1]
        print(f"正在處理第 B 欄：'{target_col_name}' ...")

        # 2. 應用轉換函數到第 B 欄 (iloc[:, 1])
        # 使用 lambda 函式將 emoji_dict 傳入
        df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: convert_emojis_to_word(x, emoji_dict))

        # 3. 存檔
        print("正在寫入新的 CSV 檔...")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"處理完成！結果已儲存至：{output_file}")

    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 '{input_file}'")
    except Exception as e:
        print(f"發生未預期的錯誤：{e}")

# ==========================================
# 執行區
# ==========================================
if __name__ == "__main__":
    # 設定檔名與路徑
    input_csv = "80%test_data.csv"          # 您的輸入檔名
    output_csv = "80%test_data_replaced.csv"  # 輸出的檔名
    emoji_dict_path = "Emoji_Dict.p" # 下載的 .p 檔路徑

    process_csv(input_csv, output_csv, emoji_dict_path)