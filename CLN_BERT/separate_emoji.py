#將表情符號分離 

import pandas as pd
import os
import emoji

# 1. 設定路徑 (沿用你原本的邏輯)
current_dir = os.path.dirname(os.path.abspath(__file__))
input_filename = 'polarity_text.csv'  # 讀取剛剛產生的檔案
output_filename = 'polarity_emoji_text.csv'  # 處理好的新檔案名稱

input_path = os.path.join(current_dir, input_filename)
output_path = os.path.join(current_dir, output_filename)

print(f"正在讀取：{input_path}")

# 定義處理函式
def extract_emojis(text):
    # 如果內容是空的 (NaN)，回傳空字串
    if pd.isna(text):
        return ""
    # 萃取所有 emoji 並串接起來
    return ''.join(c for c in str(text) if emoji.is_emoji(c))

def remove_emojis(text):
    if pd.isna(text):
        return ""
    # 將 emoji 替換為空字串，只留文字
    return emoji.replace_emoji(str(text), replace='')

try:
    # 2. 讀取 CSV
    # header=None 因為我們知道這個檔案沒有標題列
    df = pd.read_csv(input_path, header=None, encoding='utf-8-sig')
    
    # 檢查是否至少有兩欄 (Index 0 和 1)
    if df.shape[1] < 2:
        raise ValueError("CSV 檔案欄位不足，找不到第 B 欄 (Index 1)。")

    # 3. 處理資料
    # df[0] 是原本的 A 欄 (保持不變)
    # df[1] 是原本的 B 欄 (混合文本)
    
    # 產生新的 Emoji 欄位
    emojis_col = df[1].apply(extract_emojis)
    
    # 產生新的純文字欄位
    text_col = df[1].apply(remove_emojis)
    
    # 4. 組合新的 DataFrame
    # 結構：[原始A欄, Emoji欄, 純文字欄]
    # 這對應到 Excel 的 A, B, C 欄
    new_df = pd.DataFrame({
        'Column_A': df[0],
        'Column_B_Emojis': emojis_col,
        'Column_C_Text': text_col
    })

    # 5. 存檔
    # header=False 表示不寫入我們上面暫定的欄位名稱
    new_df.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')

    print(f"處理完成！")
    print(f"新的 A 欄：原始分類/資訊")
    print(f"新的 B 欄：僅包含 Emoji")
    print(f"新的 C 欄：僅包含純文字")
    print(f"檔案已儲存至：{output_path}")

except FileNotFoundError:
    print(f"找不到檔案：{input_filename}，請確認你是否已經執行過上一步驟。")
except Exception as e:
    print(f"發生錯誤：{e}")
    
