import pandas as pd
import os  

#將原本Tweets dataset 第B K 欄取出

# 1. 抓取目前這個 .py 程式檔所在的「絕對路徑」資料夾
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 組合出 csv 檔案的完整路徑
# 這會變成類似：C:\Users\Name\Desktop\Project\your_file.csv
input_filename = 'Tweets.csv'  # 請確認檔名完全一致
output_filename = 'polarity_text.csv'

input_path = os.path.join(current_dir, input_filename)
output_path = os.path.join(current_dir, output_filename)

print(f"Python 正在嘗試讀取這個路徑：{input_path}")

try:
    # 讀取 CSV
    df = pd.read_csv(input_path, encoding='utf-8-sig', header=None)
    
    # 取出 B 欄 (index 1) 與 K 欄 (index 10)
    extracted_df = df.iloc[:, [1, 10]]
    
    # 存檔
    extracted_df.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')

    print(f"成功！檔案已輸出至：{output_path}")
    
except FileNotFoundError:
    print("--------------------------------------------------")
    print("還是找不到檔案！請檢查以下兩點：")
    print(f"1. 你的 CSV 檔名真的是 '{input_filename}' 嗎？")
    print("2. Windows 是否隱藏了副檔名？（你的檔案可能其實叫做 your_file.csv.csv）")
    print("--------------------------------------------------")
except Exception as e:
    print(f"發生錯誤：{e}")
