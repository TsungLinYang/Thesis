import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch
import os

# ==========================================
# 設定區 (Configuration)
# ==========================================
INPUT_FILE = "polarity_text.csv"    # 輸入檔名
OUTPUT_FILE = "augmented_backtrans_with_id.csv"    # 輸出檔名

# ---在此設定欄位名稱---
LABEL_COLUMN = 0        # 標籤所在的欄位 
TEXT_COLUMN = 1         # 文本所在的欄位 
CSV_HAS_HEADER = False  # 如果 CSV 第一行是標題，請改成 True

TARGET_COUNT = 9182                 # 目標數量
BATCH_SIZE = 32                     # 批次大小
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 我們將 ID 欄位的名稱暫定為 'source_id'，方便程式內部處理
ID_COLUMN_NAME = 'source_id' 

# ==========================================
# 1. 初始化反向翻譯增強器
# ==========================================
print(f"正在下載並初始化翻譯模型 (Device: {DEVICE})...")
aug = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-de', 
    to_model_name='Helsinki-NLP/opus-mt-de-en',
    device=DEVICE,
    batch_size=BATCH_SIZE
)

def augment_data(df, label, target_count):
    """
    針對特定 Label 進行增強，並保留原始資料的 ID
    """
    original_data = df[df[LABEL_COLUMN] == label]
    current_count = len(original_data)
    needed_count = target_count - current_count
    
    if needed_count <= 0:
        print(f"類別 {label} 數量 ({current_count}) 已足夠，跳過增強。")
        return pd.DataFrame() 
    
    print(f"正在增強類別 '{label}' (Back Translation): {current_count} -> {target_count} (需生成 {needed_count} 筆)")
    
    new_sentences = []
    new_ids = []  # 用來儲存對應的 ID
    
    pbar = tqdm(total=needed_count)
    
    while len(new_sentences) < needed_count:
        current_batch_size = min(BATCH_SIZE, needed_count - len(new_sentences))
        
        # [修改點 1] 改為隨機抽取「整行資料」，這樣才能同時拿到 Text 和 ID
        # sample 函式會隨機選取 row
        batch_rows = original_data.sample(current_batch_size, replace=True)
        
        # 取出文本列表
        batch_src = batch_rows[TEXT_COLUMN].tolist()
        # [修改點 2] 取出對應的 ID 列表
        batch_id_list = batch_rows[ID_COLUMN_NAME].tolist()
        
        try:
            augmented_batch = aug.augment(batch_src)
            
            # 處理 nlpaug 回傳格式 (有時單句回傳 str，多句回傳 list)
            if isinstance(augmented_batch, str):
                augmented_batch = [augmented_batch]
                
            new_sentences.extend(augmented_batch)
            # [修改點 3] 將 ID 加入列表 (aug.augment 保持順序，所以 ID 會對應)
            new_ids.extend(batch_id_list)
            
            pbar.update(len(augmented_batch))
            
        except Exception as e:
            print(f"翻譯過程發生錯誤 (跳過此批次): {e}")
            continue
        
    pbar.close()
    
    # 截斷多餘資料 (雖然邏輯上不太會超出太多，但保險起見)
    new_sentences = new_sentences[:needed_count]
    new_ids = new_ids[:needed_count]
    
    # [修改點 4] 建立 DataFrame 時包含 ID
    # 注意：這裡只包含 Label, Text, ID。原本的中間欄位(如表情符號)在增強資料中會是空白(NaN)
    new_df = pd.DataFrame({
        LABEL_COLUMN: [label] * len(new_sentences),
        TEXT_COLUMN: new_sentences,
        ID_COLUMN_NAME: new_ids
    })
    
    return new_df

# ==========================================
# 主程式
# ==========================================
def main():
    try:
        print(f"正在讀取檔案：{INPUT_FILE} ...")
        
        if CSV_HAS_HEADER:
            df = pd.read_csv(INPUT_FILE)
        else:
            df = pd.read_csv(INPUT_FILE, header=None)

        # 檢查欄位是否存在
        if LABEL_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
            print(f"錯誤：找不到指定的欄位。")
            return

        # [修改點 5] 為原始資料產生 ID (0, 1, 2, ... N)
        print("正在生成原始資料 ID...")
        df[ID_COLUMN_NAME] = range(len(df))

        # 2. 分別增強 Neutral 和 Positive
        augmented_neutral = augment_data(df, 'neutral', TARGET_COUNT)
        augmented_positive = augment_data(df, 'positive', TARGET_COUNT)
        
        # 3. 合併資料
        print("正在合併資料...")
        final_df = pd.concat([
            augmented_neutral,
            augmented_positive,
            df
        ], axis=0, ignore_index=True)
        
        # [修改點 6] 整理欄位順序，確保 ID 在最後一欄
        # 找出所有欄位
        cols = list(final_df.columns)
        # 把 ID 欄位名稱從列表中移除
        if ID_COLUMN_NAME in cols:
            cols.remove(ID_COLUMN_NAME)
        # 排序其他欄位 (讓 0, 1, 2... 保持順序)
        # 注意：如果欄位名混雜了整數和字串，sort 可能會報錯，這裡簡單處理
        try:
            cols.sort() 
        except:
            pass # 如果排序失敗就維持原樣
            
        # 將 ID 欄位加到最後面
        cols.append(ID_COLUMN_NAME)
        
        # 重新排列 DataFrame
        final_df = final_df[cols]

        # 4. 存檔
        print(f"正在寫入檔案 (總筆數: {len(final_df)})...")
        final_df.to_csv(OUTPUT_FILE, index=False, header=CSV_HAS_HEADER, encoding='utf-8-sig')
        print(f"處理完成！已輸出至：{OUTPUT_FILE}")
        print(f"ID 位於最後一欄，原始資料與增強資料共用同一個 ID。")
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{INPUT_FILE}'")
    except Exception as e:
        print(f"發生未預期的錯誤：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()