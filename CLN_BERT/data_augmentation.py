import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch
import os
import random

# ==========================================
# 設定區 (Configuration)
# ==========================================
INPUT_FILE = "polarity_emoji_preprocessedtext.csv" 
OUTPUT_FILE = "augmented_emoji_backtrans_with_id_emoji(processed).csv"

# --- 欄位索引設定 ---
LABEL_INDEX = 0         # 標籤所在欄位索引
EMOJI_INDEX = 1         # 表情符號所在欄位索引
TEXT_INDEX = 2          # 文本所在欄位索引

TARGET_COUNT = 9182    
BATCH_SIZE = 32         
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ID_COLUMN = "source_id"

# 定義你真正要增強的標籤類別
VALID_LABELS = ['neutral', 'positive', 'negative']

# ==========================================
# 1. 初始化多個反向翻譯增強器
# ==========================================
print(f"正在初始化多語言翻譯模型 (Device: {DEVICE})...")
augmenters = [
    naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de', 
        to_model_name='Helsinki-NLP/opus-mt-de-en',
        device=DEVICE, batch_size=BATCH_SIZE
    ),
    naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-fr', 
        to_model_name='Helsinki-NLP/opus-mt-fr-en',
        device=DEVICE, batch_size=BATCH_SIZE
    ),
    naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-es', 
        to_model_name='Helsinki-NLP/opus-mt-es-en',
        device=DEVICE, batch_size=BATCH_SIZE
    )
]

def augment_data(df, label, target_count, label_col, text_col, emoji_col):
    """
    針對特定 Label 進行增強，補足差額
    """
    label_data = df[df[label_col] == label]
    current_count = len(label_data)
    needed_count = target_count - current_count
    
    if needed_count <= 0:
        print(f"類別 '{label}' 數量 ({current_count}) 已達標，無需增強。")
        return pd.DataFrame() 
    
    print(f"正在增強類別 '{label}': 目前 {current_count} 筆 -> 目標 {target_count} 筆 (需生成 {needed_count} 筆)")
    
    new_sentences = []
    source_ids = []
    new_emojis = []
    
    pbar = tqdm(total=needed_count, desc=f"Augmenting {label}")
    
    while len(new_sentences) < needed_count:
        batch_size = min(BATCH_SIZE, needed_count - len(new_sentences))
        sampled_rows = label_data.sample(batch_size, replace=True)
        
        batch_src = sampled_rows[text_col].tolist()
        batch_ids = sampled_rows[ID_COLUMN].tolist()
        batch_emojis = sampled_rows[emoji_col].tolist() 
        
        try:
            current_aug = random.choice(augmenters)
            augmented_batch = current_aug.augment(batch_src)
            
            if isinstance(augmented_batch, str):
                augmented_batch = [augmented_batch]
                
            new_sentences.extend(augmented_batch)
            source_ids.extend(batch_ids)
            new_emojis.extend(batch_emojis) 
            pbar.update(len(augmented_batch))
            
        except Exception as e:
            print(f"\n[錯誤] 翻譯失敗: {e}")
            continue
        
    pbar.close()
    
    return pd.DataFrame({
        label_col: [label] * needed_count,
        emoji_col: new_emojis[:needed_count], 
        text_col: new_sentences[:needed_count],
        ID_COLUMN: source_ids[:needed_count]
    })

# ==========================================
# 主程式
# ==========================================
def main():
    try:
        print(f"正在讀取檔案：{INPUT_FILE} ...")
        # 強制指定 header=0，解決把標題當資料的問題
        df = pd.read_csv(INPUT_FILE, header=0)

        # 自動獲取實際的欄位名稱
        cols = df.columns
        L_COL, E_COL, T_COL = cols[LABEL_INDEX], cols[EMOJI_INDEX], cols[TEXT_INDEX]
        
        # 建立原始資料 ID
        df[ID_COLUMN] = df.index 

        # 過濾標籤：只保留 VALID_LABELS 中有的標籤
        found_labels = [l for l in df[L_COL].unique() if l in VALID_LABELS]
        print(f"偵測到有效標籤: {found_labels}")

        all_augmented_chunks = []
        for label in found_labels:
            aug_chunk = augment_data(df, label, TARGET_COUNT, L_COL, T_COL, E_COL)
            if not aug_chunk.empty:
                all_augmented_chunks.append(aug_chunk)

        # 只保留原始資料中屬於有效標籤的部分，再與增強資料合併
        print("正在合併與整理最終資料集...")
        df_original_filtered = df[df[L_COL].isin(found_labels)]
        final_df = pd.concat([df_original_filtered] + all_augmented_chunks, axis=0).reset_index(drop=True)
        
        # 驗證數量統計
        print("\n--- 最終各類別數量統計 ---")
        print(final_df[L_COL].value_counts())
        
        # 存檔 (保留 Header 以便後續讀取)
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n處理完成！總計: {len(final_df)} 筆資料已存至 {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"發生未預期的錯誤：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()