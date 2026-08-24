import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys
from tqdm import tqdm
import time

# ==========================================
# 0. 全域設定與參數 (Configuration)
# ==========================================
# 模式選項: 'train' (訓練), 'inference' (原測試集推論), 'external_infer' (純外部獨立資料推論)
RUN_MODE = 'external_infer'  

# 檔案路徑設定
INPUT_FILE = "augmented_backtrans_with_id.csv"
SAVE_PATH = "best_sentiment_bert_baseline"
OUTPUT_REPORT_CSV = "bert_analysis_report.csv"

# 🔴 新增：純外部獨立推論資料設定 (專供 RUN_MODE = 'external_infer' 使用)
PURE_EXTERNAL_FILE = "80%test_data_replaced_final.csv" 

# 訓練時的外部資料融合設定
MERGE_EXTERNAL_TEST = True            
EXTERNAL_TEST_FILE = "LLM_Sarcasm_normal_replaced.csv" 
EXTERNAL_TEST_LIMIT = 500             

# 模型參數
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-uncased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[{RUN_MODE.upper()} MODE] Using device: {device}")

# ==========================================
# 1. 定義 Dataset
# ==========================================
class AirlineSentimentDataset(Dataset):
    def __init__(self, texts, labels, ids, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = int(self.labels[item])
        cur_id = self.ids[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'id': cur_id
        }

# ==========================================
# 2. 資料讀取函數
# ==========================================
def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        print(f"找不到檔案：{file_path}")
        return None, None, None

    print(f"正在讀取檔案：{file_path} ...")
    try:
        df = pd.read_csv(file_path, header=None)
        
        raw_labels = df.iloc[:, 0].astype(str).str.strip()
        raw_texts = df.iloc[:, 1].astype(str)
        raw_ids = df.iloc[:, -1].values

        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        mask = raw_labels.isin(label_map.keys())
        clean_texts = raw_texts[mask].values
        clean_labels = raw_labels[mask].map(label_map).values.astype(int)
        clean_ids = raw_ids[mask]
        
        print(f"資料讀取完成。有效樣本數：{len(clean_texts)}")
        return clean_texts, clean_labels, clean_ids
        
    except Exception as e:
        print(f"讀取錯誤：{e}")
        return None, None, None

# ==========================================
# 3. 訓練相關函數
# ==========================================
def train_epoch(model, data_loader, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# ==========================================
# 4. 推論分析相關函數
# ==========================================
def generate_test_report(model, data_loader, device, output_csv):
    model.eval()
    
    texts = []
    true_labels = []
    pred_labels = []
    ids = [] 
    
    total_inference_time = 0.0
    total_samples = 0
    
    print(f"正在生成詳細測試報告，輸出至: {output_csv} ...")
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Inference"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            batch_size_current = input_ids.size(0)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            total_inference_time += (end_time - start_time)
            total_samples += batch_size_current
            
            texts.extend(d['review_text'])
            true_labels.extend(targets.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            ids.extend(d['id']) 

    avg_latency_per_batch = (total_inference_time / len(data_loader)) * 1000
    avg_latency_per_sample = (total_inference_time / total_samples) * 1000
    throughput = total_samples / total_inference_time
    
    # 計算整體 Accuracy
    final_accuracy = accuracy_score(true_labels, pred_labels)
    
    print("\n" + "="*40)
    print("🚀 推論效能與準確度報告 (Inference Report)")
    print("="*40)
    print(f"整體準確度 (Accuracy): {final_accuracy:.4f}")
    print(f"總純推論時間 (排除資料載入): {total_inference_time:.4f} 秒")
    print(f"總處理樣本數: {total_samples} 筆")
    print(f"平均每 Batch 延遲: {avg_latency_per_batch:.2f} ms")
    print(f"平均單筆樣本延遲 (Latency): {avg_latency_per_sample:.2f} ms/sample")
    print(f"模型吞吐量 (Throughput): {throughput:.2f} samples/sec")
    print("="*40 + "\n")

    idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    df_result = pd.DataFrame({
        'Text': texts,
        'True_Label_ID': true_labels,
        'Pred_Label_ID': pred_labels,
        'ID': ids 
    })
    
    df_result['True_Label'] = df_result['True_Label_ID'].map(idx_to_label)
    df_result['Predicted_Label'] = df_result['Pred_Label_ID'].map(idx_to_label)
    df_result['Is_Correct'] = df_result['True_Label_ID'] == df_result['Pred_Label_ID']
    df_result['Status'] = df_result['Is_Correct'].apply(lambda x: '✅ Correct' if x else '❌ Wrong')
    
    final_cols = ['Status', 'True_Label', 'Predicted_Label', 'Text', 'ID']
    df_final = df_result[final_cols]
    
    df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    error_count = len(df_final[df_final['Status'] == '❌ Wrong'])
    print(f"[報告完成] 總筆數: {len(df_final)} | 錯誤筆數: {error_count}")


# ==========================================
# 5. 主程式邏輯 (Main Execution)
# ==========================================
if __name__ == "__main__":
    
    # ============================================================
    # 分支 A: 純外部資料推論模式 (EXTERNAL_INFER)
    # ============================================================
    if RUN_MODE == 'external_infer':
        print("\n" + "="*40)
        print("      進入純外部推論模式 (EXTERNAL INFER MODE)")
        print("="*40 + "\n")

        if not os.path.exists(SAVE_PATH):
            sys.exit(f"錯誤：找不到模型路徑 '{SAVE_PATH}'。請先執行 'train' 模式。")

        print(f"準備讀取純外部測試檔案: {PURE_EXTERNAL_FILE}")
        ext_texts, ext_labels, ext_ids = load_and_preprocess(PURE_EXTERNAL_FILE)
        
        if ext_texts is None:
            sys.exit("錯誤：外部資料載入失敗，請確認 PURE_EXTERNAL_FILE 路徑與格式正確。")

        print("正在載入已訓練的模型權重與 Tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(SAVE_PATH)
        model = BertForSequenceClassification.from_pretrained(SAVE_PATH)
        model = model.to(device)

        ext_dataset = AirlineSentimentDataset(ext_texts, ext_labels, ext_ids, tokenizer, MAX_LEN)
        ext_data_loader = DataLoader(ext_dataset, batch_size=BATCH_SIZE, shuffle=False)

        generate_test_report(model, ext_data_loader, device, OUTPUT_REPORT_CSV)
        
        print(f"\n推論分析完成，結果已匯出至: {OUTPUT_REPORT_CSV}")
        sys.exit()

    # ============================================================
    # 以下為 TRAIN 或 INFERENCE 模式 (原始資料切分邏輯)
    # ============================================================
    texts, labels, ids = load_and_preprocess(INPUT_FILE)
    
    if texts is None:
        sys.exit("錯誤：主要資料載入失敗，程式終止。")

    # 第一階段：切出 80% Training, 20% Temp
    df_train_txt, df_temp_txt, y_train, y_temp, id_train, id_temp = train_test_split(
        texts, labels, ids, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 第二階段：將 20% Temp 切半 -> 10% Validation, 10% Testing
    df_val_txt, df_test_txt, y_val, y_test, id_val, id_test = train_test_split(
        df_temp_txt, y_temp, id_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"資料切割完成 -> 訓練集(80%): {len(df_train_txt)} | 驗證集(10%): {len(df_val_txt)} | 原測試集(10%): {len(df_test_txt)}")

    # ---------------------------------------------------------
    # 外部測試資料融合邏輯 (僅應用於 Train / Inference)
    # ---------------------------------------------------------
    df_test_txt_final = df_test_txt
    y_test_final = y_test
    id_test_final = id_test

    if MERGE_EXTERNAL_TEST:
        print("\n" + "-"*40)
        print(f"準備讀取並融合外部測試檔案: {EXTERNAL_TEST_FILE}")
        ext_texts, ext_labels, ext_ids = load_and_preprocess(EXTERNAL_TEST_FILE)
        
        if ext_texts is not None:
            limit = EXTERNAL_TEST_LIMIT if EXTERNAL_TEST_LIMIT is not None else len(ext_texts)
            ext_texts = ext_texts[:limit]
            ext_labels = ext_labels[:limit]
            ext_ids = ext_ids[:limit]
            
            print(f"擷取了 {len(ext_texts)} 筆外部資料，準備與原測試集 ({len(df_test_txt)} 筆) 融合...")
            
            df_test_txt_final = np.concatenate([df_test_txt_final, ext_texts])
            y_test_final = np.concatenate([y_test_final, ext_labels])
            id_test_final = np.concatenate([id_test_final, ext_ids])
            
            print(f"融合完成！最終測試集總筆數: {len(df_test_txt_final)}")
        else:
            print("⚠️ 外部資料讀取失敗或不存在，將只使用原本的 10% 測試集。")
        print("-" * 40 + "\n")

    # ============================================================
    # 分支 B: 訓練模式 (TRAIN)
    # ============================================================
    if RUN_MODE == 'train':
        print("\n" + "="*40)
        print("      進入訓練模式 (TRAIN MODE)")
        print("="*40 + "\n")

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
        model = model.to(device)

        train_dataset = AirlineSentimentDataset(df_train_txt, y_train, id_train, tokenizer, MAX_LEN)
        val_dataset = AirlineSentimentDataset(df_val_txt, y_val, id_val, tokenizer, MAX_LEN)
        test_dataset = AirlineSentimentDataset(df_test_txt_final, y_test_final, id_test_final, tokenizer, MAX_LEN)

        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        best_accuracy = 0.0

        print(f"模型將儲存於: {SAVE_PATH}")
        
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            
            train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, len(df_train_txt))
            print(f'Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')

            val_acc, val_loss = eval_model(model, val_data_loader, device, len(df_val_txt))
            print(f'Valid | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')

            if val_acc > best_accuracy:
                print(f"--> 效能提升 ({best_accuracy:.4f} -> {val_acc:.4f})！儲存最佳權重...")
                best_accuracy = val_acc
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                model.save_pretrained(SAVE_PATH)
                tokenizer.save_pretrained(SAVE_PATH)
            
            print("-" * 30)
        
        print("10 Epochs 訓練與驗證完成。")
        
        print("\n" + "="*40)
        print("載入最佳模型權重，對最終 Testing Set 進行評估...")
        print("="*40)
        
        best_model = BertForSequenceClassification.from_pretrained(SAVE_PATH)
        best_model = best_model.to(device)
        
        test_acc, test_loss = eval_model(best_model, test_data_loader, device, len(df_test_txt_final))
        print(f'Final Test | Loss: {test_loss:.4f} | Acc: {test_acc:.4f}')
        
        generate_test_report(best_model, test_data_loader, device, OUTPUT_REPORT_CSV)

    # ============================================================
    # 分支 C: 推論模式 (INFERENCE) - 適用於原始切分的測試集
    # ============================================================
    elif RUN_MODE == 'inference':
        print("\n" + "="*40)
        print("      進入推論模式 (INFERENCE MODE)")
        print("="*40 + "\n")

        if not os.path.exists(SAVE_PATH):
            sys.exit(f"錯誤：找不到模型路徑 '{SAVE_PATH}'。請先執行 'train' 模式。")

        print("正在載入已訓練的模型權重...")
        tokenizer = BertTokenizer.from_pretrained(SAVE_PATH)
        model = BertForSequenceClassification.from_pretrained(SAVE_PATH)
        model = model.to(device)

        test_dataset = AirlineSentimentDataset(df_test_txt_final, y_test_final, id_test_final, tokenizer, MAX_LEN)
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        generate_test_report(model, test_data_loader, device, OUTPUT_REPORT_CSV)
        
        print(f"\n推論完成，請查看: {OUTPUT_REPORT_CSV}")