import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel,
    AlbertTokenizer, AlbertModel,
    DistilBertTokenizer, DistilBertModel,
    ElectraTokenizer, ElectraModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import gensim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================
# 1. 全域設定與參數 (Configuration)
# ==========================================
# 🔴🔴🔴 實驗架構與模型開關 🔴🔴🔴
CHOSEN_MODEL = 'bert' #選擇bert albert distilbert electra tinybert其中一個

# 選項: 'train' (訓練), 'analysis' (分析原始資料), 'external_test' (純外部資料推論)
RUN_MODE = 'external_test'

# 🔴 Stage 1 (VADER 攔截機制) 開關
ENABLE_STAGE_1 = False

MODEL_ZOO = {
    'bert': {'name': 'bert-base-uncased', 'dim': 768, 'has_pooler': True, 'tokenizer_cls': BertTokenizer, 'model_cls': BertModel},
    'albert': {'name': 'albert-base-v2', 'dim': 768, 'has_pooler': True, 'tokenizer_cls': AlbertTokenizer, 'model_cls': AlbertModel},
    'distilbert': {'name': 'distilbert-base-uncased', 'dim': 768, 'has_pooler': False, 'tokenizer_cls': DistilBertTokenizer, 'model_cls': DistilBertModel},
    'electra': {'name': 'google/electra-small-discriminator', 'dim': 256, 'has_pooler': False, 'tokenizer_cls': ElectraTokenizer, 'model_cls': ElectraModel},
    'tinybert': {'name': 'prajjwal1/bert-tiny', 'dim': 128, 'has_pooler': True, 'tokenizer_cls': BertTokenizer, 'model_cls': BertModel}
}

CONFIG = {
    'model_type': CHOSEN_MODEL,
    'epoch': 10,
    'batch_size': 16,
    'lr': 3e-5,
    'max_len': 128,
    'emoji_dim': 300,
    'num_classes': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'emoji_bin_path': 'emoji2vec.bin',
    'csv_path': 'augmented_emoji_raw.csv',
    
    # 🔴 獨立外部測試資料路徑 (專供 RUN_MODE = 'external_test' 使用)
    'external_infer_data': '80%test_data.csv', 
    
    # 🔴 訓練時混合的 LLM 資料設定
    'LLM_dataset': 'LLM(new).csv', 
    'num_external_samples': 500,  
    'emoji_dict_path': 'emoji_dictionary.csv'
}

CURRENT_MODEL_CFG = MODEL_ZOO[CONFIG['model_type']]
CONFIG['hidden_dim'] = CURRENT_MODEL_CFG['dim']

print("="*40)
print(f"🚀 初始化設定完畢")
print(f"執行模式: {RUN_MODE.upper()}")
print(f"前端攔截 (Stage 1): {'啟用 (Enabled)' if ENABLE_STAGE_1 else '停用 (Disabled, 100% 進入 Stage 2)'}")
print(f"採用模型架構: {CONFIG['model_type'].upper()}")
print(f"隱藏層維度: {CONFIG['hidden_dim']}")
print(f"運算設備: {CONFIG['device']}")
print(f"外部測試檔案路徑: {CONFIG['external_infer_data']}")
print("="*40)

# ==========================================
# 🔴 VADER 專家規則攔截器 (Net_Positive 客製公式版)
# ==========================================
class VaderSarcasmDetector:
    def __init__(self, emoji_csv_path):
        self.analyzer = SentimentIntensityAnalyzer()
        self.emoji_dict = {}
        
        if os.path.exists(emoji_csv_path):
            df = pd.read_csv(emoji_csv_path)
            for _, row in df.iterrows():
                char = str(row['Emoji']).strip()
                pos_count = row.get('Positive', 0)
                neg_count = row.get('Negative', 0)
                neu_count = row.get('Neutral', 0)
                occurrence = row.get('Occurrences', 0)
                denominator = pos_count + neg_count + neu_count
                
                if denominator > 0:
                    net_positive = (pos_count - neg_count) / denominator
                    self.emoji_dict[char] = {
                        'net_positive': net_positive, 
                        'occurrence': occurrence
                    }
            if ENABLE_STAGE_1:
                print(f"[Info] 成功載入 Emoji 字典，共 {len(self.emoji_dict)} 筆規則。")
        else:
            if ENABLE_STAGE_1:
                print(f"[Warn] 找不到 {emoji_csv_path}，將無法計算 Emoji 機率，反諷判定可能失效。")

    def detect(self, text, emoji_char):
        vader_score = self.analyzer.polarity_scores(text)['compound']
        probs = self.emoji_dict.get(emoji_char, {'net_positive': 0.0, 'occurrence': 0})
        net_positive = probs['net_positive']
        occurrence = probs['occurrence']

        # 寫入先前實驗獲得的最佳閾值
        X = 0.9
        Y = -0.9
        W = -0.1
        Z = 0.5
        
        is_sarcasm = False
        if occurrence > 100:
            if (net_positive > X and vader_score < Y) or (net_positive < W and vader_score > Z):
                is_sarcasm = True
                
        return is_sarcasm, vader_score, net_positive

# ==========================================
# 2. 模型組件 (Context-Aware CLN & Fusion Model)
# ==========================================
class ContextAwareCLN(nn.Module): 
    def __init__(self, input_dim, condition_dim, epsilon=1e-6):
        super(ContextAwareCLN, self).__init__()
        self.W1 = nn.Linear(condition_dim, input_dim) 
        self.W2 = nn.Linear(condition_dim, input_dim) 
        self.layernorm = nn.LayerNorm(input_dim, elementwise_affine=False, eps=epsilon)

    def forward(self, x, y):
        normalized_x = self.layernorm(x)
        condition = torch.cat([y, x], dim=1) 
        gamma = self.W1(condition) 
        beta = self.W2(condition)
        out = normalized_x * gamma + beta
        return out, gamma, beta

class FusionModel(nn.Module):
    def __init__(self, model_cls, model_name, has_pooler, emoji_input_dim=300, hidden_dim=128, num_classes=3):
        super(FusionModel, self).__init__()
        self.has_pooler = has_pooler
        self.bert = model_cls.from_pretrained(model_name)
        self.emoji_projection = nn.Linear(emoji_input_dim, hidden_dim)
        self.cln = ContextAwareCLN(input_dim=hidden_dim, condition_dim=hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, emoji_vec, return_internal_states=False):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.has_pooler:
            text_feat = bert_out.pooler_output  
        else:
            text_feat = bert_out.last_hidden_state[:, 0, :]
            
        emoji_feat = self.emoji_projection(emoji_vec) 
        fused_feat, gamma, beta = self.cln(text_feat, emoji_feat) 
        logits = self.classifier(fused_feat)

        if return_internal_states:
            return logits, gamma, beta
        return logits

# ==========================================
# 3. 資料處理 (Dataset 整合 VADER)
# ==========================================
class EmojiHandler:
    def __init__(self, bin_path):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"錯誤：找不到 Emoji2Vec 檔案: {bin_path}")
        self.model = gensim.models.KeyedVectors.load_word2vec_format(bin_path, binary=True)
        self.vector_size = self.model.vector_size

    def get_avg_vector(self, emoji_text):
        if pd.isna(emoji_text) or str(emoji_text).strip() == "":
            return np.zeros(self.vector_size, dtype=np.float32)
        vectors = [self.model[char] for char in str(emoji_text) if char in self.model]
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0)

class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, emoji_handler, sarcasm_detector, max_len=128):
        self.df = df.reset_index(drop=True) 
        self.tokenizer = tokenizer
        self.emoji_handler = emoji_handler
        self.sarcasm_detector = sarcasm_detector
        self.max_len = max_len
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = self.label_map.get(row['A'], 1)
        raw_emoji = str(row['B'])
        text = str(row['C'])
        item_id = row['ID'] if 'ID' in row else index 
        
        emoji_vec = self.emoji_handler.get_avg_vector(raw_emoji)
        
        if ENABLE_STAGE_1:
            is_sarcasm, vader_score, net_positive = self.sarcasm_detector.detect(text, raw_emoji)
        else:
            is_sarcasm, vader_score, net_positive = False, 0.0, 0.0
        
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emoji_vec': torch.tensor(emoji_vec, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
            'is_sarcasm': torch.tensor(is_sarcasm, dtype=torch.bool), 
            'vader_score': torch.tensor(vader_score, dtype=torch.float),
            'net_positive': torch.tensor(net_positive, dtype=torch.float), 
            'text': text,        
            'emoji': raw_emoji,  
            'id': item_id        
        }

# ==========================================
# 4. 訓練、驗證、測速
# ==========================================
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss, correct_predictions = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    num_trained_batches = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids_all = batch['input_ids'].to(device)
        attention_mask_all = batch['attention_mask'].to(device)
        emoji_vec_all = batch['emoji_vec'].to(device)
        labels = batch['label'].to(device)
        is_sarcasm = batch['is_sarcasm'].to(device)
        
        non_sarcasm_mask = ~is_sarcasm
        preds = torch.zeros_like(labels)
        
        if non_sarcasm_mask.sum() > 0:
            input_ids = input_ids_all[non_sarcasm_mask]
            attention_mask = attention_mask_all[non_sarcasm_mask]
            emoji_vec = emoji_vec_all[non_sarcasm_mask]
            nn_labels = labels[non_sarcasm_mask]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, emoji_vec)
            loss = loss_fn(outputs, nn_labels)
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            num_trained_batches += 1
            
            _, nn_preds = torch.max(outputs, dim=1)
            preds[non_sarcasm_mask] = nn_preds 
            
        correct_predictions += torch.sum(preds == labels)
        
    avg_loss = total_loss / num_trained_batches if num_trained_batches > 0 else 0
    return correct_predictions.double() / len(data_loader.dataset), avg_loss

def eval_model(model, data_loader, device, desc="Validating"):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds, all_labels = [], []
    num_eval_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids_all = batch['input_ids'].to(device)
            attention_mask_all = batch['attention_mask'].to(device)
            emoji_vec_all = batch['emoji_vec'].to(device)
            labels = batch['label'].to(device)
            is_sarcasm = batch['is_sarcasm'].to(device)
            
            non_sarcasm_mask = ~is_sarcasm
            preds = torch.zeros_like(labels)
            
            if non_sarcasm_mask.sum() > 0:
                input_ids = input_ids_all[non_sarcasm_mask]
                attention_mask = attention_mask_all[non_sarcasm_mask]
                emoji_vec = emoji_vec_all[non_sarcasm_mask]
                nn_labels = labels[non_sarcasm_mask]
                
                outputs = model(input_ids, attention_mask, emoji_vec)
                loss = loss_fn(outputs, nn_labels)
                total_loss += loss.item()
                num_eval_batches += 1
                
                _, nn_preds = torch.max(outputs, dim=1)
                preds[non_sarcasm_mask] = nn_preds
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive'], zero_division=0)
    avg_loss = total_loss / num_eval_batches if num_eval_batches > 0 else 0
    return accuracy, avg_loss, report

def generate_test_report_with_speed(model, data_loader, device, output_csv="cln_inference_report.csv"):
    model.eval()
    texts, emojis, ids, true_labels, pred_labels, override_flags = [], [], [], [], [], []
    vader_scores_list, net_positives_list = [], []
    sarcasm_intercept_count = 0
    
    print(f"[Info] 正在生成詳細測試報告與測量 End-to-End 推論速度...")
    
    if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'): 
        torch.cuda.synchronize()
    e2e_start_time = time.perf_counter()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference & Timing"):
            input_ids_all = batch['input_ids'].to(device)
            attention_mask_all = batch['attention_mask'].to(device)
            emoji_vec_all = batch['emoji_vec'].to(device)
            labels = batch['label'].to(device)
            is_sarcasm = batch['is_sarcasm'].to(device)
            
            non_sarcasm_mask = ~is_sarcasm
            preds = torch.zeros_like(labels)
            sarcasm_intercept_count += is_sarcasm.sum().item()
            
            if non_sarcasm_mask.sum() > 0:
                input_ids = input_ids_all[non_sarcasm_mask]
                attention_mask = attention_mask_all[non_sarcasm_mask]
                emoji_vec = emoji_vec_all[non_sarcasm_mask]
                
                outputs = model(input_ids, attention_mask, emoji_vec)
                _, nn_preds = torch.max(outputs, dim=1)
                preds[non_sarcasm_mask] = nn_preds
                
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            texts.extend(batch['text'])
            emojis.extend(batch['emoji'])
            ids.extend(batch['id'])
            override_flags.extend(is_sarcasm.cpu().numpy())
            vader_scores_list.extend(batch['vader_score'].cpu().numpy())
            net_positives_list.extend(batch['net_positive'].cpu().numpy())

    if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'): 
        torch.cuda.synchronize()
    e2e_end_time = time.perf_counter()
    
    total_e2e_time = e2e_end_time - e2e_start_time
    total_samples = len(data_loader.dataset) 
    
    avg_latency_per_sample = (total_e2e_time / total_samples) * 1000   
    throughput = total_samples / total_e2e_time                        
    
    print("\n" + "="*40)
    print(f"🚀 End-to-End 系統效能報告 ({CONFIG['model_type'].upper()})")
    print(f"配置: {'雙階段 (Stage 1 + 2)' if ENABLE_STAGE_1 else '單階段純模型 (Stage 2 Only)'}")
    print("="*40)
    print(f"總處理時間 (含前處理/VADER/模型): {total_e2e_time:.4f} 秒")
    print(f"總處理樣本數: {total_samples} 筆")
    print(f"👉 觸發 VADER 反諷強制攔截 (Early Exit): {sarcasm_intercept_count} 筆")
    print(f"平均系統延遲 (System Latency): {avg_latency_per_sample:.2f} ms/sample")
    print(f"系統吞吐量 (System Throughput): {throughput:.2f} samples/sec")
    print("="*40 + "\n")

    idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df_result = pd.DataFrame({
        'ID': ids, 
        'Emoji': emojis, 
        'Text': texts, 
        'True_Label_ID': true_labels, 
        'Predicted_Label_ID': pred_labels,
        'Vader_Override': override_flags,
        'Vader_Score': vader_scores_list,
        'Emoji_Net_Positive': net_positives_list
    })
    
    df_result['True_Label'] = df_result['True_Label_ID'].map(idx_to_label)
    df_result['Predicted_Label'] = df_result['Predicted_Label_ID'].map(idx_to_label)
    df_result['Is_Correct'] = df_result['True_Label_ID'] == df_result['Predicted_Label_ID']
    
    df_result['Status'] = df_result.apply(
        lambda x: f"{'✅' if x['Is_Correct'] else '❌'} (VADER Override)" if x['Vader_Override'] 
        else f"{'✅ Correct' if x['Is_Correct'] else '❌ Wrong'}", axis=1
    )
    
    df_final = df_result[['ID', 'Status', 'True_Label', 'Predicted_Label', 'Emoji', 'Text', 'Vader_Score', 'Emoji_Net_Positive']]
    
    actual_output_csv = f"{'stage1_enabled_' if ENABLE_STAGE_1 else 'stage2_only_'}{output_csv}"
    df_final.to_csv(actual_output_csv, index=False, encoding='utf-8-sig')
    print(f"[Info] 詳細報告已儲存至: {actual_output_csv}")
    
    override_df = df_final[df_result['Vader_Override'] == True]
    if not override_df.empty and ENABLE_STAGE_1:
        print("\n" + "!"*50)
        print(f"🚨 觸發 VADER 反諷攔截的 {len(override_df)} 筆樣本明細：")
        print("!"*50)

def plot_gamma_beta(gamma, beta, title="CLN Parameters"):
    plt.figure(figsize=(12, 4))
    dim_len = len(gamma)
    
    plt.subplot(1, 2, 1)
    plt.plot(gamma, alpha=0.7, color='blue')
    plt.title(f"Gamma (Scaling) - {title}")
    plt.xlabel(f"Feature Dimension (0-{dim_len-1})")
    plt.ylabel("Value")
    
    plt.subplot(1, 2, 2)
    plt.plot(beta, alpha=0.7, color='orange')
    plt.title(f"Beta (Shifting) - {title}")
    plt.xlabel(f"Feature Dimension (0-{dim_len-1})")
    plt.ylabel("Value")
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主程式
# ==========================================
if __name__ == "__main__":
    
    tokenizer = CURRENT_MODEL_CFG['tokenizer_cls'].from_pretrained(CURRENT_MODEL_CFG['name'])
    emoji_handler = EmojiHandler(CONFIG['emoji_bin_path'])
    sarcasm_detector = VaderSarcasmDetector(CONFIG['emoji_dict_path'])

    model = FusionModel(
        model_cls=CURRENT_MODEL_CFG['model_cls'], model_name=CURRENT_MODEL_CFG['name'],
        has_pooler=CURRENT_MODEL_CFG['has_pooler'], emoji_input_dim=emoji_handler.vector_size,
        hidden_dim=CONFIG['hidden_dim'], num_classes=CONFIG['num_classes']
    )
    model = model.to(CONFIG['device'])

    checkpoint_path = f"best_{CONFIG['model_type']}{'_with_stage1' if ENABLE_STAGE_1 else '_stage2_only'}_weights.pth"
    output_report_path = f"{CONFIG['model_type']}_inference_report.csv"
    best_accuracy = 0.0

    # ==============================================================
    # 🔴 Analysis 模式：僅讀取權重並進行推論
    # ==============================================================
    if RUN_MODE == 'analysis':
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"[Info] {CONFIG['model_type'].upper()} 本地端權重載入成功，準備進行分析！")
        else:
            print(f"[Error] 未發現對應權重 ({checkpoint_path})，無法進行分析！請先切換至 'train' 模式。")
            sys.exit()

        if os.path.exists(CONFIG['csv_path']):
            df = pd.read_csv(CONFIG['csv_path'], header=None, names=['A', 'B', 'C', 'ID'])
            df['A'] = df['A'].astype(str).str.strip()
            
            _, df_temp = train_test_split(df, test_size=0.2, random_state=42)
            _, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
            
            # 動態隨機抽樣單一外部測試資料
            num_samples = CONFIG.get('num_external_samples', 0)
            ext_path = CONFIG.get('LLM_dataset', '')
            
            if num_samples > 0 and os.path.exists(ext_path):
                print(f"[Info] 正在載入外部測試資料: {ext_path}")
                df_ext = pd.read_csv(ext_path, header=None, names=['A', 'B', 'C', 'ID'])
                df_ext['A'] = df_ext['A'].astype(str).str.strip()
                
                actual_samples = min(num_samples, len(df_ext))
                df_ext_sampled = df_ext.sample(n=actual_samples, random_state=42)
                
                df_test = pd.concat([df_test, df_ext_sampled], ignore_index=True)
                print(f"[Info] 已隨機抽取 {actual_samples} 筆外部資料並合併，當前測試集總筆數: {len(df_test)}")
            elif num_samples == 0:
                print("[Info] 設定抽取外部資料筆數為 0，僅使用原始切分之測試集。")
            else:
                print(f"[Warn] 找不到外部測試資料: {ext_path}，已略過。")
            
            export_test_csv = f"{CONFIG['model_type']}_test_data_analysis.csv"
            df_test.to_csv(export_test_csv, index=False, encoding='utf-8-sig')
            print(f"[Info] 已將測試集原始資料匯出至: {export_test_csv}")

            test_dataset = SentimentDataset(df_test, tokenizer, emoji_handler, sarcasm_detector, max_len=CONFIG['max_len'])
            test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
            
            print("\n[Info] 正在評估測試集準確度...")
            test_acc, test_loss, test_report = eval_model(model, test_loader, CONFIG['device'], desc="Evaluating metrics")
            
            print("\n" + "="*40)
            print("📊 測試集客觀評估結果 (Test Set Metrics)")
            print("="*40)
            print(f"整體準確度 (Accuracy): {test_acc:.4f}")
            print(f"測試集損失 (Loss): {test_loss:.4f}")
            print("\n詳細分類報表 (Classification Report):\n", test_report)
            print("="*40 + "\n")
            
            generate_test_report_with_speed(model, test_loader, CONFIG['device'], output_report_path)
                
        sys.exit()
    
    # ==============================================================
    # 🔵 External Test 模式：僅讀取權重，並對外部測試集進行全面評估與測速
    # ==============================================================
    if RUN_MODE == 'external_test':
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"[Info] {CONFIG['model_type'].upper()} 本地端權重載入成功，準備進行外部資料推論！")
        else:
            print(f"[Error] 未發現對應權重 ({checkpoint_path})，無法進行推論！請確認權重檔案存在。")
            sys.exit()

        # 讀取專供純推論的獨立路徑參數
        ext_path = CONFIG.get('external_infer_data', '')
        if not os.path.exists(ext_path):
            print(f"[Error] 找不到外部測試資料: {ext_path}，請確認路徑與檔案名稱。")
            sys.exit()

        print(f"[Info] 正在載入純外部測試資料: {ext_path}")
        
        # 為了支援可能帶有 Header 或缺少 ID 欄位的 CSV 檔案，這裡使用彈性讀取機制
        try:
            df_ext = pd.read_csv(ext_path)
            # 若欄位名稱非 A, B, C，則強制退回無 Header 讀取法，確保格式對應
            if not all(col in df_ext.columns for col in ['A', 'B', 'C']):
                df_ext = pd.read_csv(ext_path, header=None, names=['A', 'B', 'C', 'ID', 'E', 'F'], usecols=[0, 1, 2, 3])
                # 若原始資料欄位不足4欄，進行補齊
                if len(df_ext.columns) < 4:
                    df_ext.columns = ['A', 'B', 'C']
                    df_ext['ID'] = range(len(df_ext))
            elif 'ID' not in df_ext.columns:
                df_ext['ID'] = range(len(df_ext))
        except Exception:
            df_ext = pd.read_csv(ext_path, header=None, names=['A', 'B', 'C', 'ID'])

        df_ext['A'] = df_ext['A'].astype(str).str.strip()
        print(f"[Info] 外部測試資料載入完成，總筆數: {len(df_ext)}")

        # 建立專屬的 Dataset 與 DataLoader
        ext_dataset = SentimentDataset(df_ext, tokenizer, emoji_handler, sarcasm_detector, max_len=CONFIG['max_len'])
        ext_loader = DataLoader(ext_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
        
        print("\n[Info] 正在評估外部測試集準確度...")
        ext_acc, ext_loss, ext_report = eval_model(model, ext_loader, CONFIG['device'], desc="Evaluating External Data")
        
        print("\n" + "="*40)
        print("📊 外部測試集客觀評估結果 (External Test Metrics)")
        print("="*40)
        print(f"整體準確度 (Accuracy): {ext_acc:.4f}")
        print(f"測試集損失 (Loss): {ext_loss:.4f}")
        print("\n詳細分類報表 (Classification Report):\n", ext_report)
        print("="*40 + "\n")
        
        # 產出推論報告與測量 Throughput
        ext_output_report_path = f"{CONFIG['model_type']}_external_inference_report.csv"
        generate_test_report_with_speed(model, ext_loader, CONFIG['device'], ext_output_report_path)
            
        sys.exit()

    # ==============================================================
    # 🟢 Train 模式：強制從頭開始訓練 (忽略舊權重)
    # ==============================================================
    print(f"\n[Info] 準備從頭開始訓練 {CONFIG['model_type'].upper()} 模型...")
    
    df = pd.read_csv(CONFIG['csv_path'], header=None, names=['A', 'B', 'C', 'ID']) 
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    
    # 動態隨機抽樣單一外部測試資料
    num_samples = CONFIG.get('num_external_samples', 0)
    ext_path = CONFIG.get('LLM_dataset', '')
    
    if num_samples > 0 and os.path.exists(ext_path):
        print(f"[Info] 正在載入外部測試資料: {ext_path}")
        df_ext = pd.read_csv(ext_path, header=None, names=['A', 'B', 'C', 'ID'])
        df_ext['A'] = df_ext['A'].astype(str).str.strip()
        
        actual_samples = min(num_samples, len(df_ext))
        df_ext_sampled = df_ext.sample(n=actual_samples, random_state=42)
        
        df_test = pd.concat([df_test, df_ext_sampled], ignore_index=True)
        print(f"[Info] 已隨機抽取 {actual_samples} 筆外部資料並合併，當前測試集總筆數: {len(df_test)}")
    elif num_samples == 0:
        print("[Info] 設定抽取外部資料筆數為 0，僅使用原始切分之測試集。")
    else:
        print(f"[Warn] 找不到外部測試資料: {ext_path}，已略過。")

    train_dataset = SentimentDataset(df_train, tokenizer, emoji_handler, sarcasm_detector, max_len=CONFIG['max_len'])
    val_dataset = SentimentDataset(df_val, tokenizer, emoji_handler, sarcasm_detector, max_len=CONFIG['max_len'])
    test_dataset = SentimentDataset(df_test, tokenizer, emoji_handler, sarcasm_detector, max_len=CONFIG['max_len'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    total_steps = len(train_loader) * CONFIG['epoch']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(CONFIG['epoch']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epoch']}")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, CONFIG['device'], scheduler)
        print(f"Train loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        
        val_acc, val_loss, _ = eval_model(model, val_loader, CONFIG['device'], desc="Validating")
        print(f"Valid loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"⭐ 發現更好的權重！已儲存至 {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        
    print("\n[Info] 訓練完成，正在使用最佳權重進行測試集最終驗證...")
    test_acc, test_loss, test_report = eval_model(model, test_loader, CONFIG['device'], desc="Testing")
    print(f"Final Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
    print("\nFinal Classification Report:\n", test_report)