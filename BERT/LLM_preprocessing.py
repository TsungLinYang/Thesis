import csv

# 請替換為你的實際檔案名稱
input_filename = '80%test_data_replaced.csv'
output_filename = '80%test_data_replaced_final.csv'

with open(input_filename, mode='r', encoding='utf-8-sig') as infile, \
     open(output_filename, mode='w', encoding='utf-8-sig', newline='') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # 確保該列至少有三個欄位
        if len(row) >= 3:
            # 將第三欄 (row[2]) 的內容與第二欄 (row[1]) 的內容合併，並覆蓋第二欄
            row[1] = str(row[2]) + str(row[1])
            
            # 刪除原本的第三欄
            del row[2]
            
        # 寫入處理後的一列
        writer.writerow(row)

print("處理完成，已儲存至", output_filename)