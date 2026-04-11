import csv

input_file = r'f:\kuliah\uu-ite\Revisi\dataset_no_duplicates.csv'
output_file = r'f:\kuliah\uu-ite\Revisi\dataset_no_duplicates_temp.csv'

data = []
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    headers = [h for h in reader.fieldnames if h != 'normal_teks']
    
    for row in reader:
        new_row = {k: row[k] for k in headers}
        data.append(new_row)

print(f'Total baris: {len(data)}')
print(f'Kolom tersisa: {headers}')

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(data)

print('Selesai! File sementara disimpan ke:', output_file)
