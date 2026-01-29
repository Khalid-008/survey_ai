import pandas as pd
import os

# المسارات
input_file = r'exports/survey_data_enriched_S25120024_20260128_190104.csv'
output_file = r'exports/top_topics_analysis.xlsx'

# التأكد من وجود الملف
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found!")
    exit()

# قراءة الملف (CSV)
print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

# تحديد الأعمدة: المواضيع (قبل الأخير) والمشاعر (الأخير)
topic_col = df.columns[-2]
sentiment_col = df.columns[-4]

# فصل المواضيع وتنظيفها
df['topics_cleaned'] = df[topic_col].str.split(',')
df_exploded = df.explode('topics_cleaned')
df_exploded['topics_cleaned'] = df_exploded['topics_cleaned'].str.strip()

# إزالة "متنوع" والقيم الفارغة
df_exploded = df_exploded[df_exploded['topics_cleaned'].notna()]
df_exploded = df_exploded[df_exploded['topics_cleaned'] != '']
df_exploded = df_exploded[df_exploded['topics_cleaned'] != 'متنوع']

# 1. حساب التكرارات العامة (أهم 10 مواضيع)
top_10 = df_exploded['topics_cleaned'].value_counts().head(20)

# 2. حساب أكثر 5 مواضيع مرتبطة بمشاعر سلبية
negative_df = df_exploded[df_exploded[sentiment_col].astype(str).str.contains('negative', case=False, na=False)]
top_5_negative = negative_df['topics_cleaned'].value_counts().head(20)

# عرض النتائج
print("\nTop 10 Topics (Overall):")
print(top_10)
print("\nTop 5 Negative Topics:")
print(top_5_negative)

# حفظ النتائج في ملف Excel واحد بأوراق عمل منفصلة
print(f"\nSaving results to {output_file}...")
with pd.ExcelWriter(output_file) as writer:
    top_10.to_excel(writer, sheet_name='أهم 10 مواضيع', header=['العدد'])
    top_5_negative.to_excel(writer, sheet_name='أكثر 5 مواضيع سلبية', header=['العدد'])

print("Done!")
