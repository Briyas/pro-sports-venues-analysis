import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri Temizleme ve Anlamlandırma (Data Cleaning & EDA)


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


df = pd.read_csv('pro_venues_clean_final_2.0.csv')

print(f"Toplam Satır Sayısı: {df.shape[0]}")
print(f"Toplam Sütun Sayısı: {df.shape[1]}")
df.head()


# 1. Sütun tipleri ve bellek kullanımı

print("--- Bilgi ve Veri Tipleri ---")
df.info()

# 2. Eksik (Null/NaN) değerlerin tespiti

print("\n--- Eksik Değer Dağılımı ---")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Eksik Değer Sayısı': missing, 'Yüzde (%)': missing_percent})
print(missing_df[missing_df['Eksik Değer Sayısı'] > 0])


# 3. Sayısal sütunların genel istatistikleri (Kapasite, Açılış Yılı vb.)

df.describe().T


# # Temiz veri için kopya oluşturulması ve verilerin düzenlenmesi

clean_df = df.copy()

# 1. Sütun isimlerindeki boşlukları temizleyelim ve küçük harfe çevirilmesi (Standardizasyon)

clean_df.columns = clean_df.columns.str.strip().str.lower().str.replace(' ', '_')

# 2. Sayısal olması gereken alanları temizleme

if 'capacity' in clean_df.columns:
    if clean_df['capacity'].dtype == 'object':
        clean_df['capacity'] = clean_df['capacity'].astype(str).str.replace(',', '').str.strip()
        clean_df['capacity'] = pd.to_numeric(clean_df['capacity'], errors='coerce')

# 3. Yıl sütunu düzeltme 

year_cols = [col for col in clean_df.columns if 'year' in col or 'open' in col]
for col in year_cols:
    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

# 4. Metin sütunlarındaki gereksiz boşlukların kırpılması (Strip)

text_cols = clean_df.select_dtypes(include=['object']).columns
for col in text_cols:
    clean_df[col] = clean_df[col].astype(str).str.strip()

# 5. Koordinat kontrolü (Enlem / Boylam geçerli sınırlar içinde mi?)

if 'latitude' in clean_df.columns and 'longitude' in clean_df.columns:
    clean_df['latitude'] = pd.to_numeric(clean_df['latitude'], errors='coerce')
    clean_df['longitude'] = pd.to_numeric(clean_df['longitude'], errors='coerce')
    
    # Hatalı koordinatları filtreleme
    valid_coords = clean_df['latitude'].between(-90, 90) & clean_df['longitude'].between(-180, 180)
    print(f"Geçersiz koordinat sayısı: {(~valid_coords).sum()}")

print("Veri temizleme adımları tamamlandı.")


# # 1. Liglere göre takım ve stadyum sayıları

if 'league' in clean_df.columns:
    print("\n--- Lig Başına Takım / Stadyum Sayısı ---")
    print(clean_df['league'].value_counts())

# 2. Liglere göre ortalama stadyum kapasiteleri

if 'league' in clean_df.columns and 'capacity' in clean_df.columns:
    print("\n--- Lig Başına Ortalama ve Medyan Kapasiteler ---")
    print(clean_df.groupby('league')['capacity'].agg(['count', 'mean', 'median', 'min', 'max']))

# 3. En çok stadyuma ev sahipliği yapan şehirler/eyaletler

location_col = 'state' if 'state' in clean_df.columns else ('city' if 'city' in clean_df.columns else None)
if location_col:
    print(f"\n--- En Çok Stadyum Bulunan İlk 10 {location_col.title()} ---")
    print(clean_df[location_col].value_counts().head(10))


# # # Elde edilen veriler sonucunda ekstra görselleştirme ve analizler

print("="*50)
print("1. ORTAK KULLANILAN STADYUMLAR / ARENALAR")
print("="*50)

# Aynı arenada oynayan birden fazla takım var mı?

shared_arenas = df.groupby('arena').agg(
    takim_sayisi=('team', 'count'),
    takimlar=('team', lambda x: ', '.join(x)),
    ligler=('league', lambda x: ', '.join(x))
).reset_index()
shared = shared_arenas[shared_arenas['takim_sayisi'] > 1].sort_values(by='takim_sayisi', ascending=False)
print(f"Birden fazla takımın paylaştığı arena sayısı: {len(shared)}")
print(shared.to_string(index=False))
print("\n" + "="*50)
print("2. EYALETLERE (STATE/PROVINCE) GÖRE DAĞILIM")
print("="*50)
top_states = df['state_province'].value_counts().head(10)
print(top_states)

# Grafik : En Çok Takıma Sahip İlk 10 Eyalet (Lig Dağılımıyla)

plt.figure(figsize=(12, 6))
top_state_names = top_states.index.tolist()
df_top_states = df[df['state_province'].isin(top_state_names)]
sns.countplot(
    data=df_top_states, 
    x='state_province', 
    hue='league', 
    order=top_state_names,
    palette='Set2'
)
plt.title('En Fazla Profesyonel Spor Takımına Sahip İlk 10 Eyalet / Bölge', fontsize=14, weight='bold')
plt.xlabel('Eyalet / Bölge', fontsize=12)
plt.ylabel('Takım Sayısı', fontsize=12)
plt.legend(title='Lig')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_states_by_league.png', dpi=300)
plt.show()
print("\nGrafik 'top_states_by_league.png' olarak kaydedildi.")