import pandas as pd
import matplotlib.pyplot as plt
import os

print("="*40)
print("📡 STARTING DATA DOWNLOAD...")
print("="*40)

# 1. تحميل البيانات من مصدر موثوق (GitHub Mirror)
# المصدر: PJM East Region Hourly Data
url = "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv"

try:
    print(f"⬇️  Downloading from: {url}...")
    df = pd.read_csv(url)
    
    # تحويل عمود التاريخ ليكون بصيغة زمنية صحيحة
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True) # ترتيب البيانات زمنياً مهم جداً

    print("✅ Download Successful!")
    print(f"📊 Data Shape: {df.shape} (Rows, Columns)")
    print("-" * 30)
    print("🔍 First 5 rows:")
    print(df.head())

    # 2. حفظ الملف محلياً لاستخدامه لاحقاً
    df.to_csv('PJME_hourly.csv')
    print("💾 Saved local copy as 'PJME_hourly.csv'")

    # 3. رسم عينة من البيانات للتأكد (Sanity Check)
    print("🎨 Generating preview plot...")
    plt.figure(figsize=(15, 5))
    # نأخذ عينة من أول 1000 ساعة فقط للوضوح
    df['PJME_MW'][:1000].plot(style='-', title='PJM East Energy Consumption (First 1000 Hours)')
    plt.ylabel('MW (Megawatts)')
    plt.xlabel('Date')
    
    # حفظ الصورة بدلاً من عرضها
    plot_filename = "data_preview.png"
    plt.savefig(plot_filename)
    print(f"🖼️  Plot saved as '{plot_filename}'")
    
    print("="*40)
    print("🎉 DATA READY FOR TRAINING!")
    print("="*40)

except Exception as e:
    print(f"❌ Error downloading data: {e}")