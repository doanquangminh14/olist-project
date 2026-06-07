import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. ĐỌC DỮ LIỆU ĐÃ LÀM SẠCH (Đã sửa lại đường dẫn đúng)
print("Đang tải dữ liệu...")
order_items = pd.read_csv("../data/clean/order_items_clean.csv")
orders = pd.read_csv("../data/clean/orders_clean.csv")
products = pd.read_csv("../data/clean/products_clean.csv")
customers = pd.read_csv("../data/clean/customers_clean.csv")
sellers = pd.read_csv("../data/clean/sellers_clean.csv")

# 2. GỘP CÁC BẢNG DỮ LIỆU (MERGE DATA)
print("Đang gộp các bảng dữ liệu...")
# Gộp item với thông tin sản phẩm
df = pd.merge(order_items, products, on="product_id", how="inner")
# Gộp với thông tin đơn hàng để lấy mã khách hàng
df = pd.merge(df, orders, on="order_id", how="inner")
# Gộp với thông tin khách hàng để lấy bang (customer_state)
df = pd.merge(df, customers, on="customer_id", how="inner")
# Gộp với thông tin người bán để lấy bang (seller_state)
df = pd.merge(df, sellers, on="seller_id", how="inner")

# Chọn các cột tính năng (Features) và mục tiêu (Target)
# Chúng ta sẽ dùng: khối lượng, kích thước sản phẩm, bang khách hàng, bang người bán để dự đoán freight_value
features = [
    "product_weight_g", 
    "product_length_cm", 
    "product_height_cm", 
    "product_width_cm",
    "customer_state",
    "seller_state"
]
target = "freight_value"

# Lọc bỏ các dòng bị thiếu dữ liệu ở các biến này nếu có
df_ml = df[features + [target]].dropna()

X = df_ml[features]
y = df_ml[target]

print(f"Tổng số mẫu dữ liệu đưa vào huấn luyện: {len(df_ml)}")

# 3. CHUẨN BỊ PIPELINE XỬ LÝ DỮ LIỆU (PREPROCESSING)
# Phân loại biến số và biến chữ (Categorical)
numeric_features = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
categorical_features = ["customer_state", "seller_state"]

# Xử lý biến số: Điền giá trị thiếu (nếu có) bằng trung vị
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Xử lý biến chữ: Chuyển dạng chữ sang dạng số (One-Hot Encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Gộp 2 quá trình xử lý lại
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. ĐỊNH NGHĨA MÔ HÌNH (PIPELINE GỒM TIỀN XỬ LÝ + THUẬT TOÁN)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# 5. CHIA TẬP DỮ LIỆU TRAIN / TEST (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. HUẤN LUYỆN MÔ HÌNH (TRAINING)
print("Đang huấn luyện mô hình Random Forest Regressor (quá trình này có thể mất vài phút)...")
model_pipeline.fit(X_train, y_train)
print("Huấn luyện thành công!")

# 7. ĐÁNH GIÁ MÔ HÌNH (EVALUATION)
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n================ KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ================")
print(f"Sai số tuyệt đối trung bình (MAE): {mae:.2f}")
print(f"Sai số căn phương sai trung bình (RMSE): {rmse:.2f}")
print(f"Độ chính xác R-squared (R2 Score): {r2:.4f} (Càng gần 1 càng tốt)")
print("==========================================================")

# 8. THỬ NGHIỆM DỰ ĐOÁN MỘT ĐƠN HÀNG MỚI
print("\nVí dụ dự đoán cho 1 đơn hàng mới:")
sample_data = pd.DataFrame([{
    "product_weight_g": 1500.0,
    "product_length_cm": 30.0,
    "product_height_cm": 10.0,
    "product_width_cm": 20.0,
    "customer_state": "SP",
    "seller_state": "SP"
}])

predicted_freight = model_pipeline.predict(sample_data)
print(f"Cấu hình sản phẩm: Nặng 1.5kg, kích thước 30x10x20cm, Vận chuyển nội tỉnh (SP -> SP)")
print(f"==> Chi phí vận chuyển dự đoán: {predicted_freight[0]:.2f}")