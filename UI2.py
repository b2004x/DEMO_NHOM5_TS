import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


# ============== CONFIG ==================
# === Configuration ===
csv_files = sorted(glob.glob(r"G:\hoc\Time series\UI\data_chunks_2GB\*.csv"))[:30]
timesteps = 20
batch_size = 64
input_size = 8
store_id_list = ["Cửa hàng 1","Cửa hàng 2"]
family_ids_list = ["Deli","Eggs"]
store_id_Name = st.selectbox("Chọn store_id", options=store_id_list, )
family_id_Name = st.selectbox("Chọn family_id", options=family_ids_list)
if store_id_Name == "Cửa hàng 1":
    store_id = 0.0377358490566037
elif store_id_Name == "Cửa hàng 2":
    store_id = 0.0188679245283018
if family_id_Name == "Deli":
    family_id = 0.46875
elif family_id_Name == "Eggs":
    family_id = 0.6875
if st.button("Tiếp tục"):
    st.success("Đã chọn xong!")
    st.write("✅ Store ID:", store_id_Name)
    st.write("✅ Family ID:", family_id_Name)


    # === Fit MinMaxScaler globally on all data (first 100k rows) ===
    df_raw = pd.read_csv(csv_files[0]).dropna()
    feature_cols = ['onpromotion_scaled', 'year_scaled', 'month_sin', 'month_cos',
                    'day_sin', 'day_cos', 'family_scaled', 'store_scaled']
    target_col = 'sales_scaled'

    scaler = MinMaxScaler().fit(df_raw[feature_cols])
    # ============== MODEL ==================
    # === Dataset ===
    class ChunkedTimeSeriesDataset(Dataset):
        def __init__(self, csv_files, timesteps, feature_cols, target_col, scaler, store_id, family_id):
            self.csv_files = csv_files
            self.timesteps = timesteps
            self.feature_cols = feature_cols
            self.target_col = target_col
            self.buffer = []
            self.file_idx = 0
            self.chunk_iter = None
            self.scaler = scaler
            self.store_id = store_id
            self.family_id = family_id
            self._load_next_chunk()

        def _load_next_chunk(self):
            while self.file_idx < len(self.csv_files):
                file = self.csv_files[self.file_idx]
                self.chunk_iter = pd.read_csv(file, chunksize=100_000)
                self.file_idx += 1
                try:
                    chunk = next(self.chunk_iter).dropna()
                    chunk = chunk[(chunk['store_scaled'] == self.store_id) &
                                (chunk['family_scaled'] == self.family_id)]
                    if chunk.empty:
                        continue
                    scaled_features = self.scaler.transform(chunk[self.feature_cols])
                    targets = chunk[self.target_col].values.reshape(-1, 1)
                    self.buffer = np.hstack((targets, scaled_features)).tolist()
                except StopIteration:
                    continue

        def __len__(self):
            return int(1e5)

        def __getitem__(self, idx):
            while len(self.buffer) < self.timesteps + 1:
                self._load_next_chunk()
                if len(self.buffer) < self.timesteps + 1:
                    return self.__getitem__(idx + 1)

            start = np.random.randint(0, len(self.buffer) - self.timesteps - 1)
            window = np.array(self.buffer[start:start + self.timesteps + 1])
            return torch.tensor(window[:-1, 1:], dtype=torch.float32), torch.tensor(window[-1, 0], dtype=torch.float32)

    # === TCN Model ===
    class Chomp1d(nn.Module):
        def __init__(self, chomp_size):
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            return x[:, :, :-self.chomp_size].contiguous()

    class TemporalBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                    self.conv2, self.chomp2, self.relu2, self.dropout2)
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class TCNModel(nn.Module):
        def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
            super().__init__()
            layers = []
            for i in range(len(num_channels)):
                dilation_size = 2 ** i
                in_channels = input_size if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                        dropout=dropout)]
            self.network = nn.Sequential(*layers)
            self.linear = nn.Linear(num_channels[-1], 1)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            out = self.network(x)
            out = out[:, :, -1]
            return self.linear(out)

    # ============== STREAMLIT UI ==================
    st.set_page_config(page_title="📈 Dự báo với mô hình TCN", layout="centered")
    st.title("🔮 Dự đoán doanh số 7 ngày tới bằng mô hình TCN")

    device = torch.device("cpu")

    dataset = ChunkedTimeSeriesDataset(csv_files, timesteps, feature_cols, target_col,
                                        scaler, store_id, family_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = TCNModel(input_size=input_size, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    st.success("✅ Đã lọc dữ liệu theo store/family.")

    model = TCNModel(input_size=input_size, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2)

    # 2. Load trọng số từ file đã lưu
    model.load_state_dict(torch.load(rf"G:\hoc\Time series\UI\Models\tcn_model_store{store_id}_family{family_id}.pth", map_location=device))
    
    # 3. Chuyển sang chế độ đánh giá và đúng thiết bị
    model.to(device)
    model.eval()


    # === Evaluation ===
    st.subheader("📉 Biểu đồ Dự đoán vs Thực tế")
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
            if i >= 100:
                break

    min_y, scale_y = scaler.min_[0], scaler.scale_[0]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = y_pred * scale_y + min_y
    y_true = y_true * scale_y + min_y

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)

    print("\n\U0001F4CA Evaluation Metrics:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label="Thực tế", color="blue")
    ax.plot(y_pred, label="Dự đoán", color="orange", linestyle="--")
    ax.legend()
    ax.set_title("So sánh Dự đoán và Thực tế")
    st.pyplot(fig)

    st.markdown(f"""
    **📊 Metrics:**
    - MAE: `{mae:.2f}`
    - RMSE: `{rmse:.2f}`
    - R²: `{r2:.2f}`
    """)

    # === Forecast 7 ngày tới ===
    forecast_steps = 7
    initial_sequence = [dataset.buffer[i][1:] for i in range(len(dataset.buffer) - timesteps, len(dataset.buffer))]
    input_seq = torch.tensor([initial_sequence], dtype=torch.float32).to(device)

    multi_preds = []
    model.eval()
    with torch.no_grad():
        for _ in range(forecast_steps):
            pred = model(input_seq).squeeze().cpu().item()
            multi_preds.append(pred)
            last_known_features = input_seq[0, -1, :].cpu().numpy()
            new_row = last_known_features.copy()
            input_seq = input_seq.cpu().numpy()
            input_seq = np.append(input_seq[0], [new_row], axis=0)[-timesteps:]
            input_seq = torch.tensor([input_seq], dtype=torch.float32).to(device)

    multi_preds = np.array(multi_preds)
    multi_preds_rescaled = multi_preds * scale_y + min_y

    true_last_30 = [row[0] for row in dataset.buffer[-30:]]
    true_last_30 = np.array(true_last_30) * scale_y + min_y

    combined = np.concatenate([true_last_30, multi_preds_rescaled])
    steps = np.arange(1, len(combined) + 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(steps[:30], true_last_30, label="Thực tế (30 ngày)", color="blue", linewidth=2)
    ax.plot(steps[30:], multi_preds_rescaled, label="Dự đoán (7 ngày)", color="orange", linestyle="--", linewidth=2)
    ax.axvline(x=30, color='gray', linestyle=':', label='Bắt đầu dự đoán')
    ax.set_title(f"📈 Dự đoán 7 ngày tới (Store {store_id} - Family {family_id})")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Hiển thị số
    st.markdown("### 📊 Giá trị thực tế (30 ngày gần nhất):")
    st.write(np.round(true_last_30, 2))

    st.markdown("### 🔮 Dự đoán doanh số 7 ngày tới:")
    for i, val in enumerate(multi_preds_rescaled, 1):
        st.write(f"Ngày +{i}: **{val:.2f}**")
# Tạo thư mục lưu nếu chưa có


    base_dir = r"G:\hoc\Time series"
    save_dir = os.path.join(base_dir, f"results", f"store_{store_id}_family_{family_id}")
    os.makedirs(save_dir, exist_ok=True)

    # --- Lưu biểu đồ dự đoán vs thực tế ---
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(y_true, label="Thực tế", color="blue")
    ax1.plot(y_pred, label="Dự đoán", color="orange", linestyle="--")
    ax1.legend()
    ax1.set_title("So sánh Dự đoán và Thực tế")
    fig1.savefig(os.path.join(save_dir, "eval_plot.png"))

    # --- Lưu biểu đồ 30 ngày thực tế + 7 ngày dự đoán ---
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(steps[:30], true_last_30, label="Thực tế (30 ngày)", color="blue", linewidth=2)
    ax2.plot(steps[30:], multi_preds_rescaled, label="Dự đoán (7 ngày)", color="orange", linestyle="--", linewidth=2)
    ax2.axvline(x=30, color='gray', linestyle=':', label='Bắt đầu dự đoán')
    ax2.set_title(f"Dự đoán 7 ngày tới (Store {store_id} - Family {family_id})")
    ax2.set_xlabel("Ngày")
    ax2.set_ylabel("Sales")
    ax2.legend()
    ax2.grid(True)
    fig2.savefig(os.path.join(save_dir, "forecast_plot.png"))

    # --- Lưu metrics ---
    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # --- Lưu dữ liệu ---
    np.save(os.path.join(save_dir, "true_last_30.npy"), true_last_30)
    np.save(os.path.join(save_dir, "forecast_next_7.npy"), multi_preds_rescaled)