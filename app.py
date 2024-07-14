import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Path to the CSV file
csv_file_path = 'streamlit_data.csv'

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data(csv_file_path)

# Assuming the date column is named 'date'
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
else:
    st.error("Date column not found in the CSV file.")

def preprocess_data(df):
    cutoff_date = pd.to_datetime('2020-09-28')
    cols_to_zero = ['cabbage', 'radish', 'garlic', 'onion', 'daikon', 'cilantro', 'artichoke']
    df.loc[df.index > cutoff_date, cols_to_zero] = np.nan
    return df

def plot_predictions_over_time(df, vegetables, rolling_mean_window):
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    num_colors = len(colors)

    for i, veg in enumerate(vegetables):
        # Plot actual values
        ax.plot(df.index, df[veg], label=veg, linewidth=2, color=colors[i % num_colors])
        
        # Calculate and plot rolling mean based on user-selected window size
        rolling_mean = df[veg].rolling(window=rolling_mean_window).mean()
        ax.plot(df.index, rolling_mean, label=f'{veg} ({rolling_mean_window}-day Rolling Mean)', linestyle='--', color=colors[i % num_colors])
    
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, color='lightgrey', linestyle='--')
    fig.tight_layout()

    # Save the Matplotlib plot to a BytesIO object
    image_stream = BytesIO()
    fig.savefig(image_stream, format='png')
    plt.close(fig)  # Close the Matplotlib figure to free up memory

    # Display the plot as an image in Streamlit
    st.image(image_stream, caption='Overlapped Charts', use_column_width=True)

df = preprocess_data(df)

metric_summary = pd.read_csv("metric_summary.csv")
metric_summary.set_index('product', inplace=True)

# Streamlit app layout and interaction code here
st.title('🍇농산물 가격 예측 대시보드🥭')
st.markdown("""
    왼쪽에서 품목과 예측모델, 날짜를 입력하면 특정기간 이후 예측 가격이 표시됩니다.
    """)

st.sidebar.title('조회 기간')
start_date = st.sidebar.date_input('시작일', df.index.min())
end_date = st.sidebar.date_input('마지막일', df.index.max())

st.sidebar.title('품목을 선택해주세요')
sorted_vegetables = sorted(df.columns)
vegetables = st.sidebar.multiselect('조회 품목:', sorted_vegetables)
rolling_mean_window = st.sidebar.slider('Rolling Mean Window', min_value=1, max_value=30, value=7)

filtered_df = df.loc[start_date:end_date]

if vegetables:
    st.subheader('품목별 예측 대시보드')
    plot_predictions_over_time(filtered_df, vegetables, rolling_mean_window)

    st.subheader('가격 통계')
    mean_percentiles_df = calculate_mean_percentiles(df, vegetables)
    st.write(mean_percentiles_df.reset_index().rename(columns={'index': 'Product'}))  # Reset index and rename columns

if st.checkbox('Show Filtered DataFrame'):
    st.write(filtered_df)

st.subheader('정확도 Summary')
st.write(metric_summary)
