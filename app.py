import streamlit as st
import pandas as pd
import time
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
import asyncio
import httpx


@st.cache_data
def load_table(url):
    data = pd.read_csv(url)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data


def get_season():
    curr_time = time.localtime()
    curr_month = curr_time.tm_mon
    if curr_month in [12, 1, 2]:
        return "winter"
    elif curr_month in [3, 4, 5]:
        return "spring"
    elif curr_month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def proc_step(df_city):
    df_city = df_city.sort_values('timestamp')
    df_city["moving_mean"] = df_city.temperature.rolling(window=30, min_periods=1).mean()
    df_city["c_s_mean"] = df_city.groupby(["season"])["temperature"].transform("mean")
    df_city["c_s_std"] = df_city.groupby(["season"])["temperature"].transform("std")
    anomaly_lst = []
    df_city.index = list(range(len(df_city)))
    for i in range(len(df_city)):
        curr_std = df_city.loc[i, "c_s_std"]
        curr_mean = df_city.loc[i, "c_s_mean"]
        curr_t = df_city.loc[i, "temperature"]
        if (curr_t < (curr_mean - 2 * curr_std)) or (curr_t > (curr_mean + 2 * curr_std)):
            anomaly_lst.append(True)
        else:
            anomaly_lst.append(False)
    df_city["t_anomaly"] = anomaly_lst
    return df_city


def parallel_process(df):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(proc_step, [proc_step(df[df['city'] == city]) for city in df['city'].unique()]))
    return pd.concat(results)


def simple_process(df):
    results = [proc_step(df[df['city'] == city]) for city in df['city'].unique()]
    return pd.concat(results)


async def get_post_async(city, api_key, client):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = await client.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code}


async def fetch_posts_async(city, api_key):

    async with httpx.AsyncClient() as client:
        tasks = [get_post_async(city, api_key, client)]
        results = await asyncio.gather(*tasks)

    return results


st.title("Анализ данных с использованием Streamlit")

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "data" not in st.session_state:
    st.session_state.data = None
if "proc_df" not in st.session_state:
    st.session_state.proc_df = None
if "proc_time" not in st.session_state:
    st.session_state.proc_time = None
if "temp_df" not in st.session_state:
    st.session_state.temp_df = None

api_key = st.sidebar.text_input("Введите API-ключ:")

uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.data = load_table(uploaded_file)
    st.dataframe(st.session_state.data.head(5))
else:
    st.write("Пожалуйста загрузите CSV-файл")

if st.session_state.uploaded_file is not None:
    processing_method = st.radio(
        "Распараллелить анализ данных?",
        ("Да", "Нет"),
        horizontal=True
    )

    if st.button("Начать обработку данных"):
        start_t = time.time()
        if processing_method == "Да":
            st.session_state.proc_df = parallel_process(st.session_state.data)
        else:
            st.session_state.proc_df = simple_process(st.session_state.data)
        st.session_state.proc_time = time.time() - start_t
        
        temp_group = st.session_state.proc_df.groupby(["city", "season"])["t_anomaly"].sum() / st.session_state.proc_df.groupby(["city", "season"])["t_anomaly"].count()
        temp_city_lst = []
        temp_season_lst = []
        temp_anomaly_percent_lst = []
        temp_mean_lst = []
        temp_std_lst = []
        for curr_city, curr_season in temp_group.keys():
            temp_city_lst.append(curr_city)
            temp_season_lst.append(curr_season)
            temp_mean_lst.append(st.session_state.proc_df[(st.session_state.proc_df.city == curr_city) & (st.session_state.proc_df.season == curr_season)].c_s_mean.values[0].item())
            temp_std_lst.append(st.session_state.proc_df[(st.session_state.proc_df.city == curr_city) & (st.session_state.proc_df.season == curr_season)].c_s_std.values[0].item())
        for item in temp_group.values:
            temp_anomaly_percent_lst.append(round(item * 100, 2))
        st.session_state.temp_df = pd.DataFrame({
            "city": temp_city_lst,
            "season": temp_season_lst,
            "mean": temp_mean_lst,
            "std": temp_std_lst,
            "anomaly_percent": temp_anomaly_percent_lst
        })
        
    if st.session_state.proc_time is not None:    
        st.write(f"Время потраченное на обработку - {st.session_state.proc_time}")

    if st.session_state.temp_df is not None:    
        st.dataframe(st.session_state.temp_df)

    if st.session_state.proc_df is not None:
        city = st.selectbox("Выберите город", st.session_state.data['city'].unique(), key='city_select')

        st.session_state.city = city

        if 'last_city' not in st.session_state:
            st.session_state.last_city = None
        if 'weather_data' not in st.session_state:
            st.session_state.weather_data = {}

        st.title(f"Анализ температур для {st.session_state.city}")
        city_df = st.session_state.proc_df[st.session_state.proc_df['city'] == st.session_state.city]

        fig = px.line(city_df, x='timestamp', y=['temperature', 'moving_mean'], 
              title="Температура и скользящее среднее")
        anomalies = city_df[city_df['t_anomaly']]
        fig.add_scatter(x=anomalies['timestamp'], y=anomalies['temperature'], 
                mode='markers', name='Аномалии')
        st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.city:
            if st.session_state.last_city != st.session_state.city:
                response_json = asyncio.run(fetch_posts_async(st.session_state.city, api_key))[0]
                if "error" in response_json.keys():
                    st.write("Ошибка при обращении к сервису. Проверьте корректность введенного API-ключа")
                    st.session_state.weather_data = {}
                else:
                    season = get_season()
                    mean = st.session_state.proc_df[(st.session_state.proc_df.city == st.session_state.city) & (st.session_state.proc_df.season == season)]["c_s_mean"].values[0]
                    std = st.session_state.proc_df[(st.session_state.proc_df.city == st.session_state.city) & (st.session_state.proc_df.season == season)]["c_s_std"].values[0]
                    curr_t = response_json["main"]["temp"]
                    st.session_state.weather_data = {
                        "last_city": st.session_state.city,
                        "curr_t": response_json["main"]["temp"],
                        "mean": mean,
                        "std": std
                    }
                    st.session_state.last_city = st.session_state.city
            if st.session_state.weather_data:
                weather = st.session_state.weather_data
                st.write(f"Исторически нормальный диапазон для данного сезона от {round(weather['mean'] - weather['std'], 2)} до {round(weather['mean'] + weather['std'], 2)}")
                st.write(f"Текущая температура - {weather['curr_t']}")