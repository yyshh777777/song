# app.py 의 맨 윗부분

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

# 페이지 설정 (가장 먼저 와야 함)
st.set_page_config(layout="centered", page_title="Music Shift Widget")

# 데이터 로드 (캐싱을 통해 속도 향상)
@st.cache_data
def load_data():
    # 파일 경로가 맞는지 확인하세요. 같은 폴더에 있다면 아래처럼 씁니다.
    # encoding='latin1' 을 괄호 안에 추가합니다.
    df = pd.read_csv("spotify_songs.csv", encoding='latin1')
    
    # 결측치 제거 및 필요한 전처리
    df.dropna(inplace=True)
    
    # 중복된 곡 제거 (track_id가 같다면)
    df.drop_duplicates(subset=['track_name', 'track_artist'], inplace=True)
    return df

data = load_data()

# app.py 계속...

def preprocess_data(df):
    # 분석에 사용할 수치형 컬럼들
    features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness',  
        'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- PCA 적용 (Orange의 로직) ---
    # 에너지와 라우드니스를 합쳐서 'Intensity'라는 주성분으로 만듦
    # 실제로는 전체 데이터에 대해 PCA를 돌리지만, 
    # 여기서는 설명하신 '비슷한 요소 합치기'를 위해 명시적으로 파생변수를 만듭니다.
    
    # 1. Intensity (Energy + Loudness)
    df['pca_intensity'] = (df['energy'] + (df['loudness'] / -60)) / 2 # 단순화된 정규화 합산 예시
    
    # 2. Mood/Groove (Danceability + Valence)
    df['pca_groove'] = (df['danceability'] + df['valence']) / 2
    
    # 추천에 사용할 최종 Feature 리스트 재정의
    # 원래 컬럼은 유지하되, 추천 계산 시에는 PCA된 값을 가중치로 쓸 수도 있습니다.
    # 여기서는 심플하게 원본 데이터를 정규화한 값을 사용하되, 
    # 사용자가 'Intensity'를 고르면 energy와 loudness를 동시에 고려하도록 설계합니다.
    
    return df, X_scaled, scaler, features

df, X_scaled, scaler, feature_list = preprocess_data(data)

# NearestNeighbors 모델 학습 (모든 특성 기반)
knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
knn_model.fit(X_scaled)


# app.py 계속...

def recommend_songs(input_song_index, change_feature, df, X_scaled, model):
    """
    input_song_index: 사용자가 고른 노래의 인덱스
    change_feature: 사용자가 바꾸고 싶어하는 요소 (예: 'tempo')
    """
    
    # 1. 일단 전체적으로 가장 비슷한 노래 50개를 찾습니다 (후보군)
    distances, indices = model.kneighbors([X_scaled[input_song_index]], n_neighbors=50)
    
    candidate_indices = indices[0][1:] # 0번은 자기 자신이므로 제외
    candidates = df.iloc[candidate_indices].copy()
    
    original_value = df.iloc[input_song_index][change_feature]
    
    # 2. 그 중에서 선택한 요소(change_feature)가 원곡과 차이가 나는 것을 찾습니다.
    # 예: 원곡 템포가 120이면, 120과 많이 다른(아주 빠르거나 아주 느린) 곡을 추천
    
    # 차이(Diff) 계산
    candidates['diff'] = abs(candidates[change_feature] - original_value)
    
    # 차이가 큰 순서대로 정렬하여 상위 5개 추출
    recommendations = candidates.sort_values(by='diff', ascending=False).head(5)
    
    return recommendations


# app.py 계속...

# --- CSS 스타일링 (위젯 느낌 내기) ---
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stButton>button {
        border-radius: 20px;
        width: 100%;
        border: 2px solid #1DB954; /* Spotify Green */
    }
    .center-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px;
        border-radius: 50%; /* 원형 이미지 */
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎧 Mood Shifter Widget")
st.caption("좋아하는 노래를 입력하고, 바꾸고 싶은 '단 하나의 느낌'을 선택하세요.")

# 1. 노래 검색 (Selectbox 사용)
search_query = st.selectbox(
    "내 최애곡 검색하기:",
    options=df['track_name'] + " - " + df['track_artist']
)

if search_query:
    # 선택한 노래의 인덱스 찾기
    selected_track_name = search_query.split(" - ")[0]
    selected_idx = df[df['track_name'] == selected_track_name].index[0]
    selected_song = df.iloc[selected_idx]

    # --- 위젯 레이아웃 구현 ---
    st.markdown("---")
    
    # 중앙: 선택한 노래 정보 (앨범 커버 대신 원형 차트로 시각화)
    # 실제 데이터셋에는 이미지 URL이 없으므로 Plotly로 원형 느낌을 냅니다.
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"<div style='text-align: center;'><h3>{selected_song['track_name']}</h3></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; color:gray;'>{selected_song['track_artist']}</div>", unsafe_allow_html=True)
        
        # 시각적 재미를 위한 Radar Chart (중앙 원)
        categories = ['energy', 'danceability', 'valence', 'acousticness', ]
        values = [selected_song[c] for c in categories]
        
        fig = go.Figure(data=go.Scatterpolar(
              r=values,
              theta=categories,
              fill='toself',
              name=selected_song['track_name']
        ))
        fig.update_layout(
          polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
          showlegend=False,
          margin=dict(l=20, r=20, t=20, b=20),
          height=250,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("👆 위 원형 차트는 이 노래의 현재 성분입니다.")

    st.markdown("### 🔀 무엇을 다르게 듣고 싶나요?")
    st.write("가운데 원(현재 노래)을 중심으로, 바꾸고 싶은 요소를 클릭하세요.")

    # 버튼들을 원형으로 배치할 순 없으므로, 그리드로 배치하되 직관적으로 만듭니다.
    # 사용자가 선택할 수 있는 옵션 (PCA 적용 개념 포함)
    
    # 버튼 레이아웃 (3열)
    b_col1, b_col2, b_col3 = st.columns(3)
    
    target_feature = None
    
    with b_col1:
        if st.button("🔥 Energy\n(분위기 반전)"):
            target_feature = 'energy'
        if st.button("🎻 Acousticness\n(전자음 vs 어쿠스틱)"):
            target_feature = 'acousticness'
            
    with b_col2:
        if st.button("🏃 Tempo\n(속도만 다르게)"):
            target_feature = 'tempo'
        if st.button("🕺 Danceability\n(그루브 변경)"):
            target_feature = 'danceability'

    with b_col3:
        if st.button("😊 Valence\n(우울 vs 행복)"):
            target_feature = 'valence'
        
    # --- 추천 결과 출력 ---
    if target_feature:
        st.markdown("---")
        st.success(f"**{target_feature.upper()}** 요소만 색다른 곡을 찾았습니다!")
        
        recs = recommend_songs(selected_idx, target_feature, df, X_scaled, knn_model)
        
        for idx, row in recs.iterrows():
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.subheader(row['track_name'])
                    st.text(f"Artist: {row['track_artist']}")
                with c2:
                    # 변화된 수치 보여주기
                    diff_val = row[target_feature]
                    origin_val = selected_song[target_feature]
                    st.metric(label=target_feature, value=round(diff_val, 2), delta=round(diff_val - origin_val, 2))
                st.markdown("---")
