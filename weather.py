# -*- coding: utf-8 -*-
"""weather.ipynb

# 패키지 임포트
"""

# !pip install lxml
# !pip install datasets
# !pip install haversine

import requests
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS


from datetime import datetime, timedelta
from lxml import etree
from haversine import haversine
from datasets import load_dataset


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})



"""# 함수 정의
반드시 순서대로 호출할 것
"""
"""
사용자 입력값
basedate : 현재 날짜 "%Y%m%d" 형식
basetime : 현재 시간 "%H%M" 형식
ny : 사용자 위도(Latitude)
nx : 사용자 경도(Longitude)
"""



"""내 좌표를 넣으면 가장 가까운 기상청 x,y좌표로 변환하는 함수"""

# 내 좌표를 기상청 x,y좌표로 변환

def find_nearest_grid(my_loc):
  grid = None
  min_distance = float('inf')  # 무한대 값으로 초기화

  # 데이터셋 라이브러리를 사용하여 기상청 데이터셋 로드 - 로컬에서 안불러와도됨
  dataset = load_dataset("hscrown/weather_api_info")
  kor_loc = pd.DataFrame(dataset['train'])
  kor_loc = kor_loc.iloc[:,:15] # 필요한 컬럼만 추출
  kor_loc = kor_loc.dropna() # 2단계와 3단계가 모두 존재하는 행만 추출

  for index, row in kor_loc.iterrows():
      # 각 격자 지점에 대한 튜플 (위도, 경도)을 생성
      grid_point = (row['위도(초/100)'], row['경도(초/100)'])

      # haversine 공식을 사용하여 거리계산
      distance = haversine(my_loc, grid_point)

      # 가장 가까운 거리를 찾으면 정보를 업데이트
      if distance < min_distance:
          min_distance = distance
          grid = row
          nx = grid['격자 X']
          ny = grid['격자 Y']

      return grid,nx,ny

"""내 위치와 가장가까운 공원, 박물관 찾아주는 함수정의"""

def find_nearest_place(my_loc, df):
    min_distance = float('inf')  # 무한대 값으로 초기화
    lat, long = None, None  # 초기 위치 변수 선언

    for index, row in df.iterrows():
        # 각 격자 지점에 대한 튜플 (위도, 경도)을 생성
        grid_point = (row['LATITUDE'], row['LONGITUDE'])

        # haversine 공식을 사용하여 거리 계산
        distance = haversine(my_loc, grid_point)

        # 가장 가까운 거리를 찾으면 정보를 업데이트
        if distance < min_distance:
            min_distance = distance
            lat = row['LATITUDE']
            long = row['LONGITUDE']

    return lat, long  

"""좌표와 시간을 넣으면 기상정보를 제공하는 *함수*"""

def get_weather_info(base_date,base_time,nx,ny):

  # 초단기실황데이터
  url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'
  params ={
      'serviceKey': 'sX3JWddMWHJxC43fx9mqgcqSsbmAlTpoFTUPbnrE1Db5uVnEAs7gJIL4Z3tzW1u2S6UC+8/go3xYCnG2wDctAQ==',
      'pageNo': '1',
      'numOfRows': '1000',
      'dataType': 'XML',
      'base_date': base_date,
      'base_time': base_time,
      'nx': nx,
      'ny': ny
  }

  response = requests.get(url, params=params)
  root = etree.fromstring(response.content)

  # 데이터 파싱 및 추출
  # category = root.xpath('//category/text()')[0] # 0:강수형태, 2:습도, 3:기온, 4:풍속
  rain = root.xpath('//obsrValue/text()')[0] # 강수
  temp = root.xpath('//obsrValue/text()')[3] # 기온

  mapping = {
    '0': "비가 오고 있지 않습니다.",
    '1': "비 소식이 있습니다.",
    '2': "비 또는 눈이 내립니다.",
    '3': "눈이 오고 있습니다.",
    '4': "소나기가 옵니다.",
    '5': "빗방울이 떨어집니다.",
    '6': "빗방울과 눈날림이 있습니다.",
    '7': "눈날림이 있습니다."
  }

  rain = mapping.get(rain)

    # 초단기예보데이터
  url2 = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'


  response2 = requests.get(url2, params=params)
  root2 = etree.fromstring(response2.content)

  # 엘리먼트 선택
  items = root2.xpath('//item')

  # 딕셔너리로 만들기
  data = [{
      "baseDate": item.findtext("baseDate"),
      "baseTime": item.findtext("baseTime"),
      "category": item.findtext("category"),
      "fcstDate": item.findtext("fcstDate"),
      "fcstTime": item.findtext("fcstTime"),
      "fcstValue": item.findtext("fcstValue"),
      "nx": item.findtext("nx"),
      "ny": item.findtext("ny")
  } for item in items]

  # 데이터프레임으로 만들기
  df = pd.DataFrame(data)
  df = df[df['fcstDate'] == df['baseDate']] # 오늘 예측 값만
  # df = df[df['fcstTime'] == df['baseTime']]
  df

  sky_dict = {
      '1': "맑음",
      '2': "구름조금",
      '3': "구름많음",
      '4': "흐림"
  }

  # 30분뒤 하늘상태는
  df = df[df['category'] == 'SKY']['fcstValue'].map(sky_dict)
  sky = df.values[0]


  return rain,temp,sky

"""박물관 데이터 전처리 코드"""

def get_muse_data():
  muse = load_dataset("hscrown/seoul_museums")
  muse = muse['Train']

  # 컬럼명 변경
  muse.rename(columns={'위도':'LATITUDE','경도':'LONGITUDE'},inplace=True)

  # ['LATITUDE'] 컬럼을 실수로 변경
  # 결측행 삭제
  muse['LATITUDE'].replace('', np.nan, inplace=True)
  muse['LONGITUDE'].replace('', np.nan, inplace=True)
  muse = muse.dropna()

  # ['LATITUDE'] 컬럼을 실수로 변경
  muse['LATITUDE'] = muse['LATITUDE'].astype(float)

  # ['LONGITUDE'] 컬럼을 실수로 변경
  muse['LONGITUDE'] = muse['LONGITUDE'].astype(float)

  return muse


"""공원데이터 전처리 코드"""

def get_park_data():
  # API 요청
  start_point = 1
  end_point = 1000 # 최대 1000개까지만 호출 할 수 있음
  seoul_key = '57524f76506d656e3732636a52457a'


  url = f'http://openAPI.seoul.go.kr:8088/{seoul_key}/json/SearchParkInfoService/{start_point}/{end_point}/'

  park = requests.get(url).json()
  park.keys() # ['SearchParkInfoService']

  park = pd.DataFrame(park['SearchParkInfoService']['row'])

  # ['LATITUDE'] 컬럼을 실수로 변경
  # 결측행 삭제
  park['LATITUDE'].replace('', np.nan, inplace=True)
  park['LONGITUDE'].replace('', np.nan, inplace=True)
  park = park.dropna()

  # ['LATITUDE'] 컬럼을 실수로 변경
  park['LATITUDE'] = park['LATITUDE'].astype(float)

  # ['LONGITUDE'] 컬럼을 실수로 변경
  park['LONGITUDE'] = park['LONGITUDE'].astype(float)

  return park

"""도서관 데이터 전처리 코드"""

def get_lib_data():
  start_point = 1
  end_point = 1000 # 최대 1000개까지만 호출 할 수 있음
  api_key = '57524f76506d656e3732636a52457a'

  url = f'http://openAPI.seoul.go.kr:8088/{api_key}/json/SeoulLibraryTimeInfo/{start_point}/{end_point}/'
  url2 = f'http://openAPI.seoul.go.kr:8088/{api_key}/json/SeoulLibraryTimeInfo/1001/2000/'

  data = requests.get(url).json()
  data2 = requests.get(url2).json()

  data = pd.DataFrame(data['SeoulLibraryTimeInfo']['row'])
  data2 = pd.DataFrame(data2['SeoulLibraryTimeInfo']['row'])

  lib = pd.concat([data, data2])
  # 컬럼명 변경
  lib.rename(columns={'XCNTS':'LATITUDE','YDNTS':"LONGITUDE"},inplace=True)

  # ['LATITUDE'] 컬럼을 실수로 변경
  lib['LATITUDE'] = lib['LATITUDE'].astype(float)

  # ['LONGITUDE'] 컬럼을 실수로 변경
  lib['LONGITUDE'] = lib['LONGITUDE'].astype(float)

  return lib

"""# 실행 코드"""


# 기상청 x,y좌표 데이터 불러오기
kor_loc = bring_weather_coor()

# 내 위치와 가장 가까운 위치 찾기
grid,nx,ny = find_nearest_grid(my_loc)

# 날짜와 시간, 좌표를 넣어서 날씨정보 획득
rain, temp, sky= get_weather_info(base_date,base_time,nx,ny)

# 박물관정보 불러오기
muse = get_muse_data()

# 공원정보 불러오기
park = get_park_data()

# 도서관 정보 불러오기
lib = get_lib_data()

# 날씨 정보 불러오기
weather_info = {"rain":rain, "sky":sky, "temp":temp}


def weather():
  return jsonify(weather_info)



"""사용자 위치정보"""

# 사용자로 부터 정보 받기
basedate = 240430
basetime = 1220
my_loc = (37.5660,126.9784)
if not (basedate and basetime and my_loc[0] and my_loc[1]):
    return jsonify({"error": "location or time information not provided"}), 400


@app.route("/api/detectTrash", methods=['POST'])

# 날씨정보를 넣으면 날씨를 알려주고 장소 추천하는 함수
# 강수가 맑음이 아니거나 기온이 30도 이상이면 도서관이나 미술관에 가세요
def where_to_go(rain,temp,sky):

  if (rain != '비가 오고 있지 않습니다.' or float(temp) >= 30) :
    liblat, liblong = find_nearest_place(my_loc,lib)
    muselat, muselong = find_nearest_place(my_loc,muse)
    return jsonify({"liblat":liblat,"liblong":liblong,"muselat":muselat,"muselong":muselong})
  else: 
    parklat, parklong = find_nearest_place(my_loc,park)
    return jsonify({"parklat":"parklong"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5004)