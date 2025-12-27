import asyncio
from mcp import StdioServerParameters, stdio_client, ClientSession

# 1단계: 서버 매개변수 설정
# MCP 서버에 연결하기 위한 매개변수 정의
import os

# 현재 파일의 위치를 기준으로 절대 경로 생성
current_dir = os.path.dirname(os.path.abspath(__file__))
server_script = os.path.join(current_dir, "weather_server.py")
venv_python = os.path.join(os.path.dirname(current_dir), ".venv", "bin", "python")

server_params = StdioServerParameters(
    command=venv_python,           # 가상환경의 파이썬 실행 파일
    args=[server_script],          # 날씨 데이터 서버 스크립트 절대 경로
    env=None,                      # 선택적 환경 변수
)

# 2단계: 샘플링 콜백 정의
# 서버에서 들어오는 날씨 데이터를 처리하는 콜백 함수
async def handle_sampling_message(message):
    print(f"Received weather data: {message}")

# 메인 비동기 함수 정의
async def fetch_weather_data():
    # 3단계: 서버 연결 수립
    # stdio_client를 사용하여 MCP 서버와 연결
    async with stdio_client(server_params) as (read, write):
        # ClientSession으로 서버와의 통신 세션 생성
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            
            # 4단계: 세션 초기화
            # 서버와의 구성 및 인증 설정
            await session.initialize()
            
            # 5단계: 사용 가능한 프롬프트 가져오기
            # 서울의 날씨 프롬프트 요청
            prompt = await session.get_prompt(
                "weather-prompt", arguments={"city": "Seoul"}
            )
            
            # 6단계: 사용 가능한 리소스 및 도구 목록 확인
            # 서버에서 제공하는 리소스(파일, API 키 등) 목록
            resources = await session.list_resources()
            # 서버에서 제공하는 도구(함수, 서비스) 목록
            tools = await session.list_tools()
            
            # 7단계: 도구 호출
            # weather-tool을 호출하여 서울의 섭씨 날씨 데이터 요청
            weather_data = await session.call_tool(
                "weather-tool", 
                arguments={"city": "Seoul", "unit": "Celsius"}
            )
            
            # 선택 사항: 서버에서 리소스 파일 읽기
            try:
                result = await session.read_resource(
                    "file://weather_reports/seoul_report.pdf"
                )
                # 다운로드한 콘텐츠 미리보기 (result.contents는 리스트임)
                if result.contents:
                    content = result.contents[0].text
                    print(f"Downloaded content preview: {content[:100]}...")
            except Exception as e:
                print(f"리소스 읽기 실패: {e}")
            
            # 8단계: 결과 표시
            # 도구 호출로 받은 날씨 데이터 출력
            print(f"Weather data for Seoul: {weather_data}")

# 9단계: 코드 실행
# 이벤트 루프를 사용하여 비동기 함수 실행
if __name__ == "__main__":
    asyncio.run(fetch_weather_data())