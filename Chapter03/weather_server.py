# weather_server.py
import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Prompt,
    PromptMessage,
    GetPromptResult,
    Resource,
)

# MCP 서버 인스턴스 생성
app = Server("weather-server")

# 간단한 날씨 데이터베이스 (실제로는 API를 호출하거나 데이터베이스를 사용)
WEATHER_DATA = {
    "Lisbon": {
        "Celsius": {"temperature": 22, "condition": "Sunny", "humidity": 65},
        "Fahrenheit": {"temperature": 72, "condition": "Sunny", "humidity": 65},
    },
    "Seoul": {
        "Celsius": {"temperature": 15, "condition": "Cloudy", "humidity": 70},
        "Fahrenheit": {"temperature": 59, "condition": "Cloudy", "humidity": 70},
    },
    "New York": {
        "Celsius": {"temperature": 18, "condition": "Rainy", "humidity": 80},
        "Fahrenheit": {"temperature": 64, "condition": "Rainy", "humidity": 80},
    },
}

# 프롬프트 목록 제공
@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """사용 가능한 프롬프트 목록 반환"""
    return [
        Prompt(
            name="weather-prompt",
            description="도시의 날씨 정보를 가져오는 프롬프트",
            arguments=[
                {"name": "city", "description": "날씨를 조회할 도시", "required": True}
            ],
        )
    ]

# 특정 프롬프트 가져오기
@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    """요청된 프롬프트 반환"""
    if name != "weather-prompt":
        raise ValueError(f"알 수 없는 프롬프트: {name}")
    
    city = arguments.get("city", "Unknown")
    
    return GetPromptResult(
        description=f"{city}의 날씨 정보",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"{city}의 현재 날씨를 알려주세요."
                ),
            )
        ],
    )

# 도구 목록 제공
@app.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록 반환"""
    return [
        Tool(
            name="weather-tool",
            description="지정된 도시의 날씨 데이터를 가져옵니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "날씨를 조회할 도시",
                    },
                    "unit": {
                        "type": "string",
                        "description": "온도 단위 (Celsius 또는 Fahrenheit)",
                        "enum": ["Celsius", "Fahrenheit"],
                    },
                },
                "required": ["city"],
            },
        )
    ]

# 도구 호출 처리
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """도구 실행 및 결과 반환"""
    if name != "weather-tool":
        raise ValueError(f"알 수 없는 도구: {name}")
    
    city = arguments.get("city", "Unknown")
    unit = arguments.get("unit", "Celsius")
    
    # 날씨 데이터 조회
    if city in WEATHER_DATA and unit in WEATHER_DATA[city]:
        weather = WEATHER_DATA[city][unit]
        result = {
            "city": city,
            "temperature": weather["temperature"],
            "unit": unit,
            "condition": weather["condition"],
            "humidity": weather["humidity"],
        }
        result_text = (
            f"도시: {city}\n"
            f"온도: {weather['temperature']}°{unit[0]}\n"
            f"날씨: {weather['condition']}\n"
            f"습도: {weather['humidity']}%"
        )
    else:
        result_text = f"{city}의 {unit} 단위 날씨 데이터를 찾을 수 없습니다."
    
    return [
        TextContent(
            type="text",
            text=result_text,
        )
    ]

# 리소스 목록 제공
@app.list_resources()
async def list_resources() -> list[Resource]:
    """사용 가능한 리소스 목록 반환"""
    return [
        Resource(
            uri="file://weather_reports/seoul_report.pdf",
            name="Seoul Weather Report",
            description="서울의 상세 날씨 보고서",
            mimeType="application/pdf",
        )
    ]

# 리소스 읽기
@app.read_resource()
async def read_resource(uri: str) -> str:
    """요청된 리소스의 내용 반환"""
    uri_str = str(uri)
    if uri_str == "file://weather_reports/seoul_report.pdf":
        # 실제로는 파일을 읽지만, 여기서는 시뮬레이션
        return "서울 날씨 보고서\n\n2026년 12월 서울의 평균 기온은 -2°C이며, 맑고 추운 날씨가 지속되고 있습니다. 습도는 45% 정도로 건조한 수준입니다..."
    else:
        raise ValueError(f"알 수 없는 리소스: {uri_str}")

# 서버 실행
async def main():
    """서버 메인 함수"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

if __name__ == "__main__":
    # 서버 실행
    asyncio.run(main())