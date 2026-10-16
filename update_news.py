import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path

import feedparser
from google import genai

# 신뢰도 높은 국내 주요 경제 뉴스 RSS 피드 목록 (1차: 매일경제 경제, 2차 Fallback: 연합뉴스TV 경제)
RSS_FEED_URLS = [
    "https://www.mk.co.kr/rss/30100041/",
    "https://www.yonhapnewstv.co.kr/category/news/economy/feed/",
]

# data.json 출력 규격을 강제하기 위한 JSON Schema 정의
REPORT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": "리포트 발행 기준일 (YYYY-MM-DD 형식)",
        },
        "key_takeaways": {
            "type": "array",
            "description": "거시 경제 및 시장 핵심 요약 3건",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["order", "title", "summary"],
            },
        },
        "market_indicators": {
            "type": "array",
            "description": "주요 시장 지표 4건 (KOSPI, KOSDAQ, USD/KRW, WTI)",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "current_value": {"type": "string"},
                    "change": {"type": "string"},
                    "change_rate": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["name", "current_value", "change", "change_rate", "unit"],
            },
        },
        "featured_news": {
            "type": "array",
            "description": "오늘의 주요 경제 뉴스 5대 심층 분석 카드 (5건 고정)",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "category": {"type": "string"},
                    "source": {"type": "string"},
                    "observed_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "transmission_path": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "invalidating_conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "action_tip": {"type": "string"},
                },
                "required": [
                    "id",
                    "title",
                    "category",
                    "source",
                    "observed_facts",
                    "transmission_path",
                    "invalidating_conditions",
                    "action_tip",
                ],
            },
        },
    },
    "required": ["date", "key_takeaways", "market_indicators", "featured_news"],
}


def clean_html_text(raw_html: str) -> str:
    """HTML 태그 및 불필요한 공백을 제거하여 순수 텍스트로 정제합니다."""
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_rss_articles(max_articles: int = 15) -> list[dict]:
    """경제 뉴스 RSS 피드에서 최신 기사 10~15건을 수집합니다."""
    collected: list[dict] = []
    seen_titles: set[str] = set()

    for url in RSS_FEED_URLS:
        try:
            feed = feedparser.parse(url)
            feed_title = clean_html_text(feed.feed.get("title", "경제 뉴스 RSS"))
            for entry in feed.entries:
                title = clean_html_text(entry.get("title", ""))
                if not title or title in seen_titles:
                    continue

                summary = clean_html_text(
                    entry.get("summary", "") or entry.get("description", "")
                )
                link = entry.get("link", "").strip()
                author = clean_html_text(entry.get("author", "")) or feed_title
                published = entry.get("published", "")

                seen_titles.add(title)
                collected.append(
                    {
                        "title": title,
                        "link": link,
                        "summary": summary[:600],
                        "source": author,
                        "published": published,
                    }
                )

                if len(collected) >= max_articles:
                    return collected
        except Exception as exc:
            print(f"[WARN] RSS 피드 수집 중 오류 발생 ({url}): {exc}", file=sys.stderr)

    return collected


def load_existing_indicators(data_path: Path) -> list[dict]:
    """기존 data.json에서 시장 지표 기준값을 불러옵니다 (기사 내 수치 미언급 시 구조 유지용)."""
    default_indicators = [
        {"name": "KOSPI", "current_value": "6,082.90", "change": "+49.55", "change_rate": "+0.82%", "unit": "pt"},
        {"name": "KOSDAQ", "current_value": "833.15", "change": "-3.94", "change_rate": "-0.47%", "unit": "pt"},
        {"name": "USD/KRW", "current_value": "1,336.80", "change": "-6.20", "change_rate": "-0.46%", "unit": "KRW"},
        {"name": "WTI", "current_value": "78.20", "change": "-0.95", "change_rate": "-1.20%", "unit": "USD/bbl"},
    ]
    if not data_path.exists():
        return default_indicators
    try:
        with data_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
            indicators = existing.get("market_indicators")
            if isinstance(indicators, list) and len(indicators) > 0:
                return indicators
    except Exception:
        pass
    return default_indicators


def build_prompt(today_str: str, articles: list[dict], baseline_indicators: list[dict]) -> str:
    """Gemini 모델에 전달할 프롬프트를 구성합니다."""
    articles_text = "\n\n".join(
        [
            f"[{idx + 1}] 제목: {art['title']}\n"
            f"- 출처/기자: {art['source']}\n"
            f"- 발행시각: {art['published']}\n"
            f"- 링크: {art['link']}\n"
            f"- 본문/요약: {art['summary']}"
            for idx, art in enumerate(articles)
        ]
    )

    baseline_json_str = json.dumps(baseline_indicators, ensure_ascii=False, indent=2)

    return f"""당신은 기관 투자자용 데일리 매크로 & 마켓 브리프(DAILY MACRO & MARKET BRIEF) 수석 에디터이자 금융 데이터 애널리스트입니다.
아래 수집된 최신 경제 뉴스 RSS 기사 목록을 분석하여 정해진 JSON 스키마 규격에 맞춰 리포트 데이터를 생성하세요.

[엄격 준수 원칙]
1. 주관적 주가 예측 금지: 막연한 낙관/비관이나 근거 없는 주가 방향성 예단을 절대 배제하세요.
2. 팩트 중심 서술 (observed_facts): 기사 본문에 명시된 수치, 공시, 정책 발표, 수급 데이터 등 검증 가능한 팩트만 개조식 불릿 문장으로 작성하세요.
3. 파급 경로 구체화 (transmission_path): 해당 팩트가 거시경제, 금리/환율, 혹은 산업 밸류체인과 기업 실적에 전이되는 논리적 메커니즘을 서술하세요.
4. 구체적 기각 조건 명시 (invalidating_conditions): "이 분석이 틀렸다면(Falsification criteria)" 항목에는 거시 지표 기준치 이탈, 실적 가이던스 하회, 정책/계약 취소 등 관측 및 검증 가능한 구체적 기각 조건을 2개씩 명시하세요.
5. 실전 체크 팁 (action_tip): 투자자 및 실무자가 당일 장세나 후속 공시/지표에서 확인해야 할 핵심 관전 포인트를 1문장으로 명확히 제시하세요.
6. 출력 규격 엄수:
   - date: "{today_str}"
   - key_takeaways: order 1, 2, 3 순서로 거시 경제 및 핵심 섹터 요약 3건 고정
   - market_indicators: KOSPI, KOSDAQ, USD/KRW, WTI 4개 항목 유지. 기사 본문에 최신 종가/현재가 및 등락 수치가 명시되어 있다면 이를 반영하고, 언급이 없는 지표는 아래 기존 기준값 구조와 수치를 유지하세요. (변동폭과 변동률에는 반드시 '+' 또는 '-' 부호 포함)
   - featured_news: 가장 중요도가 높은 대표 뉴스 5건(id: 1 ~ 5)을 선정하여 작성하세요.

[기존 시장 지표 기준값 (기사 내 수치 미언급 시 유지)]
{baseline_json_str}

[수집된 최신 경제 뉴스 기사 목록 ({len(articles)}건)]
{articles_text}
"""


def generate_daily_brief() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] 환경변수 GEMINI_API_KEY가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)

    kst = timezone(timedelta(hours=9))
    today_str = datetime.now(kst).strftime("%Y-%m-%d")
    data_path = Path(__file__).resolve().parent / "data.json"

    print(f"[INFO] 1/3 경제 뉴스 RSS 수집 시작 (기준일: {today_str})...")
    articles = fetch_rss_articles(max_articles=15)
    if not articles:
        print("[ERROR] RSS 피드에서 기사를 수집하지 못했습니다.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] 수집 완료: 총 {len(articles)}건의 기사를 가져왔습니다.")

    baseline_indicators = load_existing_indicators(data_path)
    prompt = build_prompt(today_str, articles, baseline_indicators)

    print("[INFO] 2/3 Gemini API 호출 및 JSON 정제 진행 중 (model: gemini-3.8-flash)...")
    client = genai.Client(api_key=api_key)

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
        system_instruction=(
            "주관적 주가 예측을 배제하고 관측된 팩트와 명확한 기각 조건(Falsification criteria)에 기반해 "
            "지정된 JSON 스키마 규격의 순수 JSON 객체만 출력하세요."
        ),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": REPORT_JSON_SCHEMA,
        },
    )

    raw_output = (interaction.output_text or "").strip()
    if not raw_output:
        print("[ERROR] Gemini API로부터 빈 응답을 받았습니다.", file=sys.stderr)
        sys.exit(1)

    parsed_json = json.loads(raw_output)

    print(f"[INFO] 3/3 정제된 데이터를 {data_path.name} 파일에 저장합니다...")
    with data_path.open("w", encoding="utf-8") as f:
        json.dump(parsed_json, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("[SUCCESS] data.json 업데이트가 성공적으로 완료되었습니다.")


if __name__ == "__main__":
    generate_daily_brief()
