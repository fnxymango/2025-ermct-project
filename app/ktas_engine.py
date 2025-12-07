#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
119 EMS 전용 – Whisper large + SBAR v3 + KTAS 1/2/3 분류 엔진 (완전 정상작동 버전)
✔ CPR/AED 파싱 정상화
✔ SBAR 구조 정상화
✔ KTAS 로직 안정화
"""
import tempfile
from typing import Union, IO
import os
import re
from difflib import SequenceMatcher, get_close_matches
import whisper
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 0. 환경 설정 및 모델 준비
# ============================================================

# 캐시 경로 설정 (필요시 수정, 없으면 기본값 사용되므로 주석처리해도 됨)
os.environ["XDG_CACHE_HOME"] = r"C:\whisper_cache"
os.environ["WHISPER_CACHE_DIR"] = r"C:\whisper_cache"

# API KEY (사용자 관리)
# 실제 배포 시에는 os.environ.get("OPENAI_API_KEY") 권장
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ 경고: .env 파일에 OPENAI_API_KEY가 없습니다.")
client = OpenAI(api_key=OPENAI_API_KEY)

# 전역 변수로 모델 선언 (처음엔 비워둠)
whisper_model = None

def get_whisper_model():
    """
    서버 실행 시점이 아니라, 실제 필요할 때 모델을 로드하거나
    main.py의 startup 이벤트에서 호출하여 로드함 (지연 로딩)
    """
    global whisper_model
    if whisper_model is None:
        print("[INFO] Whisper large 모델 로딩 중...")
        # 모델 사이즈 변경 가능: "base", "small", "medium", "large-v3"
        whisper_model = whisper.load_model("large-v3")
        print("[INFO] Whisper large 로딩 완료.")
    return whisper_model


# ============================================================
# 1. Whisper STT
# ============================================================
def speech_to_text(audio_source: Union[str, IO]) -> str:
    """
    audio_source:
      - str : 파일 경로
      - IO  : FastAPI UploadFile.file 같은 파일 객체
    """
    # ★ 여기서 모델을 가져옵니다 (없으면 로딩)
    model = get_whisper_model()

    # 1) 경로(str)로 들어온 경우 그대로 사용
    if isinstance(audio_source, str):
        audio_path = audio_source
        delete_after = False
    else:
        # 2) 파일 객체인 경우 → 임시 파일에 저장 후 경로로 전달
        delete_after = True
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            # 파일 포인터 맨 앞으로 돌려놓기
            try:
                audio_source.seek(0)
            except Exception:
                pass
            chunk = audio_source.read()
            tmp.write(chunk)
            audio_path = tmp.name

    try:
        # GPU 사용 가능하다면 fp16=True, 아니면 False (자동 처리됨)
        result = model.transcribe(audio_path, language="ko")
        return result["text"].strip()
    finally:
        if delete_after:
            try:
                os.remove(audio_path)
            except Exception:
                pass



SEOUL_HOSPITAL_DB = [
    "연세대학교의과대학강남세브란스병원",
    "삼성서울병원",
    "강동경희대학교의대병원",
    "성심의료재단강동성심병원",
    "한국보훈복지의료공단중앙보훈병원",
    "이화여자대학교의과대학부속서울병원",
    "부민병원",
    "의료법인서울효천의료재단에이치플러스양지병원",
    "건국대학교병원",
    "혜민병원",
    "고려대학교의과대학부속구로병원",
    "구로성심병원",
    "희명병원",
    "인제대학교상계백병원",
    "노원을지대학교병원",
    "한국원자력의학원원자력병원",
    "의료법인한전의료재단한일병원",
    "경희대학교병원",
    "삼육서울병원",
    "서울특별시동부병원",
    "서울성심병원",
    "서울특별시보라매병원",
    "중앙대학교병원",
    "연세대학교의과대학세브란스병원",
    "의료법인동신의료재단동신병원",
    "학교법인가톨릭학원가톨릭대학교서울성모병원",
    "한양대학교병원",
    "학교법인고려중앙학원고려대학교의과대학부속병원(안암병원)",
    "재단법인아산사회복지재단서울아산병원",
    "경찰병원",
    "이화여자대학교의과대학부속목동병원",
    "홍익병원",
    "서울특별시서남병원",
    "가톨릭대학교여의도성모병원",
    "한림대학교강남성심병원",
    "성애의료재단성애병원",
    "명지성모병원",
    "대림성모병원",
    "순천향대학교부속서울병원",
    "가톨릭대학교은평성모병원",
    "의료법인청구성심병원",
    "서울대학교병원",
    "강북삼성병원",
    "서울적십자병원",
    "세란병원",
    "국립중앙의료원",
    "서울특별시서울의료원",
    "의료법인풍산의료재단동부제일병원",
    "녹색병원"
]

STATIC_MAP = {
    # ============================================
    # 1) 세브란스 계열
    # ============================================
    # 본원(신촌)
    "세브란스": "연세대학교의과대학세브란스병원",
    "세브": "연세대학교의과대학세브란스병원",
    "본세브": "연세대학교의과대학세브란스병원",
    "신촌세브": "연세대학교의과대학세브란스병원",

    # 강남 세브란스
    "강남세브란스": "연세대학교의과대학강남세브란스병원",
    "강남세브": "연세대학교의과대학강남세브란스병원",
    "강세브": "연세대학교의과대학강남세브란스병원",

    # ============================================
    # 2) 성모 계열
    # ============================================
    # 서울 성모
    "서울성모": "학교법인가톨릭학원가톨릭대학교서울성모병원",
    "반포성모": "학교법인가톨릭학원가톨릭대학교서울성모병원",

    # 여의도 성모
    "여의도성모": "가톨릭대학교여의도성모병원",
    "여성모": "가톨릭대학교여의도성모병원",

    # 은평 성모
    "은평성모": "가톨릭대학교은평성모병원",

    # ============================================
    # 3) 고려대 계열
    # ============================================
    # 고대 안암
    "안암": "학교법인고려중앙학원고려대학교의과대학부속병원(안암병원)",
    "고대안암": "학교법인고려중앙학원고려대학교의과대학부속병원(안암병원)",

    # 고대 구로
    "구로고대": "고려대학교의과대학부속구로병원",
    "고대구로": "고려대학교의과대학부속구로병원",
    "고대 구로": "고려대학교의과대학부속구로병원",

    # ============================================
    # 4) 이대(이화여대) 계열
    # ============================================
    # 이대 서울
    "이대서울": "이화여자대학교의과대학부속서울병원",
    "이대 서울": "이화여자대학교의과대학부속서울병원",
    "이화서울": "이화여자대학교의과대학부속서울병원",

    # 이대 목동
    "이대목동": "이화여자대학교의과대학부속목동병원",
    "목동이대": "이화여자대학교의과대학부속목동병원",
    "목동 병원": "이화여자대학교의과대학부속목동병원",

    # ============================================
    # 5) 삼성 계열
    # ============================================
    # 삼성서울병원
    "삼성서울": "삼성서울병원",
    "삼성 병원": "삼성서울병원",
    "smc": "삼성서울병원",

    # 강북삼성병원
    "강북삼성": "강북삼성병원",
    "강북 삼성": "강북삼성병원",

    # ============================================
    # 6) 성심 계열
    # ============================================
    # 강동성심
    "강동성심": "성심의료재단강동성심병원",
    # 구로성심
    "구로성심": "구로성심병원",
    # 청구성심
    "청구성심": "의료법인청구성심병원",

    
}

# ============================================================
# LLM 보정 함수 (여기에 추가)
# ============================================================


def llm_clean_text(raw_text: str) -> str:
    """
    OpenAI GPT-4.1-mini 기반 음성 인식 문장 보정
    - 숫자 보존
    - 병원명·의료용어 보정
    - 문장 구조 유지
    """

    prompt = f"""
당신은 119 구급대원의 응급실 전화보고 문장을 자연스럽고 정확하게 보정하는 LLM입니다.

[목표]
- STT 원문의 의미를 유지하며 한국어 문장만 자연스럽게 정리합니다.
- 숫자(나이·혈압·맥박·호흡수·SpO₂·체온 등)는 변경하지 않습니다.
- 병원명은 반드시 제공된 병원리스트 중 하나로 보정합니다.
- 존재하지 않는 병원명을 생성하지 않습니다.
- “평소 다니던/외래/팔로업” 언급 시 병원 보정 규칙을 적용합니다.

[병원명 보정 규칙]
STT원문에 기재된 병원명이 정식 병원명 리스트에 있다면 STT원문에 기재된 병원명 그대로 놔둡니다.
다음 단축/비표준 명칭은 → 오른쪽의 정식 병원명으로 변경합니다.
세브란스/세브/본세브/신촌세브 → 연세대학교의과대학세브란스병원
강남세브란스/강남세브/강세브 → 연세대학교의과대학강남세브란스병원
서울성모/반포성모 → 학교법인가톨릭학원가톨릭대학교서울성모병원
여의도성모/여성모/여의성모 → 가톨릭대학교여의도성모병원
은평성모 → 가톨릭대학교은평성모병원
안암/고대안암 → 학교법인고려중앙학원고려대학교의과대학부속병원(안암병원)
구로고대/고대구로 → 고려대학교의과대학부속구로병원
이대서울/이화서울 → 이화여자대학교의과대학부속서울병원
이대목동/목동이대/목동 병원 → 이화여자대학교의과대학부속목동병원
삼성서울/삼성 병원/smc → 삼성서울병원
강북삼성 → 강북삼성병원
강동성심 → 성심의료재단강동성심병원
36병원/36/삼육 → 삼육서울병원
구로성심 → 구로성심병원
청구성심 → 의료법인청구성심병원
강남성심 → 한림대학교강남성심병원
서울대/서울대병원 → 서울대학교병원
아산병원/서울아산 → 재단법인아산사회복지재단서울아산병원
보라매 → 서울특별시보라매병원
서남병원/서남 → 서울특별시서남병원

[정식 병원명 리스트]
(이 리스트는 원문 그대로 유지)  
연세대학교의과대학강남세브란스병원  
삼성서울병원  
강동경희대학교의대병원  
성심의료재단강동성심병원  
한국보훈복지의료공단중앙보훈병원  
이화여자대학교의과대학부속서울병원  
부민병원  
의료법인서울효천의료재단에이치플러스양지병원  
건국대학교병원  
혜민병원  
고려대학교의과대학부속구로병원  
구로성심병원  
희명병원  
인제대학교상계백병원  
노원을지대학교병원  
한국원자력의학원원자력병원  
의료법인한전의료재단한일병원  
경희대학교병원  
삼육서울병원  
서울특별시동부병원  
서울성심병원  
서울특별시보라매병원  
중앙대학교병원  
연세대학교의과대학세브란스병원  
의료법인동신의료재단동신병원  
학교법인가톨릭학원가톨릭대학교서울성모병원  
한양대학교병원  
학교법인고려중앙학원고려대학교의과대학부속병원(안암병원)  
재단법인아산사회복지재단서울아산병원  
경찰병원  
이화여자대학교의과대학부속목동병원  
홍익병원  
서울특별시서남병원  
가톨릭대학교여의도성모병원  
한림대학교강남성심병원  
성애의료재단성애병원  
명지성모병원  
대림성모병원  
순천향대학교부속서울병원  
가톨릭대학교은평성모병원  
의료법인청구성심병원  
서울대학교병원  
강북삼성병원  
서울적십자병원  
세란병원  
국립중앙의료원  
서울특별시서울의료원  
의료법인풍산의료재단동부제일병원  
녹색병원  

[주호소 표현 보정 규칙]
- 성별·나이 문장 뒤에서 “주호/주ㅎ” 형태가 등장하고 증상을 의미하면 → “주호소”로 수정.

[의학 용어 음성 오류 보정]
문맥상 이상한 한국어 단어가 등장하면 발음 기반으로 가장 가까운 영어 의학 약어로 수정합니다.
예:
해널비/애널비/엔알 → NRB  
나잘/내절/코줄 → Nasal prong  
브이텍/비텍 → V-tach 또는 V-fib  
노말/엔에스 → NS(Normal Saline)  

[AVPU 보정 규칙]
1) "의식은 A/V/P/U" 형태면 그대로 사용.  
2) 알파벳이 없고 한국어만 있을 경우 발음 기반 매핑:  
   - A: 에이/에/애/이  
   - V: 브이/비/브에/비에  
   - P: 피/피에/퍼/피해/프/프에  
   - U: 유/우/으  
입력 내용을 가장 유사한 그룹에 매핑하여 A·V·P·U 중 하나로 정규화합니다.  
최종 문장은 “의식은 X입니다”로 정리합니다.

[출력]
- 자연스러운 응급전화 보고 문장만 출력합니다.
- 설명을 추가하지 말고 교정된 문장만 출력합니다.

[입력 STT 원문]
{raw_text}


{raw_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("\n[LLM ERROR]", str(e))
        return raw_text



def token_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_match_hospital(raw_text, hospital_db):
    """
    [서울 전용] 병원명 매칭 로직
    1순위: 주요 대학병원 계열별 규칙 (지점명 우선 확인)
    2순위: STATIC_MAP (단순 매핑)
    3순위: Fuzzy Matching (유사도)
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    norm = text.replace(" ", "")

    # -----------------------------
    # 1) 주요 대학병원 계열별 규칙 (서울 소재만)
    # -----------------------------

    # [1] 세브란스 (연세대)
    # ★ "강남"이 있는지 먼저 보고 -> 없으면 신촌(본원)
    if ("세브란스" in norm) or ("세브" in norm):
        if "강남" in norm:
            return "연세대학교의과대학강남세브란스병원"
        return "연세대학교의과대학세브란스병원"

    # [2] 성모병원 (가톨릭대)
    # 여의도, 은평 먼저 체크 -> 나머지는 서울성모(반포)
    if "성모" in norm:
        if "여의도" in norm or "여성모" in norm:
            return "가톨릭대학교여의도성모병원"
        if "은평" in norm:
            return "가톨릭대학교은평성모병원"
        
        # 주의: "명지성모", "대림성모" 등은 대학병원이 아님. 
        # 이를 제외하고 "성모병원"이라고만 했을 때 서울성모로 연결
        if "명지" not in norm and "대림" not in norm:
            return "학교법인가톨릭학원가톨릭대학교서울성모병원"

    # [3] 고려대 (안암/구로)
    # 구로 먼저 체크 -> 나머지는 안암
    if "고대" in norm or "안암" in norm or ("구로" in norm and "고대" in norm):
        if "구로" in norm:
            return "고려대학교의과대학부속구로병원"
        return "학교법인고려중앙학원고려대학교의과대학부속병원(안암병원)"

    # [4] 이화여대 (서울/목동)
    # 목동 먼저 체크 -> 나머지는 서울(마곡)
    if ("이대" in norm) or ("이화" in norm):
        if "목동" in norm:
            return "이화여자대학교의과대학부속목동병원"
        return "이화여자대학교의과대학부속서울병원"

    # [5] 삼성서울 vs 강북삼성
    if "삼성" in norm:
        if "강북" in norm:
            return "강북삼성병원"
        # 그냥 "삼성병원" 하면 삼성서울병원
        return "삼성서울병원"
        
    # [6] 서울대
    if "서울대" in norm:
        # 보라매병원은 "보라매" 키워드로 STATIC_MAP에서 처리됨
        # 따라서 여기선 혜화동 본원만 처리
        if "보라매" not in norm:
            return "서울대학교병원"

    # [7] 아산병원
    if "아산" in norm:
        return "재단법인아산사회복지재단서울아산병원"
    
    # [8] 순천향 (서울)
    if "순천향" in norm:
        return "순천향대학교부속서울병원"
    
    # [9] 상계백병원 (서울엔 상계백만 남음)
    if "백병원" in norm:
        return "인제대학교상계백병원"

    # -----------------------------
    # 2) STATIC_MAP 매핑 (나머지 병원들)
    # -----------------------------
    # 키 길이가 긴 순서대로 정렬하여 매칭 (예: '강동성심'이 '성심'보다 먼저 걸리게)
    for key in sorted(STATIC_MAP.keys(), key=len, reverse=True):
        if key in norm:
            return STATIC_MAP[key]

    # -----------------------------
    # 3) Fuzzy Matching (이하 동일)
    # -----------------------------
    m = re.search(r'([가-힣A-Za-z0-9\s]*병원)', text)
    if m:
        core = m.group(1).strip()
    else:
        core = text

    cleaned = core.replace(" ", "").replace("병원", "")
    tokens = re.findall(r"[가-힣]+", cleaned)

    if not tokens:
        fallback = get_close_matches(core, hospital_db, n=1, cutoff=0.4)
        return fallback[0] if fallback else None

    best_hospital = None
    best_score = 0.0

    for hosp in hospital_db:
        hosp_clean = hosp.replace(" ", "")
        hosp_tokens = re.findall(r"[가-힣]+", hosp_clean)

        score = 0.0
        for t in tokens:
            for ht in hosp_tokens:
                sim = token_similarity(t, ht)
                if sim > score:
                    score = sim

        if score > best_score:
            best_score = score
            best_hospital = hosp

    if best_score >= 0.70:
        return best_hospital

    fallback = get_close_matches(core, hospital_db, n=1, cutoff=0.5)
    return fallback[0] if fallback else None



# ============================================================
# 2. SBAR 파서
# ============================================================

PATTERNS = {
    "dyspnea": [
        r"숨\s?차", r"숨이 차", r"숨이 가쁘", r"호흡곤란", r"숨 못 쉬",
        r"가쁜 호흡", r"헐떡", r"대화.*힘들", r"말.*못"
    ],
    "chest_pain": [r"가슴.*아파", r"흉통", r"가슴.*통증"],
    
    # ★ 사고기전 핵심 패턴
    "mechanism": [
        r"교통사고", r"사고", r"추돌", r"충돌",
        r"정면.*충돌", r"측면.*충돌", r"후방.*추돌",
        r"보행자.*사고", r"오토바이", r"자전거.*사고",
        r"에어백", r"전복", r"안전벨트.*안",
        r"낙상", r"추락",
        r"\d+.*미터.*떨어"   # 예: "3미터에서 떨어짐"
    ],

    "bleeding_severe": [r"대량.*출혈", r"피.*분출"],
    "anticoagulant": [r"항응고제", r"와파린", r"엘리퀴스"],
}

CHIEF_COMPLAINT_KO = {
    "chest_pain": "가슴 통증 (Chest pain)",
    "dyspnea": "호흡곤란 (Dyspnea)",
    "neuro": "신경학적 증상 (Stroke-like symptoms)",
    "abdominal": "복통/소화기 증상 (Abdominal pain / GI symptoms)",
    "bleeding": "출혈 (External / GI bleeding)",
    "altered": "의식 변화 (Altered mental status)",
    "trauma": "외상 (Trauma / burns / crush / amputation)",
    "obgyn": "산부인과 응급 (OB-GYN emergency)",
    "pediatric": "소아 응급 (Pediatric emergency)",
    "psychiatric": "정신과적 응급 (Psychiatric emergency)",
    None: "정보 없음"
}


def match_any(pattern_list, text):
    return any(re.search(p, text) for p in pattern_list)


# -------------------- A --------------------
def extract_avpu(text: str) -> str:
    t = text.replace(" ", "")

    # 1) "의식은 V입니다 / 의식은 V" 등 다 허용
    m = re.search(r"(의식은|의식상태는|의식수준은|AVPU는)\s*([AVPU])", text, re.IGNORECASE)
    if m:
        return m.group(2).upper()

    # 2) "의식 V입니다", "의식 V", "의식상태 V" 등
    m = re.search(r"(의식|의식상태|의식수준|AVPU)[은는]?\s*([AVPU])", text, re.IGNORECASE)
    if m:
        return m.group(2).upper()

    # 3) 확장 패턴 – V뒤에 글자가 붙어도 잡기
    m = re.search(r"의식[은는]?\s*([AVPU])[^A-Za-z]?", text)
    if m:
        return m.group(1).upper()

    # 4) 풀 단어 기반
    if re.search(r"의식.*(있|명료)", text, re.IGNORECASE):
        return "A"
    if re.search(r"의식.*(verbal|벌벌)", text, re.IGNORECASE):
        return "V"
    if re.search(r"의식.*(pain|페인|통증반응)", text, re.IGNORECASE):
        return "P"
    if re.search(r"의식.*(unresponsive|없|무의식)", text, re.IGNORECASE):
        return "U"

    return "A"



def parse_assessment_v3(text: str) -> dict:
    A = {
        "AVPU": "A",
        "BP_sys": None, "BP_dia": None,
        "HR": None, "RR": None, "SpO2": None,
        "BT": None,
        "pain_nrs": None,
        "dyspnea": False, "dyspnea_severity": None,
        "bleeding": False, "bleeding_severity": None
    }

    t = text

    # AVPU
    A["AVPU"] = extract_avpu(t)

    # ★ BP : 혈압
    bp = re.search(
        r'(혈압|bp|비피|b/p)\D{0,10}?(\d{2,3})\D{0,5}[/대\s]?\D{0,5}(\d{2,3})',
        t, re.IGNORECASE)
    if bp:
        A["BP_sys"] = int(bp.group(2))
        A["BP_dia"] = int(bp.group(3))

    # ====== 혈압 자동 보정 로직 ======
    sbp = A["BP_sys"]
    dbp = A["BP_dia"]

    if sbp is not None and dbp is not None:
        # 1) SBP 앞자리 잘림 보정 (예: 18/78 → 118/78)
        if sbp < 60 and dbp >= 60:
            A["BP_sys"] = sbp + 100

        # 2) SBP 한 자리 잘림 (예: 6/70 → 106/70)
        if sbp < 10 and dbp > 50:
            A["BP_sys"] = sbp + 100

        # 3) DBP가 비정상적으로 큰 경우 → SpO2나 HR이 오염된 경우
        #    예: 118/104 (원래는 118/78)
        if dbp > 120 or dbp > sbp:
            # DBP는 보통 40~110 사이
            # sbp-dbp 차이가 이상하면 DBP를 바로 앞 숫자로 재추출
            candidates = re.findall(r'\d{2,3}', text)
        
            # candidates 중 SBP 다음에 나오는 가장 현실적인 DBP 선택
            possible_dbp = [int(x) for x in candidates if 40 <= int(x) <= 120 and int(x) < A["BP_sys"]]
            if possible_dbp:
                A["BP_dia"] = possible_dbp[-1]  # 마지막 현실적인 DBP 선택

        # 4) SBP < DBP → 반드시 오류
        if A["BP_sys"] <= A["BP_dia"]:
            # SBP에 +100 보정
            A["BP_sys"] += 100



    # ★ HR : 맥박
    hr = re.search(
        r'(맥박|심박수)\D*?(\d{2,3})',
        t, re.IGNORECASE)
    if hr:
        A["HR"] = int(hr.group(2))

    # ★ RR : 호흡수
    rr = re.search(
        r'(호흡수|호흡)\D*?(\d{1,2})',
        t, re.IGNORECASE)
    if rr:
        A["RR"] = int(rr.group(2))

    # ★ SpO2
    spo = re.search(r'(SpO2|산소\s*포화[도자]?|산소)\D*(\d+)', t)
    if spo:
        A["SpO2"] = int(spo.group(2))

    # ★ BT : 체온
    bt = re.search(
        r'(체온|온도)\D*?(\d{1,2}\.?\d{0,2})\D*(도|℃)?',
        t
    )
    if bt:
        A["BT"] = float(bt.group(2))

    # ★ NRS : 통증
    pain = re.search(
        r'(통증|nrs|엔알에스)\D*?(\d{1,2})(?:[\.점\s]*?(\d))?',
        t,
        re.IGNORECASE
    )
    if pain:
        base = int(pain.group(2))
        decimal = pain.group(3)
        if decimal:
            A["pain_nrs"] = float(f"{base}.{decimal}")
        else:
            A["pain_nrs"] = float(base)

    # dyspnea/bleeding 부분
    if match_any(PATTERNS["dyspnea"], t):
        A["dyspnea"] = True
        if "심한" in t or "말" in t:
            A["dyspnea_severity"] = "severe"
        else:
            A["dyspnea_severity"] = "moderate"

    if match_any(PATTERNS["bleeding_severe"], t):
        A["bleeding"] = True
        A["bleeding_severity"] = "severe"
    elif "출혈" in t:
        A["bleeding"] = True
        A["bleeding_severity"] = "moderate"

    return A


# -------------------- B --------------------
def parse_background_v3(text: str) -> dict:
    B = {
        "HTN": False, "DM": False, "HF": False,
        "anticoagulant": False,
        "high_risk": False,
        "pregnant": False,
        "recent_surgery": False,
    }

    if "고혈압" in text:
        B["HTN"] = True
    if "당뇨" in text:
        B["DM"] = True
    if "심부전" in text:
        B["HF"] = True

    # ★ 심근경색/협심증도 high_risk에 포함
    cad = False
    if "심근경색" in text or "협심증" in text or "스텐트" in text:
        cad = True

    if match_any(PATTERNS["anticoagulant"], text):
        B["anticoagulant"] = True

    if "임신" in text:
        B["pregnant"] = True
    if "수술" in text:
        B["recent_surgery"] = True

    if B["HF"] or B["anticoagulant"] or cad:
        B["high_risk"] = True

    return B


# -------------------- S --------------------
def parse_situation_v3(text: str) -> dict:
    S = {
        "age": None,
        "gender": None,
        "chief_complaint": None,
        "mechanism": False,
        "followup_raw": None,   # 병원 관련 구간(raw)
        "requirement": None,    # 이송/전원/처치 필요 사항
    }

    t = text

    # ------------------------- 성별 -------------------------
    if "남자" in t or "남성" in t:
        S["gender"] = "M"
    if "여자" in t or "여성" in t:
        S["gender"] = "F"

    # ------------------------- 나이 -------------------------
    age = re.search(r'(\d+)\s*세', t)
    if age:
        S["age"] = int(age.group(1))


    # ------------------------- 주호소(10개 체계) -------------------------
    cc_patterns = {
        "chest_pain": [
            r"가슴", r"흉통", r"심장.*아파", r"CP", r"chest", r"흉부.*통증", r"흉부통증"
        ],
        "dyspnea": [
            r"숨\s?차", r"호흡곤란", r"숨.*못 쉬", r"숨이 가쁘", r"호흡.*곤란", r"호흡곤란"
        ],
        "neuro": [
            r"편마비", r"마비", r"말.*어눌", r"발음.*이상",
            r"발작", r"경련", r"stroke", r"뇌졸중", r"신경"
        ],
        "abdominal": [
            r"복통", r"배.*아파", r"토", r"구토", r"토혈", r"혈변", r"멜레나",
            r"소화", r"GI", r"melena", r"hematemesis", r"복부.*통증", r"복부통증"
        ],
        "bleeding": [
            r"출혈", r"피.*난", r"코피", r"지혈.*안됨"
        ],
        "altered": [
            r"의식.*떨어", r"의식.*저하", r"쓰러짐", r"실신", r"syncope", r"의식.*변화", r"멘탈.*변화",
            r"의식.*소실"
        ],
        "trauma": [
            r"교통사고", r"사고", r"낙상", r"절단", r"골절", r"화상", r"골반",
            r"두부손상", r"추락"
        ],
        "obgyn": [
            r"임신", r"진통", r"출산", r"질출혈", r"산과", r"부인과", r"임산"
        ],
        "pediatric": [
            r"소아", r"아기", r"아이", r"열", r"고열", r"경련", r"탈수"
        ],
        "psychiatric": [
            r"자살", r"우울", r"폭력", r"환청", r"망상", r"정신", r"흥분",
            r"급성.*정신병"
        ],
    }

    # -------------------------
    # 🔥 ① 명시적 주호소 인식 (최우선)
    # -------------------------
    explicit_cc_patterns = [
        r"(주호소|주호|주된\s*증상|주된\s*호소|주\s*증상|주된\s*문제|가장\s*불편한\s*곳|가장\s*아픈\s*곳)[은는]?\s*([가-힣A-Za-z0-9\s]+)",
        r"([가-힣A-Za-z0-9\s]+?)(?:을|를)\s*호소",
        r"([가-힣A-Za-z0-9\s]+?)\s*소견"
    ]

    explicit_cc_raw = None

    for ep in explicit_cc_patterns:
        m = re.search(ep, t)
        if m:
            if m.lastindex and m.lastindex >= 2:
                explicit_cc_raw = m.group(2).strip()
            else:
                explicit_cc_raw = m.group(1).strip()
            break

    # 명시적 주호소 표현이 발견된 경우 → 매칭 시도
    cc_found = False
    if explicit_cc_raw:
        for cc, patterns in cc_patterns.items():
            if match_any(patterns, explicit_cc_raw):
                S["chief_complaint"] = cc
                cc_found = True
                break  # 매칭되면 루프 탈출 (함수 리턴 X)

    # 명시적 주호소가 없었을 때만 전체 텍스트 스캔
    if not cc_found:
        for cc, patterns in cc_patterns.items():
            if match_any(patterns, t):
                S["chief_complaint"] = cc
                break

    # ------------------------- 사고기전(Mechanism) -------------------------
    # ★ 위에서 리턴하지 않았으므로 이제 정상적으로 실행됨
    if match_any(PATTERNS["mechanism"], t):
        S["mechanism"] = True

    # -------------------------
    # 🔥 병원 단어 + 부정표현 → followup 강제 없음
    # -------------------------
    hospital_none_patterns = [
        r"병원.{0,20}(안\s*다니|안\s*다녀|안\s*다녔|다닌\s*곳\s*없|다니는\s*곳\s*없|따로\s*없|없[어요습니다]*)",
        r"(안\s*다니|안\s*다녀|안\s*다녔|다닌\s*곳\s*없|다니는\s*곳\s*없|따로\s*없|없[어요습니다]*)\s*병원",
        r"평소\s*다니는\s*병원\s*없",
        r"다니던\s*병원\s*없",
        r"병원\s*모르겠",
        r"병원\s*모름",
        r"병원\s*없",
    ]

    hospital_forced_none = False

    for pattern in hospital_none_patterns:
        if re.search(pattern, t, re.IGNORECASE):
            S["followup_raw"] = None
            hospital_forced_none = True
            break

    # 병원 없음이 명확히 감지되면 병원 파싱 전체 스킵
    if not hospital_forced_none:
        # -------------------------
        # 🔽 기존 병원 파싱 로직
        # -------------------------
        followup_raw = None

        hospital_patterns = [
            r"(?:기존|평소|다니던|주로|단골)\s*.*?([가-힣A-Za-z0-9\s]+병원)",
            r"([가-힣A-Za-z0-9\s]+병원)\s*다니",
            r"(?:병원)\s*([가-힣A-Za-z0-9\s]+병원)\s*다녔",
        ]

        for hp in hospital_patterns:
            m = re.search(hp, t)
            if m:
                followup_raw = m.group(1).strip()
                break

        # follow-up 병원 패턴
        followup_patterns = [
            r'([가-힣A-Za-z0-9\s]+병원).*?(팔로업|팔로우업|follow[-\s]?up|추적|추적관찰|외래|경과관찰)',
            r'(팔로업|팔로우업|follow[-\s]?up|추적|추적관찰|외래).*?([가-힣A-Za-z0-9\s]+병원)',
            r'([가-힣A-Za-z0-9\s]+병원).*(보던|다니던|다녔던|관리받는|진료받는)',
        ]

        if followup_raw is None:
            for fp in followup_patterns:
                m = re.search(fp, t)
                if m:
                    if m.lastindex and m.group(1) and "병원" in m.group(1):
                        followup_raw = m.group(1).strip()
                    elif m.lastindex and m.group(2):
                        followup_raw = m.group(2).strip()
                    break

        S["followup_raw"] = followup_raw


    # ------------------------- 이송/요청 사항 (문장 전체 추출) -------------------------
    requirement_keywords = [
        "요청", "필요", "전원", "이송", "요구",
        "집중치료", "관찰 필요", "처치 필요",
        "필요합니다", "필요할 것 같", "필요해 보"
    ]

    requirement = None
    sentences = re.split(r'[.!?]\s*', t)

    for sent in sentences:
        for kw in requirement_keywords:
            if kw in sent:
                requirement = sent.strip()
                break
        if requirement:
            break

    S["requirement"] = requirement

    return S

# -------------------- R --------------------
def parse_response_v3(text: str) -> dict:
    R = {
        "oxygen_lpm": None,
        "fluids": False,
        "CPR": 0,           # 항상 0 또는 1로 저장
        "AED_shocks": 0,
        "response": None
    }

    t = text.lower()

    # 산소
    o2 = re.search(r'산소\s*(\d+)', t)
    if o2:
        R["oxygen_lpm"] = int(o2.group(1))

    # 수액
    if "수액" in t or "정맥로" in t:
        R["fluids"] = True

    # CPR 처리 -------------------------
    if "cpr" in t:
        neg = [
            "하지 않", "안 했", "안함", "안 하", "없었",
            "미시행", "시행하지 않", "시행 안", "필요 없어",
            "안 했습니다", "안했습니다", "안했습니다"
        ]
        if any(n in t for n in neg):
            R["CPR"] = 0
        else:
            R["CPR"] = 1
    # -----------------------------------

    # AED shock
    m = re.search(r'(aed|제세동).{0,5}(\d+)\s*회', t)
    if m:
        R["AED_shocks"] = int(m.group(2))

    # 반응
    if "호전" in t or "나아졌" in t:
        R["response"] = "improved"
    elif "악화" in t:
        R["response"] = "worsened"
    elif "변화 없" in t:
        R["response"] = "no_change"

    return R


# -------------------- SBAR 통합 --------------------
def parse_sbar_v3(text: str) -> dict:
    return {
        "S": parse_situation_v3(text),
        "B": parse_background_v3(text),
        "A": parse_assessment_v3(text),
        "R": parse_response_v3(text),
    }


# ============================================================
# 3. KTAS 1/2/3 결정
# ============================================================
def decide_ktas_1to3(sbar: dict, raw_text: str):
    S, B, A, R = sbar["S"], sbar["B"], sbar["A"], sbar["R"]

    level = 3
    reasons = []

    sbp = A["BP_sys"]
    hr = A["HR"]
    bt = A["BT"]
    rr = A["RR"]
    spo2 = A["SpO2"]

    # =====================================================
    # 0) CPR / AED → 무조건 KTAS 1
    # =====================================================
    if R["CPR"] == 1 or R["AED_shocks"] >= 1:
        return 1, "CPR 또는 AED 충격 시행 → KTAS 1"

    # =====================================================
    # 1) AVPU (의식)
    # =====================================================
    if A["AVPU"] == "U":
        return 1, "의식 없음(AVPU=U) → KTAS 1"

    if A["AVPU"] in ["V", "P"]:
        level = min(level, 2)
        reasons.append(f"의식 저하(AVPU={A['AVPU']}) → KTAS 2")

    # =====================================================
    # 쇼크(Early Shock) 판정 – 기준 2개 이상이면 KTAS 1
    # =====================================================
    shock_count = 0

    # SBP 80~90
    if sbp is not None and 80 <= sbp < 90:
        shock_count += 1

    # HR > 130
    if hr is not None and hr > 130:
        shock_count += 1

    # RR > 30
    if rr is not None and rr > 30:
        shock_count += 1

    # SpO2 85~90
    if spo2 is not None and 85 <= spo2 < 90:
        shock_count += 1

    # 피부 창백/발한
    if "창백" in raw_text or "창백해" in raw_text or "식은땀" in raw_text or "발한" in raw_text:
        shock_count += 1

    # CRT > 3초
    if "모세혈관" in raw_text or "capillary" in raw_text or "CRT" in raw_text:
        m = re.search(r'(crt|모세혈관)\D*(\d+)', raw_text, re.IGNORECASE)
        if m and int(m.group(2)) >= 3:
            shock_count += 1

    if shock_count >= 2:
        return 1, f"쇼크 의심 기준 {shock_count}개 충족 → KTAS 1"

    # =====================================================
    # 2) 혈역학 상태 (혈압 + 쇼크 징후 + 맥박)
    # =====================================================
    # --- 1단계 ---
    if sbp is not None and sbp < 80:
        return 1, f"심한 저혈압(SBP={sbp}) → KTAS 1"

    shock_keywords = ["쇼크", "shock", "혈압이 안 잡혀", "순환 안 됨", "말초 냉감"]
    if any(k in raw_text for k in shock_keywords):
        return 1, "쇼크 또는 순환붕괴 의심 표현 → KTAS 1"

    # --- 2단계 ---
    if sbp is not None and 80 <= sbp <= 100:
        level = min(level, 2)
        reasons.append(f"저혈압(SBP={sbp}) → KTAS 2")

    if hr is not None and (hr >= 130 or hr <= 40):
        level = min(level, 2)
        reasons.append(f"심한 맥박 이상(HR={hr}) → KTAS 2")

    # =====================================================
    # 3) 호흡 상태 (SpO2)
    # =====================================================
    if spo2 is not None:
        if spo2 < 90:
            return 1, f"심한 저산소증(SpO₂={spo2}) → KTAS 1"
        elif 90 <= spo2 < 92:
            level = min(level, 2)
            reasons.append(f"경도 저산소증(SpO₂={spo2}) → KTAS 2")

    # =====================================================
    # 4) 체온 / SIRS
    # =====================================================
    sirs_count = 0

    if bt is not None and (bt >= 38 or bt <= 36):
        sirs_count += 1
    if hr is not None and hr > 90:
        sirs_count += 1
    if rr is not None and rr > 20:
        sirs_count += 1

    if sirs_count >= 3:
        level = min(level, 2)
        reasons.append(f"SIRS 기준 {sirs_count}개 충족 → KTAS 2")
    elif sirs_count == 2:
        level = min(level, 3)
        reasons.append(f"SIRS 기준 2개 충족 → KTAS 3")

    # =====================================================
    # 5) 출혈성 질환 (키워드 기반)
    # =====================================================
    bleeding_text = raw_text

    severe_bleed_keywords = [
        "대량", "분출", "과다출혈", "심부", "동맥",
    ]
    mild_bleed_keywords = ["코피", "단순", "열상", "월경", "관절"]

    if any(k in bleeding_text for k in severe_bleed_keywords):
        level = min(level, 2)
        reasons.append("중등도~중증 출혈 소견 → KTAS 2")
    elif any(k in bleeding_text for k in mild_bleed_keywords):
        reasons.append("경미 출혈 → KTAS 3")

    # =====================================================
    # 6) 사고기전 (mechanism)
    # =====================================================
    if S["mechanism"]:
        level = min(level, 2)
        reasons.append("고위험 사고기전(Mechanism of Injury) → KTAS 2")

    # =====================================================
    # 7) 가슴통증 + 고위험군
    # =====================================================
    is_chest_pain = (
        S["chief_complaint"] == "chest_pain"
        or match_any(PATTERNS["chest_pain"], raw_text)
    )

    if is_chest_pain:
        chest_risk_factors = []

        # ① 압박감/조임 (통증 양상)
        pressure_keywords = ["조이", "조여", "짓누르", "압박", "쥐어짜",  "무거운"]
        if any(k in raw_text for k in pressure_keywords):
            chest_risk_factors.append("압박감/조임")

        # ② 식은땀 동반
        if "식은땀" in raw_text or "발한" in raw_text or "식은 땀" in raw_text:
            chest_risk_factors.append("식은땀 동반")

        # ③ 통증 강도 (NRS ≥ 7)
        if A["pain_nrs"] is not None and A["pain_nrs"] >= 7:
            chest_risk_factors.append(f"심한 통증(NRS={A['pain_nrs']})")

        # ④ 심근허혈 의심 소견 (키워드 매칭)
        #    (심전도, ST분절, 심근경색/허혈 언급 등)
        ischemia_keywords = ["심근허혈", "허혈", "ST", "심근경색", "협심증", "관상동맥"]
        if any(k in raw_text for k in ischemia_keywords):
            chest_risk_factors.append("심근허혈 의심 소견")

        # 위 조건 중 하나라도 해당하면 KTAS 2
        if chest_risk_factors:
            level = min(level, 2)
            reasons.append(f"전형적 흉통/고위험 소견({', '.join(chest_risk_factors)}) → KTAS 2")

    # =====================================================
    # 8) 기본 결과
    # =====================================================
    if not reasons:
        reasons.append("특이 위험 소견 없음 → KTAS 3")

    return level, "\n".join(reasons)

def build_stage2_payload(ktas_result: dict) -> dict:
    """
    Step2(병원 필터링 엔진)에 넘길 데이터 스키마 생성
    """
    return {
        "ktas_level": ktas_result["ktas"],                # 1 / 2 / 3
        "chief_complaint": ktas_result["chief_complaint"], # dyspnea / chest_pain ...
        "hospital_followup": ktas_result["followup_hospital"]  # 정식 병원명 or None
    }


def ktas_from_audio(audio_source: Union[str, IO]) -> dict:
    print(f"[INFO] 음성 인식 중... ({audio_source})")
    text = speech_to_text(audio_source)

    print("\n[STT 결과]")
    print(text)

    # ① LLM 보정
    clean_text = llm_clean_text(text)

    print("\n[LLM 보정 후 문장]")
    print(clean_text)

    # ② SBAR 파싱
    sbar = parse_sbar_v3(clean_text)

    print("\n[SBAR 결과]")
    print(sbar)

    # KTAS 판단
    level, reason = decide_ktas_1to3(sbar, clean_text)

    print("\n===== 최종 KTAS =====")
    print(f"KTAS = {level}")
    print(reason)

    ko_cc = CHIEF_COMPLAINT_KO.get(sbar["S"].get("chief_complaint"))
    raw_hospital = sbar["S"].get("followup_raw")
    final_hospital = best_match_hospital(raw_hospital, SEOUL_HOSPITAL_DB)

    print("\n===== 추가 정보 =====")
    print(f"주호소 : {ko_cc}")
    print(f"원내/기존 다니던 병원 : {final_hospital or '정보 없음'}")

    requirement = sbar["S"].get("requirement") or "None"
    print(f"요구사항 : {requirement}")

    return {
        "text": clean_text,
        "sbar": sbar,
        "ktas": level,
        "reason": reason,
        "chief_complaint": sbar["S"]["chief_complaint"],
        "followup_hospital_raw": raw_hospital,
        "followup_hospital": final_hospital,
    }




# ============================================================
# 5. 실행 예시
# ============================================================
#if __name__ == "__main__":
#    audio_path = r"C:\voice_ktas\test4.m4a"
#   result = ktas_from_audio(audio_path)

#    stage2_payload = build_stage2_payload(result)

#    print("\n===== [Stage2로 넘길 최종 Payload] =====")
#    print(stage2_payload)
    

