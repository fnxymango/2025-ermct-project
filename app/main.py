# app/main.py
from fastapi import FastAPI, Query, Response, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Set, Optional, Tuple
from fastapi import UploadFile, File # UploadFile, File 추가
# 뒤에 ', get_whisper_model' 을 꼭 붙여야 합니다!
from app.ktas_engine import ktas_from_audio, build_stage2_payload, get_whisper_model

from app.state_assignments import pending_assignments

from .services.ermct_client import ErmctClient

from app.schemas import (
    HospitalRealtime,
    HospitalBasicInfo,
    SeriousDiseaseStatus,
    HospitalMessage,
    HospitalSummary,
    TriageRequest,
    RecommendedHospital,
    TraumaCenter,
    HospitalComplaintCoverage,
    RoutingCandidateHospital,
    HospitalProcedureBeds,
    BedReservationRequest,
    BedReleaseRequest,
    RoutingCase,
    KTASRoutingRequest,
    RoutingCandidateResponse,
    NearestRoutingRequest,
)
from app.triage_utils import (
    procedure_status_for_hospital,
    choose_primary_bed_type,
    get_effective_beds_for_groups,
)

from app.procedure_groups import (
    compute_procedure_availability,
    humanize_procedure_groups,
    PROCEDURE_GROUPS,
)

from app.complaint_mapping import (
    required_procedure_groups_for_complaint,
    complaints_supported_by_hospital,
    complaint_id_from_chief_complaint,
    COMPLAINT_LABELS,
)

# 3단계 import
from .distance_logic import calculate_all_distances_async, get_top3

SERIOUS_MKIOSK_KEYS = [f"MKioskTy{i}" for i in range(1, 28)]  # 1 ~ 27

# 서울 25개 구
SEOUL_SIGUNGU_LIST = [
    "강남구", "강동구", "강북구", "강서구",
    "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구",
    "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
]

def _get_all_seoul_summaries(sm_type: int = 1) -> List[HospitalSummary]:
    """
    서울특별시 전체 25개 구에 대해
    get_hospital_summaries_by_region()를 돌려서
    중복 없이 HospitalSummary 리스트를 만들어준다.
    """
    all_summaries: List[HospitalSummary] = []
    seen: Set[str] = set()

    for gu in SEOUL_SIGUNGU_LIST:
        region_sums = get_hospital_summaries_by_region(
            sido="서울특별시",
            sigungu=gu,
            sm_type=sm_type,
            num_rows=200,
        )
        for s in region_sums:
            if not s.id or s.id in seen:
                continue
            seen.add(s.id)
            all_summaries.append(s)

    return all_summaries


def _resolve_home_hpid_from_followup(
    summaries: List[HospitalSummary],
    hospital_followup: Optional[str],
) -> Optional[str]:
    """
    KTAS 모듈에서 넘어온 hospital_followup(병원명 or HPID)을
    내부 home_hpid(HPID)로 해석.
    - "A1100010" 같이 HPID 형태면 그대로 사용
    - 아니면 이름 substring 매칭으로 찾아본다.
    """
    if not hospital_followup:
        return None

    text = hospital_followup.strip()
    if not text:
        return None

    # 1) 이미 HPID 형식인 경우
    if text.startswith("A") and text[1:].isdigit():
        return text

    # 2) 이름 기반 매칭
    target = text.replace(" ", "")

    for s in summaries:
        basic = s.basic
        name = s.name or (basic.name if basic and basic.name else None)
        if not name:
            continue
        cand = name.replace(" ", "")
        if target in cand:
            return s.id

    return None

def _compute_coverage_score_and_level(
    required_groups: List[str],
    groups_with_beds: List[str],
) -> Tuple[float, str]:
    """
    required_procedure_groups 대비 실제로 effective_beds>0 인 그룹 비율 + 등급 계산

    - score = (coverage_count / len(required_groups))  (0.0 ~ 1.0)
    - level:
        * FULL   : score == 1.0
        * HIGH   : 0.75 <= score < 1.0
        * MEDIUM : 0.5  <= score < 0.75
        * LOW    : 0.0  <  score < 0.5
        * NONE   : score == 0.0
    """
    if not required_groups:
        return 0.0, "NONE"

    req_set = set(required_groups)
    covered = sum(1 for g in groups_with_beds if g in req_set)
    score = covered / len(req_set)

    if score <= 0.0:
        level = "NONE"
    elif score >= 1.0:
        level = "FULL"
    elif score >= 0.75:
        level = "HIGH"
    elif score >= 0.5:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level


# ----------------- coverage 기반 priority/설명 헬퍼 -----------------

# coverage level → 가중치 매핑
COVERAGE_WEIGHT_BY_LEVEL = {
    "FULL": 1.00,   # 요구 시술 100% 커버
    "HIGH": 0.95,   # 대부분 커버
    "MEDIUM": 0.90, # 절반 이상
    "LOW": 0.80,    # 일부만
    "NONE": 0.70,   # 사실상 커버 안 됨
}

# coverage level → 한글 설명
COVERAGE_LEVEL_LABEL_KO = {
    "FULL": "요청된 시술을 거의 모두 커버",
    "HIGH": "핵심 시술 대부분 가능",
    "MEDIUM": "일부 핵심 시술만 가능",
    "LOW": "필수 시술 중 일부만 가능",
    "NONE": "요청 시술과 직접 일치하는 시술은 거의 없음",
}


def _apply_coverage_weight(
    base_score: float,
    coverage_level: str,
    coverage_score: float | None = None,
) -> float:
    """
    base_score(= home 병원 가산 + 총 유효 병상)를
    coverage level/score에 따라 살짝 가중치 주는 함수.
    """
    weight = COVERAGE_WEIGHT_BY_LEVEL.get(coverage_level, 0.90)

    # coverage_score(0.0~1.0)로 미세 튜닝 (대략 ±0.05 안쪽에서만 움직이게)
    if coverage_score is not None:
        bonus = 0.1 * (coverage_score - 0.7)  # 0.7을 기준으로
        bonus = max(-0.05, min(0.05, bonus))
        weight += bonus

    # 가중치
    weight = max(0.5, min(1.1, weight))

    return round(base_score * weight, 1)


def _build_reason_summary_with_coverage(
    *,
    ktas: int,
    complaint_label: str,
    groups_with_beds_labels: List[str],
    groups_with_beds: List[str],
    total_eff: int,
    coverage_level: str,
    coverage_score: float,
) -> str:
    """
    RoutingCandidateHospital.reason_summary용 문장을
    coverage 정보까지 포함해서 만들어주는 헬퍼.
    """
    if groups_with_beds_labels:
        groups_str = ", ".join(groups_with_beds_labels)
    elif groups_with_beds:
        groups_str = ", ".join(groups_with_beds)
    else:
        groups_str = "관련 시술"

    coverage_desc = COVERAGE_LEVEL_LABEL_KO.get(
        coverage_level,
        f"커버리지 {coverage_level}",
    )
    coverage_pct = int(round(coverage_score * 100))

    return (
        f"KTAS {ktas}, 주증상 '{complaint_label}' 환자에 대해 "
        f"{groups_str} 기준 총 유효 병상 {total_eff}개가 남아 있어 후보로 선정됨. "
        f"(시술 커버리지: {coverage_desc}, 약 {coverage_pct}% 충족)"
    )



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 * 허용, 추후 제한 필요해보임
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 클라이언트 인스턴스
ermct_client = ErmctClient()

@app.on_event("startup")
async def startup_event():
    print(" [Startup] Whisper AI 모델 로딩 시작...")
    get_whisper_model()
    print(" [Startup] Whisper AI 모델 로딩 완료!")

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get(
    "/api/hospitals/realtime",
    response_model=list[HospitalRealtime],
)
def get_realtime_hospitals(
    sido: str = Query(..., description="시도명 (예: 서울특별시)"),
    sigungu: str = Query(..., description="시군구명 (예: 강남구)"),
    num_rows: int = Query(50, ge=1, le=200),
):
    """
    특정 시/군/구 기준 실시간 응급실 가용 병상 정보 반환
    """
    return ermct_client.get_realtime_beds(
        sido=sido,
        sigungu=sigungu,
        num_rows=num_rows,
    )


@app.get("/debug/hospitals/realtime/xml")
def debug_realtime_xml(
    sido: str = Query(...),
    sigungu: str = Query(...),
    num_rows: int = Query(5),
    page_no: int = Query(1),
):
    xml = ermct_client.debug_raw_realtime_xml(
        sido=sido,
        sigungu=sigungu,
        num_rows=num_rows,
        page_no=page_no,
    )
    # XML로 반환
    return Response(content=xml, media_type="application/xml")


# --------------------------------------------------------------------
# 1) 응급의료기관 기본정보 조회 (getEgytBassInfoInqire)
# --------------------------------------------------------------------
@app.get(
    "/api/hospitals/basic",
    response_model=HospitalBasicInfo | None,
)
def get_hospital_basic(
    hpid: str = Query(..., description="병원 기관 코드 (HPID, 예: A1100010)"),
):
    """
    HPID 기준 응급의료기관 기본정보 조회
    (주소, 대표전화, 응급실 전화, 위경도 등)
    """
    return ermct_client.get_basic_info(hpid=hpid)


# --------------------------------------------------------------------
# 2) 중증질환자 수용가능 정보 조회 (getSrsillDissAceptncPosblInfoInqire)
# --------------------------------------------------------------------
@app.get(
    "/api/hospitals/serious",
    response_model=list[SeriousDiseaseStatus],
)
def get_serious_hospitals(
    sido: str = Query(..., description="시도명 (예: 서울특별시)"),
    sigungu: str = Query(..., description="시군구명 (예: 강남구)"),
    sm_type: int = Query(
        1,
        description="SM_TYPE (가이드 기준 중증질환 분류 타입: 1/2/3 등)",
    ),
    num_rows: int = Query(30, ge=1, le=200),
    page_no: int = Query(1, ge=1),
):
    """
    시/군/구 기준 중증질환자 수용가능정보 조회

    - MKioskTyXX: 각 중증질환 카테고리의 수용 가능/불가 상태
    - MKioskTyXXMsg: 해당 상태에 대한 상세 메시지
    """
    return ermct_client.get_serious_acceptance(
        sido=sido,
        sigungu=sigungu,
        sm_type=sm_type,
        num_rows=num_rows,
        page_no=page_no,
    )


# --------------------------------------------------------------------
# 3) 응급실 및 중증질환 메시지 조회 (getEmrrmSrsillDissMsgInqire)
# --------------------------------------------------------------------
@app.get(
    "/api/hospitals/messages",
    response_model=list[HospitalMessage],
)
def get_hospital_messages(
    hpid: str = Query(..., description="병원 기관 코드 (HPID, 예: A1100010)"),
    num_rows: int = Query(10, ge=1, le=100),
    page_no: int = Query(1, ge=1),
):
    """
    HPID 기준 응급실/중증질환 메시지 조회

    - 장비 고장, 병상 과밀, 특정 중증질환 수용 불가 등 메시지
    - symBlkMsg / symBlkMsgTyp / symTypCod / symTypCodMag 등 포함
    """
    return ermct_client.get_emergency_messages(
        hpid=hpid,
        num_rows=num_rows,
        page_no=page_no,
    )


# --------------------------------------------------------------------
# 4) 병원 정보 요약 (summary)
# --------------------------------------------------------------------
@app.get(
    "/api/hospitals/summary",
    response_model=HospitalSummary,
)
def get_hospital_summary(
    hpid: str = Query(..., description="병원 기관 코드 (HPID, 예: A1100010)"),
    # 아래 둘은 실시간/중증 수용 정보 찾을 때만 필요
    sido: str | None = Query(
        None,
        description="시도명 (실시간/중증 수용 정보를 함께 조회하려면 필요, 예: 서울특별시)",
    ),
    sigungu: str | None = Query(
        None,
        description="시군구명 (실시간/중증 수용 정보를 함께 조회하려면 필요, 예: 강남구)",
    ),
    sm_type: int = Query(
        1,
        description="중증질환 분류 타입(SM_TYPE), 가이드 기본값 1",
    ),
):
    """
    단일 병원(HPID)에 대한 통합 요약 정보

    - basic: getEgytBassInfoInqire (기본정보)
    - realtime: getEmrrmRltmUsefulSckbdInfoInqire (실시간 가용 병상)
      * sido/sigungu가 주어지면 해당 지역에서 HPID 매칭
    - serious: getSrsillDissAceptncPosblInfoInqire (중증질환 수용 가능정보)
    - messages: getEmrrmSrsillDissMsgInqire (응급실/중증 관련 메시지)
    """

    # 1) 기본정보 (HPID 기반)
    basic = ermct_client.get_basic_info(hpid=hpid)

    # 2) 실시간 병상/장비 정보, 중증 수용, 외상센터 여부 (sido/sigungu가 들어온 경우에만 시도)
    realtime: HospitalRealtime | None = None
    serious: SeriousDiseaseStatus | None = None
    trauma_hpids: Set[str] = set()

    if sido and sigungu:
        # (1) 실시간 병상 리스트 → HPID로 필터
        realtime_list = ermct_client.get_realtime_beds(
            sido=sido,
            sigungu=sigungu,
            num_rows=200,
            page_no=1,
        )
        for r in realtime_list:
            if r.id == hpid:
                realtime = r
                break

        # (2) 중증질환 수용 가능 정보 리스트 → HPID로 필터
        serious_list = ermct_client.get_serious_acceptance(
            sido=sido,
            sigungu=sigungu,
            sm_type=sm_type,
            num_rows=200,
            page_no=1,
        )
        for s in serious_list:
            s_hpid = getattr(s, "id", None)
            if not s_hpid and getattr(s, "raw_fields", None):
                s_hpid = s.raw_fields.get("hpid")
            if s_hpid == hpid:
                serious = s
                break

        # (3) 외상센터 목록 조회해서 HPID 세트 구성
        trauma_list = ermct_client.get_trauma_centers(
            sido=sido,
            sigungu=sigungu,
            num_rows=200,
            page_no=1,
        )
        trauma_hpids = {t.id for t in trauma_list if t.id}

    # 3) 응급실/중증 메시지 (HPID 기반)
    messages = ermct_client.get_emergency_messages(
        hpid=hpid,
        num_rows=50,
        page_no=1,
    )

    # 4) name 결정 (basic → realtime → messages 순으로 Fallback)
    name: str | None = None
    if basic and basic.name:
        name = basic.name
    elif realtime and realtime.name:
        name = realtime.name
    elif messages:
        first_msg = messages[0]
        msg_name = getattr(first_msg, "name", None)
        if msg_name:
            name = msg_name

    is_trauma_center = False
    if trauma_hpids:
        is_trauma_center = hpid in trauma_hpids

    # 요약 객체 생성
    summary = HospitalSummary(
        id=hpid,
        name=name,
        basic=basic,
        realtime=realtime,
        serious=serious,
        messages=messages,
        is_trauma_center=is_trauma_center,
    )

    # 수술/시술 그룹별 가능 여부 계산해서 필드 채우기
    summary.procedure_availability = compute_procedure_availability(summary)

    return summary


# --------------------------------------------------------------------
# 5) 디버그용 raw xml
# --------------------------------------------------------------------
@app.get("/debug/hospitals/serious/xml")
def debug_serious_xml(
    sido: str,
    sigungu: str,
    sm_type: int = 1,
    num_rows: int = 30,
    page_no: int = 1,
):
    # 원시 XML이 필요하면 ErmctClient에 이런 메서드 하나 추가해도 됨:
    xml = ermct_client.debug_raw_serious_xml(
        sido=sido,
        sigungu=sigungu,
        sm_type=sm_type,
        num_rows=num_rows,
        page_no=page_no,
    )
    return Response(content=xml, media_type="application/xml")


@app.get(
    "/api/hospitals/summary/by-region",
    response_model=list[HospitalSummary],
)
def get_hospital_summaries_by_region(
    sido: str = Query(..., description="시도명 (예: 서울특별시)"),
    sigungu: str = Query(..., description="시군구명 (예: 강남구)"),
    sm_type: int = Query(
        1,
        description="중증질환 분류 타입(SM_TYPE), 가이드 기본값 1",
    ),
    num_rows: int = Query(
        200,
        ge=1,
        le=500,
        description="실시간 병상 조회 시 한 번에 가져올 최대 병원 수",
    ),
):
    """
    특정 시/군/구 내 모든 응급의료기관에 대한 통합 요약 정보 리스트

    - basic: getEgytBassInfoInqire (기본정보)
    - realtime: getEmrrmRltmUsefulSckbdInfoInqire (실시간 가용 병상)
    - serious: getSrsillDissAceptncPosblInfoInqire (중증질환 수용 가능정보)
    - messages: getEmrrmSrsillDissMsgInqire (응급실/중증 관련 메시지)
    """

    # 1) 해당 지역 실시간 병상 정보 → 병원 리스트(HPID)
    realtime_list: List[HospitalRealtime] = ermct_client.get_realtime_beds(
        sido=sido,
        sigungu=sigungu,
        num_rows=num_rows,
        page_no=1,
    )

    # 2) 해당 지역 중증질환 수용 가능 정보 한 번에 조회
    serious_list: List[SeriousDiseaseStatus] = ermct_client.get_serious_acceptance(
        sido=sido,
        sigungu=sigungu,
        sm_type=sm_type,
        num_rows=num_rows,
        page_no=1,
    )

    # 2-1) 중증 정보 HPID -> SeriousDiseaseStatus 매핑
    serious_by_hpid: Dict[str, SeriousDiseaseStatus] = {}
    for s in serious_list:
        s_hpid: Optional[str] = None

        # 스키마에 id 필드를 따로 추가해뒀다면 우선 사용
        if hasattr(s, "id"):
            s_hpid = getattr(s, "id")

        # id가 없으면 raw_fields에서 hpid 추출
        if not s_hpid and getattr(s, "raw_fields", None):
            s_hpid = s.raw_fields.get("hpid") or s.raw_fields.get("HPID")

        if s_hpid:
            serious_by_hpid[s_hpid] = s

    # 2-2) 외상센터 목록도 한 번만 조회해서 HPID set으로
    trauma_list: List[TraumaCenter] = ermct_client.get_trauma_centers(
        sido=sido,
        sigungu=sigungu,
        num_rows=200,
        page_no=1,
    )
    trauma_hpids: Set[str] = {t.id for t in trauma_list if t.id}

    results: List[HospitalSummary] = []
    seen: Set[str] = set()

    # 3) 실시간 병상 리스트 기준으로 병원별 summary 구성
    for r in realtime_list:
        hpid = r.id
        if not hpid or hpid in seen:
            continue
        seen.add(hpid)

        # (1) 기본 정보
        basic = ermct_client.get_basic_info(hpid=hpid)

        # (2) 중증 정보: 미리 만든 매핑에서 가져오기
        serious = serious_by_hpid.get(hpid)

        # (3) 응급실/중증 메시지
        messages = ermct_client.get_emergency_messages(
            hpid=hpid,
            num_rows=50,
            page_no=1,
        )

        # (4) 이름 결정 (basic → realtime → messages 순)
        name: Optional[str] = None
        if basic and basic.name:
            name = basic.name
        elif r.name:
            name = r.name
        elif messages:
            first_msg = messages[0]
            msg_name = getattr(first_msg, "name", None)
            if msg_name:
                name = msg_name

        summary = HospitalSummary(
            id=hpid,
            name=name,
            basic=basic,
            realtime=r,
            serious=serious,
            messages=messages,
            is_trauma_center=(hpid in trauma_hpids),
        )

        summary.procedure_availability = compute_procedure_availability(summary)

        results.append(summary)

    return results


# --------------------------------------------------------------------
# 6) 병원 필터링 (2학기 대비 1단계에서 지역을 받는 버전)
# --------------------------------------------------------------------
@app.post("/api/triage/recommend", response_model=list[RecommendedHospital])
def recommend_hospitals(triage: TriageRequest = Body(...)):
    """
    환자 정보(KTAS, 주호소 증상, 원내/기존 병원)를 입력받아
    - 해당 지역(sido, sigungu)의 병원 요약을 가져오고
    - 주호소 증상에 맞는 procedure group들을 계산한 뒤
    - 수술 가능 + 병상 남아있는 병원만 필터링해서 추천 리스트를 반환
    """
    # 1) 지역 정보는 이제 요청에서 직접 받음
    sido = triage.sido
    sigungu = triage.sigungu

    # 2) 이 complaint가 요구하는 procedure group 목록
    required_groups = required_procedure_groups_for_complaint(triage.complaint_id)
    if not required_groups:
        # 정의 안 된 complaint면 빈 리스트 반환 (혹은 400 에러로 바꿔도 됨)
        return []

    # 3) 해당 지역 병원 요약 가져오기
    #    이미 위에서 정의한 get_hospital_summaries_by_region() 함수를 그대로 재사용
    summaries: List[HospitalSummary] = get_hospital_summaries_by_region(
        sido=sido,
        sigungu=sigungu,
        sm_type=1,
        num_rows=200,
    )

    candidates: List[RecommendedHospital] = []

    for s in summaries:
        # 4) 이 병원이 해당 procedure group들에 대해
        #    수용 가능 + 병상 몇 개 있는지 계산
        proc_status = procedure_status_for_hospital(s, required_groups)
        # proc_status: {group_id: {"api_beds": int, "effective_beds": int}}

        # effective_beds > 0 인 그룹만 따로 추출
        groups_with_beds = [
            gid
            for gid, info in proc_status.items()
            if info.get("effective_beds", 0) > 0
        ]

        # 시술 자체가 전부 불가능하면 스킵
        if not groups_with_beds:
            continue

        # complaint 전체 기준 병상 수는 bed_type 합집합으로 계산
        if s.realtime:
            _, total_eff, _ = get_effective_beds_for_groups(
                hpid=s.id,
                realtime=s.realtime,
                group_ids=groups_with_beds,
            )
        else:
            total_eff = 0

        # 5) 수용 가능하지만 병상이 0이면 필터링
        if total_eff <= 0:
            continue

        # coverage_score / coverage_level 계산
        coverage_score, coverage_level = _compute_coverage_score_and_level(
            required_groups=required_groups,
            groups_with_beds=groups_with_beds,
        )

        # 6) RecommendedHospital 엔티티로 변환
        candidates.append(
            RecommendedHospital(
                id=s.id,
                name=s.name or (s.basic.name if s.basic else s.id),
                ktas=triage.ktas,
                complaint_id=triage.complaint_id,
                total_effective_beds=total_eff,
                procedure_beds=proc_status,
                basic=s.basic,
                realtime=s.realtime,
                serious=s.serious,
                messages=s.messages or [],
                coverage_score=coverage_score,
                coverage_level=coverage_level,
            )
        )

    # 7) 정렬: 거리 안 쓰고,
    #    - home_hpid(기존 다니던 병원) 우선
    #    - 그 다음 병상 많은 순
    home_hpid = triage.home_hpid

    def sort_key(h: RecommendedHospital):
        is_home = 1 if (home_hpid and h.id == home_hpid) else 0
        return (-is_home, -h.total_effective_beds)

    candidates.sort(key=sort_key)

    return candidates


# --------------------------------------------------------------------
# 7) 중증 외상센터 정보
# --------------------------------------------------------------------
@app.get("/api/hospitals/trauma/by-region", response_model=List[TraumaCenter])
def get_trauma_by_region(
    sido: str,
    sigungu: str,
    num_rows: int = 50,
):
    return ermct_client.get_trauma_centers(
        sido=sido,
        sigungu=sigungu,
        num_rows=num_rows,
        page_no=1,
    )


# --------------------------------------------------------------------
# 8) 지역 기준 증상 출력 (디버그용)
# --------------------------------------------------------------------
@app.get(
    "/api/hospitals/complaint-coverage/by-region",
    response_model=list[HospitalComplaintCoverage],
)
def get_complaint_coverage_by_region(
    sido: str = Query(..., description="시도명 (예: 서울특별시)"),
    sigungu: str = Query(..., description="시군구명 (예: 강남구)"),
    sm_type: int = Query(
        1,
        description="중증질환 분류 타입(SM_TYPE), 가이드 기본값 1",
    ),
    num_rows: int = Query(
        200,
        ge=1,
        le=500,
        description="실시간 병상 조회 시 한 번에 가져올 최대 병원 수",
    ),
):
    """
    특정 시/군/구 내 모든 병원에 대해
    - MKioskTy 기반으로
    - 이 병원이 어떤 complaint(1~10)를 커버하는지 미리 계산해서 내려주는 디버깅용 API.
    """

    # 기존 요약 API 로직을 재사용
    summaries: List[HospitalSummary] = get_hospital_summaries_by_region(
        sido=sido,
        sigungu=sigungu,
        sm_type=sm_type,
        num_rows=num_rows,
    )

    results: List[HospitalComplaintCoverage] = []

    for s in summaries:
        supported = complaints_supported_by_hospital(s)  # Set[int]
        # 정렬해서 내려주자
        supported_ids = sorted(list(supported))

        labels = [COMPLAINT_LABELS[cid] for cid in supported_ids if cid in COMPLAINT_LABELS]

        results.append(
            HospitalComplaintCoverage(
                id=s.id,
                name=s.name
                or (s.basic.name if s.basic else None)
                or s.id,
                supported_complaints=supported_ids,
                supported_complaint_labels=labels,
            )
        )

    return results

# --------------------------------------------------------------------
# 9) 증상 기준 병원 출력 (디버그용)
# --------------------------------------------------------------------
@app.post(
    "/api/triage/candidates",
    response_model=RoutingCandidateResponse,
)
def get_routing_candidates(triage: TriageRequest = Body(...)):
    """
    '가능 수술 기준' 후보 병원 리스트를 반환하는 엔드포인트.

    - 상세 과정:
      * 해당 지역(sido, sigungu)의 병원들 중
      * complaint_id에 맞는 procedure group을 수용 가능하고
      * 그 procedure에 대해 effective_beds > 0 인 병원만 골라서
      * 위치/연락처 + 근거 정보와 함께 리스트로 넘겨준다.
    """

    # 1) 이 complaint가 요구하는 procedure group 목록
    required_groups = required_procedure_groups_for_complaint(triage.complaint_id)
    if not required_groups:
        # 정의 안 된 complaint면 빈 리스트
        return RoutingCandidateResponse(
            hid=triage.home_hpid or None,
            hospitals=[],
        )

    # 2) 지역 내 병원 summary들 불러오기
    summaries: List[HospitalSummary] = get_hospital_summaries_by_region(
        sido=triage.sido,
        sigungu=triage.sigungu,
        sm_type=1,
        num_rows=200,
    )

    candidates: List[RoutingCandidateHospital] = []
    home_hpid = triage.home_hpid
    complaint_label = COMPLAINT_LABELS.get(
        triage.complaint_id,
        f"Complaint {triage.complaint_id}",
    )

    for s in summaries:
        basic = s.basic
        if not basic:
            continue

        lat = basic.latitude
        lon = basic.longitude
        if lat is None or lon is None:
            # 위치정보 없는 병원은 T-MAP에서 쓸 수 없으니 제외
            continue

        # 응급실 있는 병원만
        duty_eryn = basic.raw_fields.get("dutyEryn") if basic.raw_fields else None
        if duty_eryn != "1":
            continue

        # 3) 이 병원이 required_groups에 대해 얼마나 수용 가능한지 계산
        proc_status = procedure_status_for_hospital(s, required_groups)
        if not proc_status:
            continue

        # effective_beds > 0 인 그룹만 뽑기
        groups_with_beds = [
            gid
            for gid, info in proc_status.items()
            if info.get("effective_beds", 0) > 0
        ]

        # 하나도 병상이 없는 병원은 후보에서 제외
        if not groups_with_beds:
            continue

        # 🔹 complaint 전체 기준 병상 수 = bed_type 합집합으로 계산
        if s.realtime:
            _, total_eff, _ = get_effective_beds_for_groups(
                hpid=s.id,
                realtime=s.realtime,
                group_ids=groups_with_beds,
            )
        else:
            total_eff = 0

        if total_eff <= 0:
            continue

        coverage_score, coverage_level = _compute_coverage_score_and_level(
            required_groups=required_groups,
            groups_with_beds=groups_with_beds,
        )

        has_any_bed = True  # 위에서 이미 필터링함

        # 코드 → 라벨 변환
        required_group_labels = humanize_procedure_groups(required_groups)
        groups_with_beds_labels = humanize_procedure_groups(groups_with_beds)

        # 4) MKioskTy 기준 이 병원이 커버 가능한 complaint들 계산
        supported_complaints = sorted(list(complaints_supported_by_hospital(s)))
        supported_labels = [
            COMPLAINT_LABELS[cid]
            for cid in supported_complaints
            if cid in COMPLAINT_LABELS
        ]

        # 5) MKioskTy Y 플래그 수집
        mkiosk_flags: List[str] = []
        if s.serious and s.serious.mkiosk:
            mkiosk_flags.extend(
                [
                    k
                    for k, v in s.serious.mkiosk.items()
                    if v and str(v).upper().startswith("Y")
                ]
            )
        if basic.raw_fields:
            for k, v in basic.raw_fields.items():
                if not k.startswith("MKioskTy"):
                    continue
                if v and str(v).upper().startswith("Y") and k not in mkiosk_flags:
                    mkiosk_flags.append(k)

        # 6) home_hpid 여부 + 내부 priority_score
        is_home = bool(home_hpid and s.id == home_hpid)
        base_score = float(total_eff + (100 if is_home else 0))
        priority_score = _apply_coverage_weight(
            base_score=base_score,
            coverage_level=coverage_level,
            coverage_score=coverage_score,
        )

        # 7) 사람이 읽기 좋은 reason_summary (coverage 포함)
        reason = _build_reason_summary_with_coverage(
            ktas=triage.ktas,
            complaint_label=complaint_label,
            groups_with_beds_labels=groups_with_beds_labels,
            groups_with_beds=groups_with_beds,
            total_eff=total_eff,
            coverage_level=coverage_level,
            coverage_score=coverage_score,
        )

        # 8) RoutingCandidateHospital로 변환
        candidates.append(
            RoutingCandidateHospital(
                id=s.id,
                name=s.name or (basic.name if basic.name else s.id),
                address=basic.address,
                phone=basic.phone,
                emergency_phone=basic.emergency_phone,
                latitude=lat,
                longitude=lon,
                ktas=triage.ktas,
                complaint_id=triage.complaint_id,
                complaint_label=complaint_label,
                required_procedure_groups=required_groups,
                required_procedure_group_labels=required_group_labels,
                procedure_beds=proc_status,
                total_effective_beds=total_eff,
                has_any_bed=has_any_bed,
                groups_with_beds=groups_with_beds,
                groups_with_beds_labels=groups_with_beds_labels,
                supported_complaints=supported_complaints,
                supported_complaint_labels=supported_labels,
                mkiosk_flags=mkiosk_flags,
                coverage_score=coverage_score,
                coverage_level=coverage_level,
                priority_score=priority_score,
                reason_summary=reason,
            )
        )

    # 9) 정렬: coverage까지 반영된 priority_score 우선
    def sort_key(c: RoutingCandidateHospital):
        return (-c.priority_score, -c.total_effective_beds)

    candidates.sort(key=sort_key)

    return RoutingCandidateResponse(
        hid=home_hpid or None,
        hospitals=candidates,
    )


# --------------------------------------------------------------------
# 10) 지역내 raw 병상 정보 출력 (디버그용)
# --------------------------------------------------------------------
@app.get(
    "/api/hospitals/procedure-beds/by-region",
    response_model=List[HospitalProcedureBeds],
)
def get_procedure_beds_by_region(
    sido: str,
    sigungu: str,
    complaint_id: Optional[int] = None,
):
    """
    디버그용:
    - 특정 시/군/구 내 병원들에 대해
    - 주증상(complaint_id)에 해당하는 procedure group 기준으로
      병상 상태를 그대로 보여주는 엔드포인트.

    complaint_id가 없으면 모든 PROCEDURE_GROUPS에 대해 병상 계산.
    """

    # 1) 평가 대상 procedure group 결정
    if complaint_id is not None:
        groups = required_procedure_groups_for_complaint(complaint_id)
        complaint_label = COMPLAINT_LABELS.get(
            complaint_id,
            f"Complaint {complaint_id}",
        )
    else:
        groups = list(PROCEDURE_GROUPS.keys())
        complaint_label = None

    # 2) 지역별 병원 요약 불러오기
    summaries: List[HospitalSummary] = get_hospital_summaries_by_region(
        sido=sido,
        sigungu=sigungu,
        sm_type=1,
        num_rows=200,
    )

    results: List[HospitalProcedureBeds] = []

    for s in summaries:
        # 1) procedure group 병상 계산
        proc_status = procedure_status_for_hospital(s, groups)

        # 2) 응급실 일반 병상(hvec / er_beds)
        er_beds = 0
        if s.realtime and s.realtime.er_beds is not None:
            er_beds = s.realtime.er_beds

        # 3) 병상 있음 여부는 ER 기준
        has_any_bed = er_beds > 0

        basic = s.basic
        name = s.name
        if not name and basic and basic.name:
            name = basic.name

        results.append(
            HospitalProcedureBeds(
                id=s.id,
                name=name or s.id,
                complaint_id=complaint_id,
                complaint_label=complaint_label,
                required_procedure_groups=groups,
                procedure_beds=proc_status,
                er_beds=er_beds,
                has_any_bed=has_any_bed,
            )
        )

    # 병상 있는 병원 먼저 보이도록 er_beds 기준으로 정렬
    results.sort(
        key=lambda r: (-int(r.has_any_bed), -r.er_beds, r.id)
    )

    return results

# --------------------------------------------------------------------
# 10) 병상 예약 (프론트 소통용)
# --------------------------------------------------------------------
@app.post("/api/triage/reservations")
def create_bed_reservation(req: BedReservationRequest):
    """
    선택된 병원(hpid)에 대해
    - complaint_id → procedure group → bed_types 체인으로
    - 우리 쪽 in-memory pending_assignments에 '예약'을 반영하는 API.

    지금은 '1 환자 = 대표 bed_type 1개(보통 ER)'만 예약으로 반영한다.
    이후 /api/triage/candidates, /api/hospitals/procedure-beds/by-region 등에서
    get_effective_beds()를 통해 자동으로 감산된 병상 수가 반영됨.
    """
    # 1) complaint → procedure group 목록
    groups = required_procedure_groups_for_complaint(req.complaint_id)
    if not groups:
        raise HTTPException(status_code=400, detail="지원하지 않는 complaint_id 입니다.")

    # 2) procedure group → bed_types 집합
    bed_types: set[str] = set()
    for gid in groups:
        cfg = PROCEDURE_GROUPS.get(gid)
        if not cfg:
            continue
        for bt in cfg["bed_types"]:
            bed_types.add(bt)

    if not bed_types:
        raise HTTPException(
            status_code=400,
            detail="해당 complaint에 매핑된 bed_types가 없습니다.",
        )

    # 3) 대표 bed_type 하나 선택 (우선 er)
    primary_bed_type = choose_primary_bed_type(bed_types)
    if not primary_bed_type:
        raise HTTPException(
            status_code=400,
            detail="예약에 사용할 bed_type을 결정할 수 없습니다.",
        )

    # 4) in-memory pending_assignments에 예약 반영
    hospital_assign = pending_assignments[req.hpid]
    hospital_assign[primary_bed_type] += req.num_patients

    # 5) 현재 이 병원의 pending 상태를 그대로 리턴
    return {
        "hpid": req.hpid,
        "complaint_id": req.complaint_id,
        "ktas": req.ktas,
        "num_patients": req.num_patients,
        "reserved_bed_types": [primary_bed_type],
        "pending_assignments": dict(hospital_assign),
    }


# --------------------------------------------------------------------
# 11) 병상 예약 해제 (프론트 소통용)
# --------------------------------------------------------------------
@app.post("/api/triage/reservations/release")
def release_bed_reservation(req: BedReleaseRequest):
    """
    in-memory pending_assignments에서 예약을 되돌리는 API.
    - 지금은 create와 마찬가지로 '대표 bed_type 1개(보통 ER)'에 대해서만 해제한다.
    - 실제 운영이면 '환자 도착 취소/오류' 같은 상황을 표현할 때 사용.
    """
    groups = required_procedure_groups_for_complaint(req.complaint_id)
    if not groups:
        raise HTTPException(status_code=400, detail="지원하지 않는 complaint_id 입니다.")

    bed_types: set[str] = set()
    for gid in groups:
        cfg = PROCEDURE_GROUPS.get(gid)
        if not cfg:
            continue
        for bt in cfg["bed_types"]:
            bed_types.add(bt)

    if not bed_types:
        raise HTTPException(
            status_code=400,
            detail="해당 complaint에 매핑된 bed_types가 없습니다.",
        )

    primary_bed_type = choose_primary_bed_type(bed_types)
    if not primary_bed_type:
        raise HTTPException(
            status_code=400,
            detail="해제에 사용할 bed_type을 결정할 수 없습니다.",
        )

    hospital_assign = pending_assignments[req.hpid]

    # 음수로 내려가지 않도록 max(..., 0) 처리
    current = hospital_assign[primary_bed_type]
    hospital_assign[primary_bed_type] = max(current - req.num_patients, 0)

    return {
        "hpid": req.hpid,
        "complaint_id": req.complaint_id,
        "num_patients": req.num_patients,
        "released_bed_types": [primary_bed_type],
        "pending_assignments": dict(hospital_assign),
    }


# --------------------------------------------------------------------
# 12) 현재 병상 예약 현황 (디버깅용)
# --------------------------------------------------------------------
@app.post(
    "/api/ktas/route/seoul",
    response_model=RoutingCandidateResponse,
)
def route_from_ktas_seoul(req: KTASRoutingRequest = Body(...)):
    """
    KTAS 모듈에서 넘겨준 결과를 바탕으로
    - 서울특별시 전체 병원 중
    - chief_complaint에 해당하는 complaint_id(1~10)를 커버하고
    - 해당 procedure group 기준 effective_beds > 0 인 병원만
      RoutingCandidateHospital 리스트로 반환.
    """

    # 1) chief_complaint → complaint_id
    complaint_id = complaint_id_from_chief_complaint(req.chief_complaint)
    if not complaint_id:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 chief_complaint: {req.chief_complaint}",
        )

    required_groups = required_procedure_groups_for_complaint(complaint_id)
    if not required_groups:
        raise HTTPException(
            status_code=400,
            detail=f"complaint_id {complaint_id}에 매핑된 procedure group이 없습니다.",
        )

    complaint_label = COMPLAINT_LABELS.get(
        complaint_id,
        f"Complaint {complaint_id}",
    )

    routing_case = RoutingCase(
        ktas=req.ktas_level,
        complaint_id=complaint_id,
        complaint_label=complaint_label,
        required_procedure_groups=required_groups,
        required_procedure_group_labels=humanize_procedure_groups(required_groups),
    )

    # 2) 서울 전체 병원 요약 불러오기
    summaries = _get_all_seoul_summaries(sm_type=1)

    # 3) hospital_followup → home_hpid 해석
    home_hpid = _resolve_home_hpid_from_followup(
        summaries=summaries,
        hospital_followup=req.hospital_followup,
    )

    candidates: List[RoutingCandidateHospital] = []

    for s in summaries:
        basic = s.basic
        if not basic:
            continue

        lat = basic.latitude
        lon = basic.longitude
        if lat is None or lon is None:
            # 위치정보 없는 병원은 제외
            continue

        # 응급실 있는 병원만
        duty_eryn = basic.raw_fields.get("dutyEryn") if basic.raw_fields else None
        if duty_eryn != "1":
            continue

        # 4) 이 병원이 required_groups에 대해 얼마나 수용 가능한지 계산
        proc_status = procedure_status_for_hospital(s, required_groups)
        if not proc_status:
            continue

        groups_with_beds = [
            gid
            for gid, info in proc_status.items()
            if info.get("effective_beds", 0) > 0
        ]

        # 한 개도 effective_beds가 없다면 제외
        if not groups_with_beds:
            continue

        # complaint 전체 기준 병상 수 = bed_type 합집합으로 계산
        if s.realtime:
            _, total_eff, _ = get_effective_beds_for_groups(
                hpid=s.id,
                realtime=s.realtime,
                group_ids=groups_with_beds,
            )
        else:
            total_eff = 0

        if total_eff <= 0:
            continue

        has_any_bed = True  # 위에서 이미 필터링함

        coverage_score, coverage_level = _compute_coverage_score_and_level(
            required_groups=required_groups,
            groups_with_beds=groups_with_beds,
        )

        groups_with_beds_labels = humanize_procedure_groups(groups_with_beds)

        # 5) MKioskTy 기준 커버 가능한 complaint들 정보 (부가정보용)
        supported_complaints = sorted(list(complaints_supported_by_hospital(s)))
        supported_labels = [
            COMPLAINT_LABELS[cid]
            for cid in supported_complaints
            if cid in COMPLAINT_LABELS
        ]

        # 6) MKioskTy Y 플래그 수집
        mkiosk_flags: List[str] = []
        if s.serious and s.serious.mkiosk:
            mkiosk_flags.extend(
                [
                    k
                    for k, v in s.serious.mkiosk.items()
                    if v and str(v).upper().startswith("Y")
                ]
            )
        if basic.raw_fields:
            for k, v in basic.raw_fields.items():
                if not k.startswith("MKioskTy"):
                    continue
                if v and str(v).upper().startswith("Y") and k not in mkiosk_flags:
                    mkiosk_flags.append(k)

        # 7) home_hpid 여부 + priority_score
        is_home = bool(home_hpid and s.id == home_hpid)
        base_score = float(total_eff + (100 if is_home else 0))
        priority_score = _apply_coverage_weight(
            base_score=base_score,
            coverage_level=coverage_level,
            coverage_score=coverage_score,
        )

        # 8) reason_summary (coverage 포함)
        reason = _build_reason_summary_with_coverage(
            ktas=req.ktas_level,
            complaint_label=complaint_label,
            groups_with_beds_labels=groups_with_beds_labels,
            groups_with_beds=groups_with_beds,
            total_eff=total_eff,
            coverage_level=coverage_level,
            coverage_score=coverage_score,
        )

        # 9) RoutingCandidateHospital로 변환
        candidates.append(
            RoutingCandidateHospital(
                id=s.id,
                name=s.name or (basic.name if basic.name else s.id),
                address=basic.address,
                phone=basic.phone,
                emergency_phone=basic.emergency_phone,
                latitude=lat,
                longitude=lon,
                procedure_beds=proc_status,
                total_effective_beds=total_eff,
                has_any_bed=has_any_bed,
                groups_with_beds=groups_with_beds,
                groups_with_beds_labels=groups_with_beds_labels,
                supported_complaints=supported_complaints,
                supported_complaint_labels=supported_labels,
                mkiosk_flags=mkiosk_flags,
                coverage_score=coverage_score,
                coverage_level=coverage_level,
                priority_score=priority_score,
                reason_summary=reason,
            )
        )

    # 10) 정렬: priority_score 우선
    def sort_key(c: RoutingCandidateHospital):
        return (-c.priority_score, -c.total_effective_beds)

    candidates.sort(key=sort_key)

    return RoutingCandidateResponse(
        followup_id=home_hpid or None,
        case=routing_case,
        hospitals=candidates,
    )

@app.post(
    "/api/ktas/route/seoul/nearest",
    response_model=RoutingCandidateResponse,
)
async def route_seoul_nearest(
    req: NearestRoutingRequest = Body(...)
):
    """
    1단계 라우팅 결과(서울 전체 후보들) + 사용자 위치를 받아,
    Tmap 거리 기준 상위 3개 병원만 골라 distance/duration_sec을 채워서 반환.
    """

    # 1) distance_logic에 줄 payload 구성
    hospitals_payload = [
        {
            "name": h.name,
            "latitude": h.latitude,
            "longitude": h.longitude,
            "reason_summary": h.reason_summary,
        }
        for h in req.hospitals
    ]

    # 2) Tmap API로 모든 후보 병원까지 거리/시간 계산
    results = await calculate_all_distances_async(
        user_lat=req.user_lat,
        user_lon=req.user_lon,
        hospitals=hospitals_payload,
    )

    # 3) 거리 기준 상위 3개만 선택
    top3_results = get_top3(results)

    # 4) name 기준으로 매핑 (이름이 중복될 가능성이 낮다고 가정)
    result_by_name = {r["name"]: r for r in top3_results}

    top3_hospitals: List[RoutingCandidateHospital] = []

    for h in req.hospitals:
        r = result_by_name.get(h.name)
        if not r:
            continue

        # 기존 필드는 그대로 두고 distance, duration만 덧입힘
        data = h.model_dump()
        data["distance"] = float(r["distance"])
        data["duration_sec"] = int(r["duration_sec"])

        top3_hospitals.append(RoutingCandidateHospital(**data))

    # 5) followup_id는 그대로 유지, 병원 리스트만 top3로 교체
    return RoutingCandidateResponse(
        followup_id=req.followup_id,
        case=req.case,
        user_lat=req.user_lat,
        user_lon=req.user_lon,
        hospitals=top3_hospitals,
    )


# 파일 맨 끝에 붙여넣으세요

@app.post("/api/ktas/predict-audio", response_model=RoutingCandidateResponse)
async def predict_audio(audio: UploadFile = File(...)):
    """
    [Stage 1 + Stage 2 통합]
    """
    # 1. [Stage 1] 음성 엔진 실행
    print("\n[Stage 1] 음성 분석 및 KTAS 분류 중...")
    stage1_result = ktas_from_audio(audio.file)

    # 2. 데이터 변환
    payload_dict = build_stage2_payload(stage1_result)
    req_obj = KTASRoutingRequest(**payload_dict)

    # 3. [Stage 2] 병원 추천 로직 실행 (변수에 담기!)
    print("[Stage 2] 병원 필터링 및 순위 선정 중...")
    
    # ★ 여기서 바로 return 하지 말고, 변수(final_response)에 저장합니다.
    final_response = route_from_ktas_seoul(req_obj) 

    # ====================================================
    # ★ 터미널 출력용 코드 (여기서 확인!)
    # ====================================================
    print("\n" + "="*60)
    print(f" 🚑 [최종 추천 결과] 총 {len(final_response.hospitals)}개 병원 발견")
    print("="*60)

    # 상위 3개 병원만 터미널에 찍어보기
    for i, hosp in enumerate(final_response.hospitals[:3]):
        print(f" {i+1}순위: {hosp.name}")
        print(f"    - 병상수: {hosp.total_effective_beds}개")
        print(f"    - 추천사유: {hosp.reason_summary}")
        print("-" * 40)
    
    print("="*60 + "\n")
    # ====================================================

    # 4. 최종 리턴
    return final_response
