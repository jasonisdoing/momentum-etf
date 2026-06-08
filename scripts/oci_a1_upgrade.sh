#!/usr/bin/env bash
# OCI A1.Flex 단계적 증설 (1 → 2 → 4 OCPU).
#
# 동작:
#   1단계: 현재 인스턴스를 2 OCPU/12GB 로 update_instance 시도. 자리 없으면 90초 후 재시도.
#   2단계: 1단계 성공 후 4 OCPU/24GB 로 update_instance 시도.
#   각 단계 모두 update API 응답만 믿지 않고 get_instance 로 실제 shape_config 확인.
#
# 전제:
#   - 로컬에 OCI CLI 설치되어 있고 ~/.oci/config 인증 셋업 완료
#   - 대상 인스턴스가 VM.Standard.A1.Flex (ARM)
#   - SHELL: bash (zsh 권장 안 함)
#
# 사용:
#   bash scripts/oci_a1_upgrade.sh
#
# 중단:
#   Ctrl+C — 안전. 중간 단계에서 멈춰도 인스턴스 상태는 마지막 성공한 단계로 유지됨.

set -uo pipefail

# ─────────────────────────────────────────────────────────────
# 설정 (이 부분만 수정)
# ─────────────────────────────────────────────────────────────
INSTANCE_OCID="ocid1.instance.oc1.ap-chuncheon-1.an4w4ljrbfkqmnycqt3vekeuwnpgv5hk6lfw5u7gvnhayjnfxmewrqp77rya"

# 백오프 (초)
OOH_BACKOFF=60       # Out of host capacity → 60초
RATE_BACKOFF=180     # 429 Too Many Requests → 180초
ERR_BACKOFF=60       # 기타 에러 → 60초

LOG_FILE="${HOME}/oci_a1_upgrade.log"
# ─────────────────────────────────────────────────────────────

log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG_FILE"
}

require_oci() {
    if ! command -v oci >/dev/null 2>&1; then
        log "ERROR: OCI CLI(oci) 가 설치되어 있지 않습니다."
        log "       설치: bash -c \"\$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)\""
        exit 1
    fi
    if [ ! -f "${HOME}/.oci/config" ]; then
        log "ERROR: ~/.oci/config 가 없습니다. 'oci setup config' 로 인증을 셋업하세요."
        exit 1
    fi
}

# 현재 shape_config 값 출력 ("OCPU MEM" 형태). 실패 시 "0 0".
get_current_shape() {
    local out ocpus mem
    out="$(oci compute instance get \
        --instance-id "$INSTANCE_OCID" \
        --query 'data."shape-config".[ocpus, "memory-in-gbs"]' \
        --raw-output 2>/dev/null)" || { echo "0 0"; return; }
    # 응답 예: [ 1.0, 6.0 ]  → 1 6
    ocpus="$(echo "$out" | tr -d '[]\n ,' | awk '{print substr($0, 1, index($0, ".")-1)}')"
    mem="$(echo "$out" | tr -d '[]\n ' | awk -F',' '{print $2}' | awk '{print substr($0, 1, index($0, ".")-1)}')"
    [ -z "$ocpus" ] && ocpus=0
    [ -z "$mem" ] && mem=0
    echo "$ocpus $mem"
}

# 단계: update_instance + 실제 검증. 성공 시 0, 자리 없음 1, rate-limit 2, 기타 3.
try_resize() {
    local target_ocpu="$1"
    local target_mem="$2"
    local response

    log "→ update_instance 시도: ${target_ocpu} OCPU / ${target_mem} GB"

    # --force: 확인 프롬프트 없음. stdout=응답, stderr=에러
    response="$(oci compute instance update \
        --instance-id "$INSTANCE_OCID" \
        --shape-config "{\"ocpus\": ${target_ocpu}, \"memoryInGBs\": ${target_mem}}" \
        --force 2>&1)" && rc=0 || rc=$?

    if [ $rc -ne 0 ]; then
        # 에러 응답에서 흔한 패턴 분기
        if echo "$response" | grep -qi "Out of host capacity"; then
            log "  ✖ Out of host capacity (자리 없음)"
            return 1
        fi
        if echo "$response" | grep -qiE "TooManyRequests|status: 429"; then
            log "  ✖ 429 Too Many Requests (rate-limit)"
            return 2
        fi
        if echo "$response" | grep -qi "LimitExceeded"; then
            log "  ✖ LimitExceeded — Always Free 한도 초과 가능성. 응답:"
            log "$response"
            return 3
        fi
        log "  ✖ 기타 에러:"
        log "$response"
        return 3
    fi

    # API 가 성공이라 응답해도 실제 적용까지 시간이 필요. 최대 90초 대기 후 검증.
    log "  ⏳ 응답 OK — 실제 shape 적용 검증 중 (최대 90초)…"
    local waited=0 cur_ocpu cur_mem
    while [ "$waited" -lt 90 ]; do
        sleep 10
        waited=$((waited + 10))
        read -r cur_ocpu cur_mem < <(get_current_shape)
        if [ "$cur_ocpu" = "$target_ocpu" ] && [ "$cur_mem" = "$target_mem" ]; then
            log "  ✅ 검증 성공: 현재 ${cur_ocpu} OCPU / ${cur_mem} GB"
            return 0
        fi
        log "    … 진행 중 (${waited}s, 현재 ${cur_ocpu} OCPU / ${cur_mem} GB)"
    done
    log "  ⚠️ API 는 성공했으나 90초 안에 적용 미확인 (현재 ${cur_ocpu}/${cur_mem}). 다음 루프에서 재시도."
    return 3
}

# 메인 루프: 한 단계의 목표를 잡고 성공할 때까지 반복.
run_stage() {
    local stage_name="$1"
    local target_ocpu="$2"
    local target_mem="$3"

    log ""
    log "══════ ${stage_name} 시작 (목표: ${target_ocpu} OCPU / ${target_mem} GB) ══════"

    # 이미 목표 달성? 그러면 skip.
    local cur_ocpu cur_mem
    read -r cur_ocpu cur_mem < <(get_current_shape)
    log "현재 shape: ${cur_ocpu} OCPU / ${cur_mem} GB"
    if [ "$cur_ocpu" -ge "$target_ocpu" ] && [ "$cur_mem" -ge "$target_mem" ]; then
        log "이미 목표 이상. 단계 건너뜀."
        return 0
    fi

    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        log "─ 시도 #${attempt}"
        if try_resize "$target_ocpu" "$target_mem"; then
            log "🎉 ${stage_name} 성공!"
            return 0
        fi
        case $? in
            1) sleep "$OOH_BACKOFF" ;;
            2) sleep "$RATE_BACKOFF" ;;
            *) sleep "$ERR_BACKOFF" ;;
        esac
    done
}

# ─────────────────────────────────────────────────────────────
require_oci
log "===== A1.Flex 단계적 증설 시작 ====="
log "인스턴스: ${INSTANCE_OCID:0:60}…"

run_stage "1단계: 2 OCPU / 12 GB" 2 12
run_stage "2단계: 4 OCPU / 24 GB" 4 24

log ""
log "🏁 전체 증설 완료. 최종 검증:"
read -r FINAL_OCPU FINAL_MEM < <(get_current_shape)
log "최종 shape: ${FINAL_OCPU} OCPU / ${FINAL_MEM} GB"
