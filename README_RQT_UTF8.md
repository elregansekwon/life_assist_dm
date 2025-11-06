# rqt UTF-8 인코딩 오류 해결 가이드

## 💥 문제 상황

rqt를 실행할 때 `rqt_service_caller`에서 한글을 포함한 서비스 요청을 처리하면 다음과 같은 오류가 발생합니다:

```
UnicodeEncodeError: 'ascii' codec can't encode characters in position 107-108: ordinal not in range(128)
```

## 🔍 원인 분석

**`rqt_service_caller`가 한글 포함된 ROS2 서비스 요청(request)을 ASCII로 출력하려다가 깨져서 종료됩니다.**

예시:
- `'권서연'`
- `'안녕'`
- `'예약했어'`
- `'나는 저녁마다 혈압약을 먹어'`

**근본 원인:**
- `rqt` 내부에서 PyQt 위젯 로그를 출력할 때, `print()` 또는 `qWarning()`이 **ASCII 인코딩을 강제**로 사용
- **`service_caller_widget.py` 375번째 줄에서 `%r` 포맷 사용 → `repr()` 호출로 ASCII 강제**
- UTF-8 문자(한글)를 처리하지 못함 → `UnicodeEncodeError`
- **ROS2나 네 코드가 잘못된 것이 아니라, rqt GUI가 한글 포함된 서비스 요청을 출력하는 순간 죽는 버그**

**문제 코드 (375번째 줄):**
```python
qWarning('ServiceCaller.on_call_service_button_clicked(): request:\n%r' % (request))
```
- `%r`은 `repr()`을 호출하여 ASCII 인코딩을 강제함
- 한글 문자가 포함된 `request` 객체를 출력하려 할 때 크래시 발생

## ✅ 해결 방법

### **방법 0: 근본 해결 - rqt_service_caller 버그 수정 (가장 확실) 🔧⭐**

**ROS2 Humble의 rqt_service_caller 버그를 직접 수정하는 방법:**

```bash
cd /home/keti_demo_machine/ros_ws/dm_ws/src/life_assist_dm
./scripts/fix_rqt_service_caller.sh
```

**수정 내용:**
- `/opt/ros/humble/lib/python3.10/site-packages/rqt_service_caller/service_caller_widget.py` 375번째 줄
- `%r` (repr, ASCII 강제) → `%s` (str, UTF-8 안전)

**수정 후:**
```python
# 기존 (버그)
qWarning('ServiceCaller.on_call_service_button_clicked(): request:\n%r' % (request))

# 수정 (해결)
qWarning('ServiceCaller.on_call_service_button_clicked(): request:\n%s' % str(request))
```

**장점:**
- ✅ 근본적인 해결 (한글 포함 요청도 완벽하게 처리)
- ✅ 환경 변수 설정 없이도 작동
- ✅ 자동 백업 생성 (복구 가능)

**참고:**
- ROS2 Jazzy 이상 버전에서는 이미 수정됨 (f-string 사용)
- 이 스크립트는 Humble 버전에서만 필요

---

### **방법 1: 자동 래퍼 스크립트 사용 (임시 우회) ⚠️**

`~/bin/rqt` 래퍼가 자동으로 환경 변수를 설정합니다:

```bash
# 현재 터미널에서 적용 (한 번만 실행)
source ~/.bashrc

# 그 후 그냥 rqt 실행하면 됨
rqt
```

**특징:**
- 자동으로 UTF-8 환경 변수 설정
- 일반 실행: `rqt`
- 원격 접속 시 안정화: `rqt --no-sandbox`

---

### **방법 2: 프로젝트 스크립트 사용**

```bash
cd /home/keti_demo_machine/ros_ws/dm_ws/src/life_assist_dm
./scripts/start_rqt.sh

# 원격 접속 시
./scripts/start_rqt.sh --no-sandbox
```

---

### **방법 3: 수동 환경 변수 설정**

터미널에서 다음 명령을 실행한 후 rqt를 실행:

```bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
rqt
```

---

### **방법 4: 대안 - ros2 service call CLI 사용**

`rqt_service_caller` 대신 터미널에서 직접 서비스를 호출:

```bash
# 직접 호출
ros2 service call /conversation life_assist_dm_msgs/srv/Conversation "{ask: '내가 먹는 약 뭐였지?'}"
```

**장점:** rqt GUI 출력 버그를 완전히 회피, UTF-8 인코딩 문제 없음

---

### **방법 5: GUI 안정화 옵션 (원격 접속 시)**

VNC, SSH X-forward 등 원격 접속 환경에서 rqt를 실행하는 경우:

```bash
rqt --no-sandbox
```

Qt 가속 관련 충돌도 함께 방지됩니다.

## 문제 진단

현재 환경 변수 상태를 확인하려면:
```bash
cd /home/keti_demo_machine/ros_ws/dm_ws/src/life_assist_dm
# 환경 변수 확인 (수동)
echo "LC_ALL=$LC_ALL"
echo "LANG=$LANG"
echo "PYTHONIOENCODING=$PYTHONIOENCODING"
```

## 📊 해결 방법 요약

| 문제 원인 | 해결 방법 | 효과 |
|---------|---------|------|
| **`%r` 포맷이 ASCII 강제** | **`fix_rqt_service_caller.sh` 실행** | **근본 해결 ⭐** |
| 한글 출력 시 ASCII 인코딩 오류 | `export LC_ALL=C.UTF-8` | 임시 우회 |
| Python 표준 출력 인코딩 문제 | `export PYTHONIOENCODING=utf-8` | 임시 우회 |
| rqt GUI 버그 | `rqt --no-sandbox` (원격 접속 시) | 안정화 |
| 임시 회피 | `ros2 service call` CLI로 대체 | 완전 회피 |

## ⚠️ 중요 사항

- **`LC_ALL`만 설정하는 것으로는 부족합니다**
- **`PYTHONIOENCODING`과 `PYTHONUTF8`도 반드시 설정해야 합니다**
- `C.UTF-8`이 `ko_KR.UTF-8`보다 더 범용적이고 안정적입니다
- 터미널을 새로 열면 환경 변수가 초기화되므로, `~/.bashrc`에 추가하거나 래퍼 스크립트를 사용하는 것이 권장됩니다

## 🔧 참고

### 버그 상세 정보
- **파일:** `/opt/ros/humble/lib/python3.10/site-packages/rqt_service_caller/service_caller_widget.py`
- **라인:** 375번째 줄
- **문제 코드:** `qWarning('... request:\n%r' % (request))`
- **원인:** `%r`은 `repr()`을 호출하여 ASCII 인코딩을 강제

### 해결 방법 비교
1. **근본 해결 (권장):** `fix_rqt_service_caller.sh` 실행 → 버그 자체를 수정
2. **임시 우회:** 환경 변수 설정 → 대부분의 경우 작동하지만 완벽하지 않음
3. **완전 회피:** `ros2 service call` CLI 사용 → rqt 사용 안 함

### ROS2 버전별 상황
- **ROS2 Humble:** 버그 존재 → `fix_rqt_service_caller.sh` 필요
- **ROS2 Jazzy 이상:** 이미 수정됨 (f-string 사용) → 수정 불필요

### 백업 및 복구
- 수정 스크립트는 자동으로 백업 파일을 생성합니다
- 복구가 필요한 경우: `sudo cp /opt/ros/humble/.../service_caller_widget.py.backup.* /opt/ros/humble/.../service_caller_widget.py`

