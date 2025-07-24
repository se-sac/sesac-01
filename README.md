# 청년취업사관학교 인텔교육 1기
## Clone code 

```shell
git clone --recurse-submodules https://github.com/se-sac/sesac-01.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정하여 활용

3. 팀 별로 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함 (강사의 가이드에 따라 진행)

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

### 제출현황

### Team: 머지?해요

**< Summary >**
> 🚀 **M.A.R.S. (Make Agent Really Sexy)**
> 😎 **Fun하고 Cool하고 Sexy한 Multi-Agent System**

사용자의 요청을 기반으로, 그와 걸맞은 다양한 역할의 AI 에이전트 / 챗봇을 자동으로 양성하는 에이전트 / 챗봇 프로젝트입니다. 

**Members**
| 이름 | 역할 | 설명 | 
| - | - | - |
| 🔥 이주용 | Project Manager | 경애하고 친애하는 위대한 령도자 이주용동지 |
| 🛠️ 김지선 | Developer | "망치와 모루" 전술의 "망치와 모루" 담당 |
| 🐾 김지현 | Archiver & Speaker | PAGA(빠가): 프로젝트를 기록으로 다시 위대하게 |
| 🏹 김태윤 | QA Tester | 망하면 한양도성박물관 20바퀴 완주 예정 |
| 🕸️ 정의한 | UI Designer | 다정한 웹 · 앱 개발자 Spiderman |
* 프로젝트 링크: https://github.com/SeSac01/MARS.git
* 발표 자료: https://github.com/SeSac01/MARS/doc/slide.ppt

### Team: 그머말(그게 머선 말이야)

**< Summary >**
> 🤖 **Park Intel Sentinel**
> 🚘 **자동 주차장 감시 경비봇 시스템**

**Members**
  | Name | Role |
  |----|----|
  | 🐹 이건희 | Project lead, 프로젝트 기획 및 Back-end 구현 |
  | 🐑 나성심 | Project manager, Back-end 구현 및 시스템 아키텍처 설계 |
  | 🦤 신지혜 | UI design, 사용자 인터페이스를 정의 및 구현, Unity 활용한 Front-end 구현 |
  | 🐨 하성민 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 🐻 최현원 | Architect, 사용자 인터페이스 정의 및 CMS 구현. |
* Project Github : https://github.com/simeeeeee/park-intel-sentinel
* 발표자료 : https://github.com/simeeeeee/park-intel-sentinel/doc/slide.ppt

### Team: AIinSai

<프로젝트 요약>
* LLM을 활용한 Dungeon and Dragon 장르의 TRPG게임 
* LLM의 생성한 문장을 바탕으로 몰입감 있는 게임 경험 제공 
* Langchain, Langgraph를 활용하여 LLM이 개연성을 유지할 수 있도록 유도 

**Members**
| Name | Role | 
| ---- | ---- |
| 김주원 | Project lead, Main Developer | 
| 박찬영 | 기획자, Sub Developer | 

* Project Github : https://github.com/joowining/LLM_D-D
* 발표 자료 : https://github.com/joowining/LLM_D-D

### Team: Connection


 **프로젝트명: Road Vision**

> 이 프로젝트는 CNN(합성곱 신경망), GRU(게이트 순환 유닛), MLP(다층 퍼셉트론)을 결합한 딥러닝 아키텍처를 임베디드 환경에서 활용하여 도로 노면 상태(정상, 젖음, 빙판, 파손) 실시간 감지를 통해 데이터 공유·알림 시스템을 구현하는 것이 목표입니다.


**Members**
  | Name | Role |
  |----|----|
  | 성세빈 | Project lead, 프로젝트 전체 기획,  데이터 전처리 및 분류 알고리즘 개발 담당 |
  | 김예진 | Project manager, 문서화 및 커뮤니케이션 총괄, 딥러닝 모델 구조 설계 및 학습 |
  | 정소령 | System Architect, 모델 아키텍처 설계, 임베디드 시스템 구조 및 통신 구성 담당 |

* Project Github : https://github.com/SeBin7/Road_Vision
* 발표자료 : https://github.com/SeBin7/Road_Vision/blob/main/doc/Road_Vision.pptx