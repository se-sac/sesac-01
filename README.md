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

### Team: 커넥션
* 차량 탑재 YOLO와 RNN 기반의 노면이상탐지시스템을 통해 결빙, 젖은 노면, crack 등의 실시간 감지 및 V2X 통신 기반 차량 간 정보공유를 통한 공공데이터 구축을 최종 목표로 한다.
* Members
  | Name | Role |
  |----|----|
  | 성세빈 | Project lead, 프로젝트를 총괄 및 기획  |
  | 김예진 | Vision AI Engineer, 모델 학습 및 추론, 노면 탐지 알고리즘 구현  |
  | 정소령 | Data & Evaluation, 데이터셋 구축, 전처리 및 성능 평가 담당  |
* Project Github : https://github.com/SeBin7/Road_Vision.git
* 발표자료 : https://github.com/SeBin7/Road_Vision/doc/RoadVision.ppt

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
