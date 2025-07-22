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

### Team: 그머말
<프로젝트 요약>
* 수화물 낙하 및 이동 사고 예방 시스템
* 스마트 적재 가이드 시스템: Vision AI와 센서 융합으로 짐의 무게 배분 및 이동 모니터링 시스템
* 단순 모니터링을 넘어 사고를 사전 예방하며 안전한 운전 경험을 제공
  | Name | Role |
  |----|----|
  | 이건희 | Project lead, 프로젝트 기획 및 Back-end 구현 |
  | 나성심 | Project manager, Back-end 구현 및 시스템 아키텍처 설계 |
  | 신지혜 | UI design, 사용자 인터페이스를 정의 및 구현, Unity 활용한 Front-end 구현 |
  | 하성민 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 최현원 | Architect, 사용자 인터페이스 정의 및 CMS 구현. |
* Project Github : https://github.com/simeeeeee/Intelligence-Cargo-System
* 발표자료 : https://github.com/simeeeeee/Intelligence-Cargo-System/doc/slide.ppt

