---
title: "JetPack 6.2.2 Flash Troubleshooting: AMD USB 비호환과 chroot 해법"
date: 2026-03-19
tags: ["jetson", "nvidia", "edge-ai", "jetpack", "troubleshooting", "cuda", "tegrarcm", "amd", "chroot"]
categories: ["Embedded"]
summary: "Jetson AGX Orin을 플래시할 때 AMD 호스트에서 발생하는 tegrarcm_v2 USB write timeout의 근본 원인, 그리고 RAM이 부족한 Intel 노트북을 사용한 chroot 기반 우회법을 정리한다."
draft: false
---

## Problem

JetPack 6.2.2 (L4T R36.5.0)를 Jetson AGX Orin 64GB Developer Kit에 플래시하면 AMD B650 기반 호스트 PC에서 `tegrarcm_v2` 단계에서 `ERROR: might be timeout in USB write`로 실패한다. 이 현상은 호스트 OS 환경(Windows WSL2, Ubuntu Live Boot, Ubuntu 네이티브 설치)과 무관하게 발생한다.

```
[ 0.2574 ] Sending bct_br
[ 0.2576 ] ERROR: might be timeout in USB write.
Error: Return value 3
Command tegrarcm_v2 --new_session --chip 0x23 ...
```

## Background

목표 구성은 온디바이스로 동작하는 실시간 음성 채팅 데모([LlamaSpeak](https://www.jetson-ai-lab.com/archive/tutorial_llamaspeak.html))였다. 보드는 JetPack 5.1.3 (CUDA 11.4)이 미리 플래시된 상태로 도착했고, 텍스트 기반 LLM 추론(MLC를 통한 Llama 3 8B)은 이미 동작하고 있었다. 그런데 사용 가능한 모든 ASR 경로가 JetPack 6을 요구했다.

| ASR 옵션 | JP5.1.3에서의 실패 원인 |
|---|---|
| Riva 2.14.0 | NGC에서 모델 제거됨 (HTTP 403) |
| Riva 2.19.0 | CUDA 12 필요, `cudaError_t 35` |
| Riva 2.13.1 | `riva_init.sh` 내부 NGC CLI가 호스트 API 키를 무시함 |
| whisper_trt | JP6 전용 모듈, `ImportError` |

이 때문에 JetPack 6으로의 재플래시가 필수가 되었다.

## Environment

#### Host PC (AMD, 실패)
- AMD B650 AORUS ELITE V2
- RTX 4060 Ti
- USB 3.x 포트 (AMD USB 컨트롤러)

#### Host PC (Intel, 성공)
- Samsung NT930SBE 노트북
- Intel Core i5-8265U
- 8GB RAM, USB-C 전용 (3포트)

#### Target
- Jetson AGX Orin 64GB Developer Kit
- eMMC + 1TB NVMe
- 정품 USB-A to C 데이터 케이블

## Root Cause: AMD USB Controller Incompatibility

NVIDIA의 `tegrarcm_v2` 도구는 APX 리커버리 모드의 Jetson과 USB로 통신한다. AMD B650 USB 컨트롤러에서는 이 통신이 가장 첫 단계(`Sending bct_br`)에서 write timeout으로 실패한다.

AMD 호스트에서 다음 환경들을 테스트했고, 모두 동일한 에러를 냈다.

| 호스트 환경 | 결과 |
|---|---|
| Windows SDK Manager (WSL2) | USB timeout |
| Ubuntu 22.04 Live Boot (USB) | USB timeout |
| Ubuntu 22.04 네이티브 (외장 SSD) | USB timeout |
| Windows SDK Manager (Native WSL) | USB timeout |

시도한 완화책(전부 효과 없음):
- USB autosuspend 비활성화: `echo -1 > /sys/module/usbcore/parameters/autosuspend`
- 다른 USB 포트 (전면/후면)
- Zadig 드라이버 교체 (VBoxUSB → WinUSB)
- 다른 호스트 OS 환경

공통 분모는 AMD USB 컨트롤러였다. Intel 기반 호스트로 바꾸자 USB timeout이 완전히 사라졌고, `tegrarcm_v2`가 첫 시도에서 에러 없이 완료되었다.

## Partial Flash Consequences

근본 원인을 파악하기 전, AMD 호스트에서 `flash.sh`를 한 번 돌렸고 일부만 완료되었다. 그 결과는 다음과 같다.

- CLI 부팅 동작 (`tty2` 로그인 가능)
- 인터넷 연결 정상
- `nvidia` 커널 모듈 로드됨
- `dpkg`에 L4T 36.5.0 패키지 설치 확인
- GUI 완전 고장: NVIDIA 로고 이후 검은 화면
- `/dev/nvhost*` 없음: GPU 디바이스 노드가 하나도 없음
- Xorg: `eglInitialize() failed`, `no screens found`
- Weston: `NvRmMemInitNvmap failed with Permission denied`

근본 원인은 USB 전송 손상으로 인한 불완전한 Device Tree Blob (DTB)이었다. 이것은 소프트웨어로는 복구할 수 없다. xorg.conf를 아무리 편집하거나, 패키지를 재설치하거나, ldconfig를 설정해도 누락된 device tree 항목은 고쳐지지 않는다. 신뢰할 수 있는 USB 전송으로 깨끗하게 재플래시하는 것이 유일한 방법이다.

`flash.sh`가 성공을 보고해도 깨끗한 플래시가 보장되지는 않는다. USB 전송 에러가 조용히 펌웨어/DTB를 손상시킬 수 있다. 플래시 후에는 항상 `/dev/nvhost*` 존재 여부를 확인하라.

## Solution: Intel Host + chroot

Intel 노트북은 RAM이 8GB뿐이었고 BIOS에서 외장 SSD로 부팅할 수 없었다. 이 때문에 해결해야 할 문제가 두 개 더 생겼다.

1. RAM 고갈: `flash.sh`는 메모리에 약 6GB짜리 `system.img`를 빌드한다. Live Boot에서 Ubuntu GUI가 돌아가는 상태에서는 8GB RAM으로는 부족해서 rootfs에 대한 `sed` 작업 중 segfault가 난다.
2. SQUASHFS 손상: Ubuntu Live Boot는 USB 스틱의 SQUASHFS 파일시스템에서 시스템 바이너리를 로드한다. 무거운 I/O 부하에서 USB 스틱이 열화되면서 read error가 발생했고, `sed`, `bash` 등을 로드할 때 segfault가 났다.

### Failed Intermediate Approaches

| 접근 | 실패 |
|---|---|
| Live Boot + GUI | Segfault, RAM 고갈 |
| Live Boot + GUI + SSD에 4GB swap | Segfault, GUI가 여전히 RAM을 너무 많이 씀 |
| Live Boot + CLI (`systemd.unit=multi-user.target`) | Segfault, 죽어가는 USB 스틱의 SQUASHFS read error |

### Working Approach: CLI Boot + chroot

해법은 USB 스틱을 최소한의 부트스트랩 용도로만 쓰고, 외장 SSD에 설치된 완전한 Ubuntu로 `chroot`하는 것이었다. chroot 안에서는 모든 바이너리가 SSD에서 로드되므로 USB 스틱의 SQUASHFS를 완전히 우회한다.

#### Step 1: Boot into CLI mode

GRUB 메뉴에서 `e`를 눌러 부트 항목을 편집한다. `linux` 줄 끝에 `systemd.unit=multi-user.target`을 추가한다.

```
linux /casper/vmlinuz file=/cdrom/preseed/ubuntu.seed maybe-ubiquity quiet splash --- systemd.unit=multi-user.target
```

`F10`을 눌러 부팅한다. `ubuntu`(비밀번호 없음)로 로그인한다.

#### Step 2: Mount SSD and enter chroot

```bash
sudo -i
mount -o rw /dev/sda1 /mnt
mount --bind /dev /mnt/dev
mount --bind /proc /mnt/proc
mount --bind /sys /mnt/sys
mount --bind /run /mnt/run
chroot /mnt /bin/bash
export PATH=/usr/bin:/usr/sbin:/bin:/sbin
```

#### Step 3: Configure networking

WiFi는 chroot 바깥에서 설정해야 한다 (NetworkManager는 호스트에서 돈다).

```bash
# Exit chroot
exit

# Connect WiFi
nmcli dev wifi connect <BSSID> password "<password>"

# Re-enter chroot
chroot /mnt /bin/bash
export PATH=/usr/bin:/usr/sbin:/bin:/sbin
echo "nameserver 8.8.8.8" > /etc/resolv.conf
```

#### Step 4: Install dependencies

```bash
apt-get update && apt-get install -y libxml2-utils binutils
```

둘 다 `flash.sh`가 요구한다. XML 검증용 `xmllint`, 바이너리 검사용 `strings`다.

#### Step 5: Flash

Jetson을 리커버리 모드로 둔다(전원 분리 → 가운데 Force Recovery 버튼 누른 채 유지 → 전원 연결 → 10초 대기 → 버튼 놓기). USB 케이블을 연결한 뒤 실행한다.

```bash
cd /home/<user>/nvidia/nvidia_sdk/JetPack_6.2.2_Linux_JETSON_AGX_ORIN_TARGETS/Linux_for_Tegra/
./flash.sh jetson-agx-orin-devkit internal
```

`internal` 플래그는 eMMC를 타깃으로 삼아 NVMe의 데이터를 보존한다.

### Result

```
[ 913.6337 ] Flashing completed
[ 913.6338 ] Coldbooting the device
*** The target generic has been flashed successfully. ***
```

전체 60개 파티션이 100%로 기록되었다. Jetson을 재부팅한 뒤:
- Ubuntu GUI: 정상
- `/dev/nvhost*`: 15개 이상의 디바이스 노드 존재
- Docker 내부의 `nvidia-smi`: Driver 540.5.0, CUDA 12.6

## Post-Flash Setup

```bash
# Install JetPack SDK components
sudo apt update && sudo apt install -y nvidia-jetpack

# Verify CUDA
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version  # Should show CUDA 12.6

# Mount NVMe (if applicable)
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1p1 /mnt/nvme
echo "/dev/nvme0n1p1 /mnt/nvme ext4 defaults 0 2" | sudo tee -a /etc/fstab

# Docker setup with NVMe storage
sudo apt install -y docker.io nvidia-container-toolkit
sudo tee /etc/docker/daemon.json << 'EOF'
{
  "data-root": "/mnt/nvme/docker",
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
sudo systemctl restart docker

# Verify Docker + GPU
sudo docker run --rm --runtime nvidia nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## Summary

| 항목 | 내용 |
|---|---|
| 근본 원인 | AMD USB 컨트롤러가 `tegrarcm_v2`와 비호환 |
| 해법 | Intel USB 호스트 사용 |
| RAM 제약 | Live Boot의 SQUASHFS 의존을 피하려 SSD 기반 Ubuntu로 chroot |
| CLI 부팅 | RAM 사용량을 8GB 이하로 유지하기 위해 필요 |
| 플래시 타깃 | NVMe 데이터 보존을 위한 `internal` (eMMC) |
| 검증 | `flash.sh` 종료 상태가 아니라 `/dev/nvhost*`를 확인 |

## References

- [NVIDIA SDK Manager Install Guide](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)
- [JetPack Installation Guide](https://docs.nvidia.com/jetson/jetpack/install-setup/index.html)
- [LlamaSpeak Tutorial](https://www.jetson-ai-lab.com/archive/tutorial_llamaspeak.html)
- [NVIDIA Developer Forum, Jetson AGX Orin](https://forums.developer.nvidia.com/c/robotics-edge-computing/jetson-systems/jetson-agx-orin/486)
- [Forum Post: USB Timeout on Flash](https://forums.developer.nvidia.com/t/jetson-agx-orin-64gb-usb-timeout-on-flash-gui-broken-after-flash-sh-sdk-manager-no-sdks-on-windows/363988)
