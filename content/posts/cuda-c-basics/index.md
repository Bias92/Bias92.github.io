---
title: "01 CUDA C Basics"
date: 2026-05-22
draft: false
tags: ["CUDA", "GPU Programming", "Parallel Programming", "Video Notes"]
categories: ["CUDA"]
series: ["CUDA C"]
summary: "cat blue의 01 CUDA C Basics 영상 정리. CUDA C의 기본 실행 모델, host/device 코드 분리, kernel launch, thread/block/grid 구조를 정리한다."
---

> Source: [01 CUDA C Basics](https://youtu.be/OsK8YFHTtNs)

### What is CUDA?

CUDA(Compute Unified Device Architecture)는 NVIDIA의 병렬 컴퓨팅 플랫폼이자 프로그래밍 모델(SIMT)이다. GPU를 그래픽 전용이 아닌 범용 연산 장치(GPGPU)로 쓸 수 있게 해주는 C/C++ 확장이며, 2007년 처음 공개되어 지금까지 딥러닝 인프라의 사실상 표준이 됐다.

CUDA라는 것은 정확히 무엇일까? 그리고 현용 모델들을 구축할 때 사용되는 PyTorch와 무슨 관계인 걸까? 알아보기 전에 계층화된 피라미드부터 알아가는 것이 좋을 것 같다.

## ㅉ

![VEGAS GPU 가속 설정](./images/vegas.png)
