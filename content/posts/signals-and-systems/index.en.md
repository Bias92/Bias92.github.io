---
title: "Signals and Systems"
date: 2026-09-02
lastmod: 2026-09-02
tags: ["EE", "Signals and Systems"]
categories: ["EE"]
series: ["EE"]
math: true
summary: "Lecture notes for Signals and Systems based on Soliman & Srinath: the definition of a signal, continuity of a function, the value at a discontinuity, the rectangular pulse and pulse train, and the distinction between continuous-time and discrete-time."
draft: false
---

> Reference: Samir S. Soliman, Mandyam D. Srinath, *Continuous and Discrete Signals and Systems*, 2nd ed., Prentice Hall, 1998. Chapter 1, Representing Signals.

This course covers continuous-time signals with a single independent variable, time, and leaves discrete-time signals to the DSP[^dsp] course. The material below corresponds to Sections 1.1 and 1.2 of the textbook and to lecture slides[^slides] 2 through 6.

## Definition of a signal

A signal is a detectable physical quantity that carries information. The definition has three parts.

1. It is a physical quantity or a variable that represents one. The amplitude of a voice, the brightness of a screen pixel, and the current in a circuit all qualify.
2. It is detectable[^detectable]. The receiving side must be able to measure the quantity.
3. It carries information. A quantity that can be detected but carries no information is not treated as a signal.

Mathematically a signal is written as a function of one or more independent variables (the inputs of the function). The number of independent variables[^variables] depends on the signal.

| Signal | Notation | Independent variables |
|---|---|---|
| Voice, circuit current $i(t)$, voltage $v(t)$ | $A(t)$ | Time $t$ only |
| Image | $p(x, y)$ | Two coordinates $x, y$ |
| Electric field | $E(x, y, z, t)$ | Three spatial coordinates and time |

This course deals only with signals $x(t)$ whose single independent variable is time $t$. An image $p(x, y)$ belongs to an image processing course, and a video is $p(x, y, t)$ with time added.

## Continuous-time and discrete-time signals

A continuous-time signal is a signal whose independent variable is defined over an entire interval of real numbers. In the definition, the word continuous modifies the independent variable, not the function value[^axis]. For a radio signal $v(t)$ the independent variable is time $t$, and for atmospheric pressure $p(h)$ it is altitude $h$. Both independent variables sweep the whole real line, so both are continuous-time signals. An independent variable other than time, such as altitude, falls in the same class as long as it is continuous.

A discrete-time signal is a signal whose independent variable takes only discrete values. Taking the values of a continuous-time signal $x(t)$ only at $t = kT_s$ gives $x(kT_s)$. $T_s$[^sampling] is a fixed positive real number and $k$ is an integer ($0, \pm 1, \pm 2, \dots$). With $T_s = 2$ the signal is defined only at $t = 0, \pm 2, \pm 4, \dots$. In the figure below the cyan curve is $x(t)$, which has a value at every real $t$, and the magenta dots are $x(kT_s)$, which has values only at $t = kT_s$. To tell the two apart, look at where the points sit on the $t$ axis rather than at the height of the function values.

![Continuous-time and discrete-time signals](images/fig2-sampling.svg)

## Continuity of a function

A signal is a function, so the continuity of a signal follows the continuity of a function. $x(t)$ is continuous at $t = t_1$ when two conditions hold in order.

1. The left-hand limit $x(t_1^-)$ and the right-hand limit $x(t_1^+)$ are equal. The limit at $t_1$ then exists.
2. That limit equals the function value $x(t_1)$.

A signal continuous at every $t$ is a continuous signal. A signal with a finite number of discontinuities is piecewise continuous[^piecewise]. A signal with infinitely many discontinuities is not piecewise continuous.

## Value at a discontinuity

Ordinary mathematics leaves the function value at a discontinuity undefined. This textbook defines the value at a discontinuity $t_1$ as the average[^average] of the left-hand and right-hand limits.

$$
x(t_1) = \frac{1}{2}\left[x(t_1^+) + x(t_1^-)\right]
$$

The unit step function $u(t)$ is 0 for $t < 0$ and 1 for $t > 0$. At $t = 0$ the left-hand limit 0 and the right-hand limit 1 differ, so the limit does not exist, and the signal is piecewise continuous rather than continuous. Applying the average definition gives $u(0) = 1/2$. In the left panel of the figure below, the two open circles are the left-hand limit $a$ and the right-hand limit $b$, and the filled dot is the value $(a + b)/2$ defined by the textbook. Applying the same rule in the right panel with $a = 0$ and $b = 1$ gives $u(0) = 1/2$.

![Value at a discontinuity and the unit step function](images/fig1-discontinuity.svg)

As an example, $x(t) = 2u(t - 1) - u(t - 3)$ is 0 for $t < 1$, 2 for $1 < t < 3$, and 1 for $t > 3$. The discontinuities are at $t = 1, 3$, with $x(1) = (0 + 2)/2 = 1$ and $x(3) = (2 + 1)/2 = 3/2$. With two discontinuities the signal is piecewise continuous.

## Rectangular pulse and pulse train

The rectangular pulse $\mathrm{rect}(t/\tau)$ is 1 for $|t| < \tau/2$ and 0 for $|t| > \tau/2$.

$$
\mathrm{rect}(t/\tau) =
\begin{cases}
1, & |t| < \tau/2 \\
0, & |t| > \tau/2
\end{cases}
$$

The height is fixed at 1, and the parameter $\tau$[^tau] sets the width of the base. With $\tau = 10^8$ the pulse is $10^8$ wide, with $\tau = 10^{-7}$ it is $10^{-7}$ wide, and the height is 1 in both cases. The discontinuities are the two points $t = \pm\tau/2$, so the pulse is piecewise continuous, and since $t$ is defined over the whole real line it is a continuous-time signal.

Shifting $\mathrm{rect}(t/\tau)$ in time by a fixed spacing and adding the copies gives a pulse train. A train of pulses of width 1 repeated every 2 units is written as follows, with $n$ ranging over all integers.

$$
x(t) = \sum_{n=-\infty}^{\infty} \mathrm{rect}(t - 2n)
$$

The discontinuities at $t = 2n \pm 1/2$ are infinite in number[^train], so the train is neither continuous nor piecewise continuous. Since $t$ is defined over the whole real line, it is a continuous-time signal. Cutting the train to a finite number of pulses leaves a finite number of discontinuities, which makes it piecewise continuous.

## Continuous versus continuous-time

The two terms describe properties on different axes.

| Term | Axis | Criterion |
|---|---|---|
| continuous / piecewise continuous | Vertical axis (function value $x$) | Number of points where the limit conditions fail |
| continuous-time / discrete-time | Horizontal axis (independent variable $t$) | Whether $t$ is defined over a whole real interval or only at $t = kT_s$ |

| Signal | Function-value property | Independent-variable property |
|---|---|---|
| $\sin t$ | continuous | continuous-time |
| Rectangular pulse, $u(t)$ | piecewise continuous | continuous-time |
| Infinite pulse train | Infinitely many discontinuities, not piecewise continuous | continuous-time |
| $x(kT_s)$ | Function-value property not considered | discrete-time |

However many discontinuities there are, a signal whose $t$ axis is the whole real line is a continuous-time signal. Every signal in this course is continuous-time, and the only question within that class is whether it is continuous or piecewise continuous.

## Exercise

For $x(t) = 3\,\mathrm{rect}\!\left(\frac{t - 4}{2}\right)$, find the following.

(a) The interval of $t$ where the value is 3
(b) The discontinuities
(c) The value at each discontinuity under the textbook definition
(d) All integers $k$ for which $x(kT_s)$ is nonzero when sampled with $T_s = 1$

Solution. From the definition of $\mathrm{rect}(t/\tau)$, $\tau = 2$ and $t$ is replaced by $t - 4$, so the value is 3 where $|t - 4| < 1$, that is, for $3 < t < 5$. The discontinuities are at $t = 3, 5$. At both points the left-hand and right-hand limits are 0 and 3, so $x(3) = x(5) = (0 + 3)/2 = 3/2$. With $T_s = 1$ we have $t = k$, and the values at $k = 3, 4, 5$ are $3/2, 3, 3/2$ respectively, with 0 at every other $k$.

[^dsp]: Digital Signal Processing. Lecture slide 6 defers discrete-time signals with "Covered intensively in Digital Signal Processing (DSP) courses".

[^slides]: Slide numbers are the printed numbers in the top-right corner of the lecture slides "1. Representing Signals" (Seung-Chan Lim, School of Electronics and Electrical Engineering, Hongik University). Timestamps `[MM:SS]` refer to the transcript of the lecture recording from 2026-09-02.

[^detectable]: Lecture `[04:19]`. The instructor glossed detectable as "can be detected, can be recognized" and used the example of students' ears detecting the instructor's voice in the classroom.

[^variables]: Lecture `[09:11]`, `[10:45]`. The amplitude of a voice changing over time was modeled as $A(t)$, and a pixel value determined by two coordinates as $p(x, y)$. In electromagnetics an electric field has four independent variables, three spatial coordinates plus time.

[^axis]: Lecture `[26:38]`, `[37:56]`. To separate continuous-time from discrete-time, look at the independent-variable axis rather than the function value. No matter how many discontinuities the function value has, the signal is continuous-time if $t$ is defined over a whole interval.

[^sampling]: Slide 6 only describes $T_s$ as "a fixed positive real number". In DSP it is called the sampling period, and its reciprocal $f_s = 1/T_s$ is the sampling frequency. At `[39:36]` the instructor set $T_s = 2$ and $k = 0, \pm 1, \pm 2$ to produce $t = 0, \pm 2, \pm 4$.

[^piecewise]: Slide 3 reads "piecewise continuous if it has only finite discontinuities". At `[36:06]` the instructor distinguished a train of finitely many pulses, which is piecewise continuous, from an infinitely extended train, which has infinitely many discontinuities and is not.

[^average]: Slide 3, lecture `[24:31]`. The value that ordinary mathematics leaves undefined is fixed by the textbook as the average of the two one-sided limits. The instructor explained that this single convention lets the textbook account for many of its later results.

[^tau]: Lecture `[31:02]`. With $\tau = 10^8$ the base extends to $\pm 5 \times 10^7$, and with $10^{-7}$ the pulse becomes very narrow. The height stays 1 in both cases.

[^train]: Slide 5 describes the pulse train as "Continuous at all $t$ except at $t = 0, \pm 1, \pm 2, \cdots$". At `[36:06]` the instructor explained, using the independent-variable axis, why the train is a continuous-time signal despite its infinitely many discontinuities.
