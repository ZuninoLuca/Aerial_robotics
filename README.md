# Aerial robotics
This repository contains multiple resources related to the final project of the course "Aerial robotics" (taught by Professor D. Floreano), attended in the Spring semester of 2022 at EPFL. The project has been done in cooperation with Nay Abi Akl, Mariam Hassan and Yehya El Hassan.

The final project, called "CrazyPractical", consisted in implementing the software of the Crazyflie drone (see the picture below, source: [Bitcraze website](https://www.bitcraze.io/products/crazyflie-2-1/)) to make it capable of taking off from a pad, navigating through an arena while avoiding obstacles, locating a landing pad, safely landing, and taking off again to return to the initial position. Therefore, the project involved path planning, obstacle avoidance, and precise and robust localization.
![Crazyflie drone](/Images/Crazyflie.jpg)

The strategy used by the devised algorithm in order to safely land to the landing pad (first half of the mission) is highlighted in the following graph:
![Strategy 1](/Images/Strategy_1.png)

While the strategy used to take-off again, and return to the starting pad (secold half of the mission) is presented in the following graph:
![Strategy 2](/Images/Strategy_2.png)

The following resources are present in the repository:
- [Presentation slides](/CrazyPractical_Project_Slides.pdf), introducing the strategies and the approach followed to implement the code in the drone;
- [Demo video](/CrazyPractical_Project_Demo_Video.mp4), showcasing the capabilities of our code. In the video, the drone is able to complete the mission (taking off, avoiding obstacles, landing, taking off again and returning to the starting pad) in less then two minutes;
- [Complete source code](/CrazyPractical_Project_Source_Code/), useful to verify our proposed solution, and to reproduce the project.