#include <iostream>
#include <chrono>
#include <thread>
#include <memory>
#include <functional>
#include "../include/doosan_controller.h"

#define SLEEP(x) std::this_thread::sleep_for(std::chrono::milliseconds(x))


using namespace std;



int main() {
    std::unique_ptr<DoosanController> robot = std::make_unique<DoosanController>("192.168.1.30");

    cout << "로봇 연결함" << endl;
    robot->connect();
    robot->stop();

    SLEEP(1000);

    cout << "로봇 서보 ON" << endl;
    robot->servo_on();

    SLEEP(3000);

    cout << "로봇 Move Idle" << endl;
    float idle[6] = {90.0, -25.0, 120.0, 9.0, 50.0, 0.0};
    robot->movej(idle, 3.0);

    SLEEP(3000);
    cout << "로봇 RT Start" << endl;
    robot->start_rt(idle);


    while(1) {
        cout << "loop" << endl;
        auto joint = robot->get_current_joint();
        for (const float& joint1 : joint)
            cout << joint1 << ", ";
        cout << endl;
        SLEEP(1000);
    }

    return 0;
}