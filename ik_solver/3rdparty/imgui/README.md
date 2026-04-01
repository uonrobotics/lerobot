# Imgui
* ImGui와 ImPlot를 쉽게 셋업하기 위해 만들었습니다.
* 간단한 래핑 클래스로 쉽게 사용할 수 있습니다.
* ImPlot과 ImPlot3D가 포함되어 있습니다.

![screenshot1.png](screenshot/screenshot1.png)
![screenshot2.png](screenshot/screenshot2.png)

<br>

## 프로젝트 구조
```text
workspace/
├── 3rdparty/
│   └── imgui/
├── main.cpp
└── CMakeLists.txt
```

<br>


## 종속성
* GLFW3 설치
```shell
  sudo apt install libglfw3 libglfw3-dev
```

<br>

## 설치
* 예시는 3rdparty디렉토리에서 git clone을 실행합니다
```shell
  mkdir 3rdparty && cd 3rdparty
  git clone https://github.com/code2j/lib_ImGui.git
```

<br>

## 프로젝트에 추가
* 프로젝트 메인 CMakeLists.txt에 다음을 추가하세요
```cmake
add_subdirectory(3rdparty/imgui)

target_link_libraries(your_target PRIVATE imgui)
```

```c++
#include "imgui.h"
#include <iostream>
#include <string>

int main() 
{
    // 컨텍스트 생성 및 시작
    std::string title = "DEMO";
    ImVec2 size = ImVec2(1280, 720);
    ImGui::start(title, size);
    
    // 메인 루프
    while (ImGui::isRunning()) {
        ImGui::context([&]() {
            /*
            이곳에 ImGui를 사용하는 어떠한 코드든 작성하세요. 
            */
            ImGui::ShowDemoWindow();
        });
    }
    
    // 리소스 정리
    ImGui::stop();
    
    return 0;
}
```

<br>

## 이 프로젝트는 다음 오픈소스 라이브러리를 사용합니다
- **ImGui** (MIT License): Dear ImGui 라이브러리 (https://github.com/ocornut/imgui)
- **ImPlot** (MIT License): ImGui 플로팅 라이브러리 (https://github.com/epezent/implot)
- **ImPlot3D** (MIT License): ImGui 3D 플로팅 라이브러리 (https://github.com/brenocq/implot3d)
- **ImCoolBar** (MIT License): ImGui 툴바 컴포넌트 (https://github.com/aiekick/ImCoolBar)

각 라이브러리의 라이선스는 해당 라이브러리의 LICENSE 파일을 참조하세요. 이 프로젝트의 전체 라이선스는 [LICENSE.txt](LICENSE.txt)를 확인하세요.

프로젝트에 기여하거나 이슈를 제기하려면 GitHub 리포지토리를 방문하세요.