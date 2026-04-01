# Changelog

<details>
<summary>[v0.1.0] - 2026-03-24</summary>

> 제어 함수 분리 및 URDF 연동성 강화

### 새로운 기능
* **제어 함수 추가**:
  * `movex()` 함수 추가
* **정보 조회 API**:
  * `get_joint_info()`: 조인트 상태 및 정보 확인 가능
  * `get_end_effector_offset_tf()`: End-Effector의 오프셋 변환(Transformation) 정보 확인 가능
* **URDF 속성 반영**: URDF 파일 내에 정의된 `end-effector` 속성이 있으면 end-effector가 자동으로 설정됨

### 변경 사항
* **모드 설정 방식 변경**:
  * 기존 `STRICT` / `RELAX` 모드 플래그 방식 제거
  * 함수명을 통해 직관적으로 구분: `movel` (STRICT) / `movex` (RELAX)
* **데이터 타입 변경**:
  * `movej` 함수의 인수를 `std::array<double, 6>`에서 `vec<6>`로 변경
* **의존성 관리**:
  * `imgui` 라이브러리를 submodule 형태에서 프로젝트 소스 내 직접 포함 방식으로 변경
* ** 함수명 변경
  * set_joint_vlimit -> set_joint_velocity_limit
  * set_joint_vlimit_scale -> set_joint_velocity_limit_scale
  * set_end_effector_tf -> set_end_effector_offset
  * get_curr_jvel -> get_curr_joint_velocity
  * get_end_effector_tf -> get_end_effector_offset

### 예제 및 문서
* **EE 설정 예제**:
  * End-Effector 설정 시 회전(Rotation) 값을 적용하는 가이드 코드를 예제에 추가
</details>

---

<details>
<summary>[v0.0.1] - 2026-03-16</summary>

* **Initial Release**: 프로젝트 첫 공식 배포

</details>